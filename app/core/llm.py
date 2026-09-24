"""Thin, testable LLM client abstraction.

The pipeline depends on the `LLMClient` protocol, not on Groq directly, so:
  * tests inject a scripted fake (no network, deterministic),
  * the provider can be swapped (OpenAI, Anthropic, vLLM...) by writing one class.

`GroqLLM` normalizes provider exceptions into our domain errors so callers can
distinguish "rate limited" from "timed out" from "misconfigured".
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import (
    LLMConfigError,
    LLMError,
    LLMOutputError,
    LLMRateLimitError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

Message = dict[str, str]
T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    # Time spent waiting between retries (rate limits / transient errors). Kept
    # separate so latency can be analysed with and without provider throttling.
    throttle_ms: float = 0.0
    attempts: int = 1


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        purpose: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse: ...


class GroqLLM:
    """Groq chat-completions client with timeouts, retries and error mapping."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = None  # created lazily so the app can boot without a key

    def _get_client(self):
        if self._client is None:
            key = self._settings.groq_api_key
            if key is None or not key.get_secret_value().strip():
                raise LLMConfigError("GROQ_API_KEY is not set. Add it to your environment or .env file.")
            import groq

            self._client = groq.Groq(
                api_key=key.get_secret_value(),
                timeout=self._settings.llm_timeout_s,
                max_retries=0,  # we retry ourselves so throttle time is measurable (see complete())
            )
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        purpose: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        import groq

        client = self._get_client()
        model_name = model or self._settings.llm_model
        kwargs: dict = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._settings.generation_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self._model_specific_params(model_name))

        start = time.perf_counter()
        throttle_s = 0.0
        max_retries = self._settings.llm_max_retries
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except (groq.RateLimitError, groq.APITimeoutError, groq.APIConnectionError, groq.InternalServerError) as exc:
                wait = _backoff_seconds(exc, attempt)
                if attempt == max_retries or throttle_s + wait > self._settings.llm_max_wait_s:
                    raise _map_error(exc, purpose) from exc
                logger.warning("llm call retry", extra={"purpose": purpose, "model": model_name, "attempt": attempt + 1,
                                                        "wait_s": round(wait, 2), "error_type": type(exc).__name__})
                time.sleep(wait)
                throttle_s += wait
            except groq.APIError as exc:  # auth errors and other non-retryable 4xx
                raise _map_error(exc, purpose) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            throttle_ms=throttle_s * 1000,
            attempts=attempt + 1,
        )

    def _model_specific_params(self, model_name: str) -> dict:
        """Provider quirks for reasoning models, kept out of the call sites."""
        if model_name.startswith("openai/gpt-oss") and self._settings.llm_reasoning_effort:
            return {"reasoning_effort": self._settings.llm_reasoning_effort}
        if model_name.startswith("qwen/"):
            return {"reasoning_format": "hidden"}  # never leak <think> blocks into outputs
        return {}


def _backoff_seconds(exc: Exception, attempt: int) -> float:
    """Honor the provider's Retry-After header when present, else exponential backoff with jitter."""
    response = getattr(exc, "response", None)
    retry_after = response.headers.get("retry-after") if response is not None else None
    if retry_after:
        try:
            return min(float(retry_after), 60.0) + random.uniform(0, 0.25)
        except ValueError:
            pass
    return min(0.5 * 2**attempt, 20.0) + random.uniform(0, 0.25)


def _map_error(exc: Exception, purpose: str) -> LLMError:
    import groq

    if isinstance(exc, groq.RateLimitError):
        return LLMRateLimitError(f"LLM rate limit hit during '{purpose}'.")
    if isinstance(exc, groq.APITimeoutError):
        return LLMTimeoutError(f"LLM timed out during '{purpose}'.")
    if isinstance(exc, groq.AuthenticationError):
        return LLMConfigError("LLM authentication failed; check GROQ_API_KEY.")
    return LLMError(f"LLM call failed during '{purpose}': {type(exc).__name__}")


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_model(text: str, model_cls: type[T]) -> T:
    """Parse an LLM reply into a Pydantic model.

    Tolerates code fences / surrounding prose by extracting the outermost
    `{...}` block. Raises `LLMOutputError` if nothing valid can be recovered.
    """
    candidate = text.strip()
    try:
        return model_cls.model_validate(json.loads(candidate))
    except (json.JSONDecodeError, ValidationError):
        pass
    match = _JSON_OBJECT_RE.search(candidate)
    if match:
        try:
            return model_cls.model_validate(json.loads(match.group(0)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LLMOutputError(f"LLM returned invalid {model_cls.__name__}: {exc}") from exc
    raise LLMOutputError(f"LLM returned no JSON object for {model_cls.__name__}.")
