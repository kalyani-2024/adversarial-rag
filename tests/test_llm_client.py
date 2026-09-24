from types import SimpleNamespace

import groq
import httpx
import pytest

from app.core.config import Settings
from app.core.errors import LLMConfigError, LLMRateLimitError
from app.core.llm import GroqLLM


def _rate_limit(retry_after: str = "1") -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.test/v1/chat/completions")
    return groq.RateLimitError("rate limited", response=httpx.Response(429, headers={"retry-after": retry_after}, request=request), body=None)


def _ok():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                           usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def create(self, **kwargs):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(settings, outcomes, monkeypatch):
    llm = GroqLLM(settings)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(outcomes)))
    sleeps = []
    monkeypatch.setattr("app.core.llm.time.sleep", lambda s: sleeps.append(s))
    return llm, sleeps


def test_retries_on_rate_limit_and_records_throttle(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="k", llm_max_retries=3)
    llm, sleeps = _client(settings, [_rate_limit("2"), _rate_limit("1"), _ok()], monkeypatch)
    resp = llm.complete([{"role": "user", "content": "x"}], purpose="t")
    assert resp.text == "hi" and resp.attempts == 3
    assert len(sleeps) == 2 and sleeps[0] >= 2.0  # honors Retry-After
    assert resp.throttle_ms == pytest.approx(sum(sleeps) * 1000)


def test_gives_up_after_max_retries(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="k", llm_max_retries=1)
    llm, _ = _client(settings, [_rate_limit(), _rate_limit()], monkeypatch)
    with pytest.raises(LLMRateLimitError):
        llm.complete([{"role": "user", "content": "x"}], purpose="t")


def test_wait_budget_caps_retries(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="k", llm_max_retries=5, llm_max_wait_s=1.0)
    llm, sleeps = _client(settings, [_rate_limit("30"), _ok()], monkeypatch)
    with pytest.raises(LLMRateLimitError):
        llm.complete([{"role": "user", "content": "x"}], purpose="t")
    assert sleeps == []  # a 30 s Retry-After exceeds the 1 s budget: fail fast instead of hanging


def test_missing_api_key():
    with pytest.raises(LLMConfigError):
        GroqLLM(Settings(_env_file=None, groq_api_key=None)).complete([], purpose="t")


def _chunk(content=None, usage=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if (content or reasoning) else [], usage=usage)


def test_stream_forwards_content_only_and_reads_usage(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="k")
    chunks = [_chunk(reasoning="thinking…"), _chunk("Hello "), _chunk("world [1]."),
              _chunk(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4))]
    llm, _ = _client(settings, [iter(chunks)], monkeypatch)
    seen = []
    resp = llm.stream([{"role": "user", "content": "x"}], purpose="generate", on_token=seen.append)
    assert seen == ["Hello ", "world [1]."] and resp.text == "Hello world [1]."
    assert (resp.prompt_tokens, resp.completion_tokens) == (11, 4)


def test_stream_retries_only_when_opening(monkeypatch):
    settings = Settings(_env_file=None, groq_api_key="k", llm_max_retries=2)
    llm, sleeps = _client(settings, [_rate_limit("1"), iter([_chunk("ok")])], monkeypatch)
    resp = llm.stream([{"role": "user", "content": "x"}], purpose="generate", on_token=lambda t: None)
    assert resp.text == "ok" and resp.attempts == 2 and len(sleeps) == 1
