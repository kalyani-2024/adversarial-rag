"""Centralized configuration.

Every tunable lives here and can be overridden with an environment variable
(or a `.env` file). Names map 1:1 to upper-case env vars, e.g. `final_top_k`
-> `FINAL_TOP_K`. Secrets are held as `SecretStr` so they never end up in logs
or `repr()` output.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider -------------------------------------------------------
    groq_api_key: SecretStr | None = None
    llm_model: str = "openai/gpt-oss-120b"
    # Judge/critic use a different model family to reduce self-preference bias.
    judge_model: str | None = "qwen/qwen3.8-27b"
    # Critic on a third model: spreads load across per-model rate-limit buckets
    # (Groq free tier: 8k tokens/min per model) and adds another independent reviewer.
    critic_model: str | None = "openai/gpt-oss-20b"
    # Reasoning models spend completion tokens "thinking"; keep it low for latency.
    llm_reasoning_effort: str | None = "low"
    llm_timeout_s: float = Field(30.0, gt=0)
    llm_max_retries: int = Field(3, ge=0, description="Retries on 429/5xx/timeouts/connection errors")
    llm_max_wait_s: float = Field(45.0, ge=0, description="Cap on total retry wait per LLM call")
    generation_temperature: float = Field(0.1, ge=0, le=2)
    generation_max_tokens: int = Field(1200, gt=0)

    # Optional pricing, USD per 1M tokens. Left unset on purpose: provider
    # prices change, so cost is only reported when you configure it.
    price_prompt_per_1m: float | None = Field(None, ge=0)
    price_completion_per_1m: float | None = Field(None, ge=0)

    # --- Storage ------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    # Optional folder of documents ingested at startup (useful on hosts with
    # ephemeral disks). Already-indexed files are skipped via content hash.
    seed_dir: Path | None = None

    # --- Ingestion ----------------------------------------------------------
    chunk_size: int = Field(800, ge=100, description="Target chunk size in characters")
    chunk_overlap: int = Field(120, ge=0)
    max_upload_mb: float = Field(20, gt=0)

    # --- Retrieval ----------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_top_k: int = Field(20, ge=1)
    bm25_top_k: int = Field(20, ge=1)
    rrf_k: int = Field(60, ge=1)
    final_top_k: int = Field(5, ge=1, le=20)
    rerank_candidates: int = Field(20, ge=1, description="Fused candidates sent to the reranker")
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Evidence gate: below these scores we abstain instead of calling the LLM.
    # Calibrated on ms-marco-MiniLM logits: on-topic >= -3.2, off-topic ~ -11 (see docs).
    min_rerank_score: float = -5.0
    min_dense_score: float = 0.25  # used when the reranker is disabled/failed

    # --- Query rewrite ------------------------------------------------------
    query_rewrite_enabled: bool = True

    # --- Reliability loop ---------------------------------------------------
    faithfulness_threshold: float = Field(0.80, ge=0, le=1)
    relevance_threshold: float = Field(0.70, ge=0, le=1)
    completeness_threshold: float = Field(0.60, ge=0, le=1)
    fail_on_unsupported_claims: bool = True
    max_retries: int = Field(2, ge=0, le=5)

    # --- Observability / service -------------------------------------------
    # Load embedding + reranker models at startup so the first query is not ~8 s slower.
    warmup_models: bool = True
    log_level: str = "INFO"
    log_json: bool = True
    # Include user text (query, unsupported-claim excerpts, critique notes) in logs. Disable for privacy.
    log_content: bool = True
    cors_origins: list[str] = ["*"]

    # --- UI -----------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def effective_judge_model(self) -> str:
        return self.judge_model or self.llm_model

    @property
    def effective_critic_model(self) -> str:
        return self.critic_model or self.effective_judge_model

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.sqlite3"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (cached; override in tests via DI)."""
    return Settings()
