"""Query API request/response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.reliability import Attempt, JudgeResult
from app.schemas.retrieval import RetrievedChunk
from app.schemas.trace import TraceOut

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I don't have enough evidence in the uploaded documents to answer this reliably."
)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryOptions(BaseModel):
    """Per-request overrides. `None` means "use the server default"."""

    mode: Literal["adversarial", "baseline"] = Field(
        "adversarial", description="'baseline' skips the judge/critic loop (single-pass RAG)"
    )
    top_k: int | None = Field(None, ge=1, le=20, description="Chunks passed to the generator")
    max_retries: int | None = Field(None, ge=0, le=5)
    faithfulness_threshold: float | None = Field(None, ge=0, le=1)
    relevance_threshold: float | None = Field(None, ge=0, le=1)
    completeness_threshold: float | None = Field(None, ge=0, le=1)
    use_reranker: bool | None = None
    use_query_rewrite: bool | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["What ROC-AUC does the Transformer achieve?"])
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    options: QueryOptions = Field(default_factory=QueryOptions)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class Citation(BaseModel):
    index: int = Field(description="1-based marker used in the answer, e.g. [1]")
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None = None
    chunk_index: int
    label: str = Field(description="Human-readable source, e.g. 'paper.pdf — page 4'")
    text: str


class ReliabilityReport(BaseModel):
    enabled: bool = Field(description="False in baseline mode or when the pipeline abstained early")
    initial: JudgeResult | None = None
    final: JudgeResult | None = None
    retries: int = 0
    max_retries: int = 0
    improved: bool = Field(False, description="True when a retry beat the initial judge score")
    attempts: list[Attempt] = Field(default_factory=list)
    judge_error: str | None = None


AnswerStatus = Literal["answered", "insufficient_evidence", "no_documents"]


class QueryResponse(BaseModel):
    request_id: str
    query: str
    retrieval_query: str
    query_rewritten: bool
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    reliability: ReliabilityReport
    trace: TraceOut
