"""Evaluation harness: baseline RAG vs adversarial RAG on a golden dataset.

Design choices (and why):
  * Isolated index: the harness builds its own container in a temp/output
    directory and never touches the user's data (the v1 script deleted it).
  * Same retrieval for both modes, so differences come from the reliability
    loop alone.
  * Independent evaluator: answers are scored by `eval_judge_model`, NOT the
    pipeline's judge. Otherwise the adversarial loop, which optimizes against
    the pipeline judge, would be graded by the same model it was tuned
    to satisfy ("teaching to the test").
  * Deterministic metrics first (fact recall, abstention, retrieval recall);
    LLM-judged metrics are reported alongside with that caveat.
  * Resumable: every (question, mode) result is appended to results.jsonl.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.core.config import Settings
from app.core.errors import LLMError, LLMRateLimitError
from app.core.llm import GroqLLM, LLMClient
from app.evaluation.dataset import GoldenItem
from app.evaluation.metrics import fact_recall, mean, percentile, retrieval_scores
from app.generation.generator import is_abstention
from app.generation.judge import judge_answer
from app.schemas.query import QueryOptions, QueryRequest
from app.schemas.reliability import Thresholds
from app.services.container import Container, build_container

logger = logging.getLogger("rag.eval")

MODES = ("baseline", "adversarial")


@dataclass
class EvalConfig:
    out_dir: Path
    corpus: list[Path]
    eval_judge_model: str
    k: int = 5
    max_rate_limit_retries: int = 6
    rate_limit_backoff_s: float = 20.0
    pause_s: float = 0.0


def build_eval_container(settings: Settings, out_dir: Path, llm: LLMClient | None = None) -> Container:
    eval_settings = settings.model_copy(update={"data_dir": out_dir / "index_data", "seed_dir": None, "warmup_models": True})
    container = build_container(eval_settings, llm=llm)
    container.startup()
    return container


def ingest_corpus(container: Container, corpus: list[Path]) -> str:
    """Ingest corpus files (duplicates skipped) and return their concatenated raw text for label checks."""
    from app.core.errors import DuplicateDocumentError

    texts = []
    for path in corpus:
        try:
            container.documents.ingest(path.name, path.read_bytes())
        except DuplicateDocumentError:
            pass
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(texts)


# --- retrieval ablation (deterministic, no LLM) -------------------------------------------------
def retrieval_ablation(container: Container, items: list[GoldenItem], k: int) -> dict[str, dict[str, Any]]:
    retriever = container.retriever
    store = container.store
    labelled = [i for i in items if i.evidence]

    def texts(ids: list[int]) -> list[str]:
        recs = store.get_chunks(ids)
        return [recs[i].text for i in ids if i in recs]

    def dense_only(q: str):
        t0 = time.perf_counter()
        hits = retriever.dense_index.search(retriever.embedder.embed([q])[0], k)
        return texts([i for i, _ in hits]), (time.perf_counter() - t0) * 1000

    def bm25_only(q: str):
        t0 = time.perf_counter()
        hits = retriever.bm25_index.search(q, k)
        return texts([i for i, _ in hits]), (time.perf_counter() - t0) * 1000

    def hybrid(q: str):
        r = retriever.retrieve(q, top_k=k, use_reranker=False)
        return [c.text for c in r.chunks], r.retrieval_ms

    def hybrid_rerank(q: str):
        r = retriever.retrieve(q, top_k=k, use_reranker=True)
        return [c.text for c in r.chunks], r.retrieval_ms + r.rerank_ms

    systems: dict[str, Callable[[str], tuple[list[str], float]]] = {"dense_only": dense_only, "bm25_only": bm25_only, "hybrid_rrf": hybrid, "hybrid_rrf_rerank": hybrid_rerank}
    report: dict[str, dict[str, Any]] = {}
    for name, fn in systems.items():
        rows, latencies = [], []
        for item in labelled:
            chunks, ms = fn(item.question)
            latencies.append(ms)
            rows.append(retrieval_scores(chunks, item.evidence, k) | {"id": item.id})
        report[name] = {
            "n": len(rows),
            f"hit@{k}": mean([r["hit"] for r in rows]),
            f"evidence_recall@{k}": mean([r["recall"] for r in rows]),
            "mrr": mean([r["mrr"] for r in rows]),
            "latency_p50_ms": percentile(latencies, 0.5),
            "misses": [r["id"] for r in rows if r["hit"] == 0.0],
        }
    return report


# --- answer evaluation ------------------------------------------------------------------------------
def _with_rate_limit_retry(fn: Callable[[], Any], cfg: EvalConfig) -> Any:
    for attempt in range(cfg.max_rate_limit_retries + 1):
        try:
            return fn()
        except LLMRateLimitError:
            if attempt == cfg.max_rate_limit_retries:
                raise
            wait = cfg.rate_limit_backoff_s * (attempt + 1)
            logger.warning("rate limited; backing off", extra={"wait_s": wait})
            time.sleep(wait)


def evaluate_item(container: Container, evaluator: LLMClient, item: GoldenItem, mode: str, cfg: EvalConfig) -> dict[str, Any]:
    request = QueryRequest(query=item.question, options=QueryOptions(mode=mode))
    resp = _with_rate_limit_retry(lambda: container.queries.run(request), cfg)
    abstained = resp.status != "answered" or is_abstention(resp.answer)

    row: dict[str, Any] = {
        "id": item.id, "category": item.category, "mode": mode, "question": item.question,
        "answerable": item.answerable, "status": resp.status, "abstained": abstained, "answer": resp.answer,
        "fact_recall": None if abstained and item.answerable else fact_recall(resp.answer, item.expected_facts),
        "citations": len(resp.citations),
        "retrieval": retrieval_scores([c.text for c in resp.retrieved_chunks], item.evidence, cfg.k) if item.evidence else None,
        "latency_ms": resp.trace.total_ms, "throttle_ms": resp.trace.throttle_ms,
        "llm_retries": resp.trace.llm_retries, "llm_calls": resp.trace.llm_calls,
        "prompt_tokens": resp.trace.prompt_tokens, "completion_tokens": resp.trace.completion_tokens,
        "retries": resp.reliability.retries, "improved": resp.reliability.improved,
        "pipeline_initial_verdict": resp.reliability.initial.verdict if resp.reliability.initial else None,
        "pipeline_final_verdict": resp.reliability.final.verdict if resp.reliability.final else None,
        "eval": None,
    }
    if item.expected_facts and abstained and item.answerable:
        row["fact_recall"] = 0.0  # a false abstention scores zero on facts

    # Independent evaluator scores every non-abstained answer against the retrieved context.
    if not abstained:
        lenient = Thresholds(faithfulness=0, relevance=0, completeness=0, fail_on_unsupported_claims=False)
        try:
            judged, _ = _with_rate_limit_retry(
                lambda: judge_answer(evaluator, item.question, resp.retrieved_chunks, resp.answer, lenient,
                                     model=cfg.eval_judge_model), cfg)
            row["eval"] = {"faithfulness": judged.faithfulness, "relevance": judged.relevance,
                           "completeness": judged.completeness, "unsupported_claims": judged.unsupported_claims}
        except LLMError as exc:
            row["eval_error"] = exc.message
    return row


def run_answers(container: Container, evaluator: LLMClient, items: list[GoldenItem], cfg: EvalConfig,
                modes: tuple[str, ...] = MODES) -> list[dict[str, Any]]:
    results_path = cfg.out_dir / "results.jsonl"
    done: dict[tuple[str, str], dict] = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["id"], r["mode"])] = r
    rows = []
    with results_path.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(items, start=1):
            for mode in modes:
                if (item.id, mode) in done:
                    rows.append(done[(item.id, mode)])
                    continue
                try:
                    row = evaluate_item(container, evaluator, item, mode, cfg)
                except LLMError as exc:
                    row = {"id": item.id, "category": item.category, "mode": mode, "error": exc.message}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                rows.append(row)
                logger.info("evaluated", extra={"n": n, "total": len(items), "id": item.id, "mode": mode,
                                                "status": row.get("status"), "fact_recall": row.get("fact_recall"),
                                                "retries": row.get("retries"), "error": row.get("error")})
                if cfg.pause_s:
                    time.sleep(cfg.pause_s)
    return rows


def rescore(rows: list[dict[str, Any]], items: list[GoldenItem]) -> list[dict[str, Any]]:
    """Recompute deterministic metrics from saved answers (e.g. after a metric fix) without new LLM calls."""
    by_id = {i.id: i for i in items}
    for r in rows:
        item = by_id.get(r["id"])
        if item is None or "error" in r:
            continue
        if r["abstained"]:
            r["fact_recall"] = 0.0 if item.answerable and item.expected_facts else None
        else:
            r["fact_recall"] = fact_recall(r["answer"], item.expected_facts)
    return rows


# --- aggregation ---------------------------------------------------------------------------------
def summarize(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in MODES:
        rs = [r for r in rows if r.get("mode") == mode and "error" not in r]
        if not rs:
            continue
        answerable = [r for r in rs if r["answerable"]]
        unanswerable = [r for r in rs if not r["answerable"]]
        answered = [r for r in rs if not r["abstained"] and r.get("eval")]
        lat = [r["latency_ms"] for r in rs]
        net = [r["latency_ms"] - r.get("throttle_ms", 0.0) for r in rs]
        out[mode] = {
            "n": len(rs),
            "errors": len([r for r in rows if r.get("mode") == mode and "error" in r]),
            "fact_recall": mean([r["fact_recall"] for r in answerable if r["fact_recall"] is not None]),
            "false_abstention_rate": mean([1.0 if r["abstained"] else 0.0 for r in answerable]),
            "correct_abstention_rate": mean([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
            "eval_faithfulness": mean([r["eval"]["faithfulness"] for r in answered]),
            "eval_relevance": mean([r["eval"]["relevance"] for r in answered]),
            "eval_completeness": mean([r["eval"]["completeness"] for r in answered]),
            "answers_with_unsupported_claims": mean([1.0 if r["eval"]["unsupported_claims"] else 0.0 for r in answered]),
            "unsupported_claims_per_answer": mean([float(len(r["eval"]["unsupported_claims"])) for r in answered]),
            f"retrieval_hit@{k}": mean([r["retrieval"]["hit"] for r in rs if r.get("retrieval")]),
            "latency_mean_ms": mean(lat), "latency_p50_ms": percentile(lat, 0.5), "latency_p95_ms": percentile(lat, 0.95),
            "net_latency_mean_ms": mean(net), "net_latency_p50_ms": percentile(net, 0.5),
            "net_latency_p95_ms": percentile(net, 0.95),
            "throttle_ms_mean": mean([r.get("throttle_ms", 0.0) for r in rs]),
            "llm_calls_mean": mean([float(r["llm_calls"]) for r in rs]),
            "tokens_mean": mean([float(r["prompt_tokens"] + r["completion_tokens"]) for r in rs]),
            "retry_rate": mean([1.0 if r["retries"] > 0 else 0.0 for r in rs if not r["abstained"]]) if mode == "adversarial" else None,
            "improved_rate": mean([1.0 if r["improved"] else 0.0 for r in rs if not r["abstained"]]) if mode == "adversarial" else None,
            "by_category": {
                cat: {
                    "n": len(cr),
                    "fact_recall": mean([r["fact_recall"] for r in cr if r["fact_recall"] is not None]),
                    "eval_faithfulness": mean([r["eval"]["faithfulness"] for r in cr if r.get("eval")]),
                    "answers_with_unsupported_claims": mean([1.0 if r["eval"]["unsupported_claims"] else 0.0 for r in cr if r.get("eval")]),
                    "abstained": sum(1 for r in cr if r["abstained"]),
                }
                for cat in ("lookup", "synthesis", "bait", "compound", "unanswerable")
                if (cr := [r for r in rs if r["category"] == cat])
            },
        }
    return out


def make_evaluator(settings: Settings) -> GroqLLM:
    return GroqLLM(settings)
