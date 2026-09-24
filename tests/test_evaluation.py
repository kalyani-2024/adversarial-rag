import json
from pathlib import Path

from app.evaluation.dataset import GoldenItem, load_golden, validate_evidence
from app.evaluation.harness import EvalConfig, retrieval_ablation, run_answers, summarize
from app.evaluation.metrics import fact_recall, normalize, retrieval_scores
from app.evaluation.report import render_report
from app.services.container import build_container
from tests.fakes import FakeLLM, HashEmbedder

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_handles_dashes_and_thousands():
    assert normalize("ROC‑AUC of 15,000 Students") == "roc auc of 15000 students"
    assert normalize("gradient‑reversal") == normalize("gradient reversal")
    assert normalize("0.950, 0.968") == "0.950, 0.968"  # decimals in lists are untouched


def test_fact_recall_aliases():
    facts = [["15,000", "15000"], ["rtx 4060"], ["missing"]]
    assert fact_recall("Sampled 15 000 students on an RTX 4060.", facts) == 2 / 3
    assert fact_recall("anything", []) is None


def test_retrieval_scores():
    chunks = ["alpha beta", "gamma delta", "epsilon"]
    s = retrieval_scores(chunks, ["gamma", "epsilon", "zeta"], k=3)
    assert s == {"hit": 1.0, "recall": 2 / 3, "mrr": 0.5}
    assert retrieval_scores(chunks, ["zeta"], k=3)["mrr"] == 0.0


def test_golden_dataset_is_valid_against_corpus():
    items = load_golden(ROOT / "eval" / "golden.jsonl")
    corpus = (ROOT / "eval" / "corpus" / "drift_detection_paper.txt").read_text(encoding="utf-8")
    assert validate_evidence(items, corpus) == []
    assert {i.category for i in items} == {"lookup", "synthesis", "bait", "unanswerable"}
    assert all(not i.answerable for i in items if i.category == "unanswerable")


def test_harness_end_to_end_with_fakes(settings, tmp_path):
    judge = json.dumps({"faithfulness": 1.0, "relevance": 1.0, "completeness": 1.0, "unsupported_claims": [], "reason": "ok"})
    llm = FakeLLM({"generate": ["The Transformer reaches ROC-AUC 0.959 [1]."], "judge": [judge]})
    container = build_container(settings, llm=llm, embedder=HashEmbedder())
    container.startup()
    container.documents.ingest("paper.txt", b"The Transformer reaches ROC-AUC 0.959 on the test split.")
    items = [
        GoldenItem(id="a", category="lookup", question="What ROC-AUC does the Transformer reach?",
                   expected_facts=[["0.959"]], evidence=["ROC-AUC 0.959"]),
        GoldenItem(id="u", category="unanswerable", question="quantum chromodynamics lattice gauge", answerable=False),
    ]
    ablation = retrieval_ablation(container, items, k=5)
    assert ablation["hybrid_rrf"]["hit@5"] == 1.0 and ablation["bm25_only"]["n"] == 1

    cfg = EvalConfig(out_dir=tmp_path, corpus=[], eval_judge_model="evaluator")
    rows = run_answers(container, llm, items, cfg)
    assert len(rows) == 4 and (tmp_path / "results.jsonl").exists()
    summary = summarize(rows, k=5)
    assert summary["baseline"]["fact_recall"] == 1.0
    assert summary["adversarial"]["correct_abstention_rate"] == 1.0
    assert summary["adversarial"]["retry_rate"] == 0.0

    # resume: nothing is re-run
    calls = len(llm.calls)
    run_answers(container, llm, items, cfg)
    assert len(llm.calls) == calls

    meta = {"timestamp": "t", "dataset": "d", "n_items": 2, "corpus": ["paper.txt"], "llm_model": "g", "judge_model": "j",
            "critic_model": "c", "eval_judge_model": "e", "embedding_model": "m", "reranker_model": "r", "rrf_k": 60, "k": 5,
            "max_retries": 2, "thresholds": {"faithfulness": 0.8, "relevance": 0.7, "completeness": 0.6, "fail_on_unsupported_claims": True}}
    report = render_report(meta, summary, ablation, None)
    assert "| Fact recall" in report and "hybrid_rrf" in report
    container.shutdown()
