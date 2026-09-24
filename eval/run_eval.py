"""Run the evaluation harness.

    python -m eval.run_eval                       # full run: ablation + calibration + both modes
    python -m eval.run_eval --retrieval-only      # deterministic retrieval ablation, no LLM calls
    python -m eval.run_eval --limit 5             # quick smoke run
    python -m eval.run_eval --resume eval/runs/<ts>

Working files (isolated index, raw results) go to eval/runs/<timestamp>/ (git-ignored).
With --publish, report.md / summary.json / results.jsonl are copied to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.evaluation.dataset import load_golden, validate_evidence  # noqa: E402
from app.evaluation.harness import (  # noqa: E402
    EvalConfig, build_eval_container, ingest_corpus, make_evaluator, retrieval_ablation, run_answers, summarize,
)
from app.evaluation.judge_calibration import run_calibration  # noqa: E402
from app.evaluation.report import render_report  # noqa: E402
from app.observability.logging import configure_logging  # noqa: E402
from app.schemas.reliability import Thresholds  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=ROOT / "eval" / "golden.jsonl")
    p.add_argument("--corpus", type=Path, nargs="+", default=sorted((ROOT / "eval" / "corpus").glob("*")))
    p.add_argument("--eval-judge-model", default="openai/gpt-oss-120b",
                   help="Independent evaluator (must differ from the pipeline judge).")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--skip-calibration", action="store_true")
    p.add_argument("--pause", type=float, default=0.0, help="Seconds to sleep between queries (rate limits).")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--publish", action="store_true", help="Copy report/summary/results to eval/results/.")
    p.add_argument("--rescore", type=Path, default=None,
                   help="Recompute deterministic metrics + report for an existing run dir (no LLM calls).")
    p.add_argument("--name", default="", help="Suffix for published files, e.g. 'stress' -> report_stress.md")
    args = p.parse_args()

    settings = get_settings()
    configure_logging("INFO", json_logs=False)
    if args.rescore:
        _rescore(args)
        return
    if args.eval_judge_model == settings.effective_judge_model:
        sys.exit("The evaluator must differ from the pipeline judge model (otherwise the loop is graded by its own judge).")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.resume or ROOT / "eval" / "runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_golden(args.dataset)[: args.limit]
    container = build_eval_container(settings, out_dir)
    corpus_text = ingest_corpus(container, args.corpus)
    missing = validate_evidence(items, corpus_text)
    if missing:
        sys.exit("Evidence labels not found in corpus:\n" + "\n".join(missing))

    s = container.settings
    meta = {
        "timestamp": ts, "dataset": args.dataset.name, "n_items": len(items), "corpus": [c.name for c in args.corpus],
        "llm_model": s.llm_model, "judge_model": s.effective_judge_model, "critic_model": s.effective_critic_model,
        "eval_judge_model": args.eval_judge_model, "embedding_model": s.embedding_model, "reranker_model": s.reranker_model,
        "rrf_k": s.rrf_k, "k": args.k, "max_retries": s.max_retries,
        "thresholds": {"faithfulness": s.faithfulness_threshold, "relevance": s.relevance_threshold,
                       "completeness": s.completeness_threshold, "fail_on_unsupported_claims": s.fail_on_unsupported_claims},
    }

    print(f"[eval] output: {out_dir}")
    ablation = retrieval_ablation(container, items, args.k)
    print(json.dumps(ablation, indent=2))

    summary, calibration = {}, None
    if not args.retrieval_only:
        evaluator = make_evaluator(settings)
        if not args.skip_calibration:
            thresholds = Thresholds(faithfulness=s.faithfulness_threshold, relevance=s.relevance_threshold,
                                    completeness=s.completeness_threshold, fail_on_unsupported_claims=s.fail_on_unsupported_claims)
            calibration = run_calibration(evaluator, thresholds, s.effective_judge_model)
            print(f"[calibration] detection={calibration['fault_detection_rate']:.2f} false_alarm={calibration['false_alarm_rate']:.2f}")
        cfg = EvalConfig(out_dir=out_dir, corpus=args.corpus, eval_judge_model=args.eval_judge_model, k=args.k, pause_s=args.pause)
        rows = run_answers(container, evaluator, items, cfg)
        summary = summarize(rows, args.k)

    result = {"meta": meta, "retrieval_ablation": ablation, "summary": summary, "judge_calibration": calibration}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = render_report(meta, summary, ablation, calibration)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)

    if args.publish:
        _publish(out_dir, args.name)
    container.shutdown()


def _publish(run_dir: Path, name: str) -> None:
    dest = ROOT / "eval" / "results"
    dest.mkdir(exist_ok=True)
    suffix = f"_{name}" if name else ""
    for fname in ("report.md", "summary.json", "results.jsonl"):
        if (run_dir / fname).exists():
            stem, ext = fname.rsplit(".", 1)
            shutil.copy2(run_dir / fname, dest / f"{stem}{suffix}.{ext}")
    print(f"[eval] published to {dest}")


def _rescore(args) -> None:
    from app.evaluation.harness import rescore

    run_dir = args.rescore
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rescore(rows, load_golden(args.dataset))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    saved["summary"] = summarize(rows, saved["meta"]["k"])
    saved["meta"]["rescored"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (run_dir / "summary.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    report = render_report(saved["meta"], saved["summary"], saved["retrieval_ablation"], saved["judge_calibration"])
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    if args.publish:
        _publish(run_dir, args.name)


if __name__ == "__main__":
    main()
