"""Render evaluation results as Markdown. Every number comes from the run's JSON."""

from __future__ import annotations

from typing import Any


def _f(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.0f}%"


def _delta(a: float | None, b: float | None, digits: int = 3) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{'+' if d >= 0 else ''}{d:.{digits}f}"


def render_report(meta: dict[str, Any], summary: dict[str, Any], ablation: dict[str, Any] | None,
                  calibration: dict[str, Any] | None) -> str:
    k = meta["k"]
    lines = [
        "# Evaluation report: baseline RAG vs adversarial RAG",
        "",
        f"- **Run:** {meta['timestamp']} · dataset `{meta['dataset']}` ({meta['n_items']} questions) · corpus: {', '.join(meta['corpus'])}",
        f"- **Models:** generator `{meta['llm_model']}`, pipeline judge `{meta['judge_model']}`, critic `{meta['critic_model']}`, "
        f"**independent evaluator** `{meta['eval_judge_model']}`",
        f"- **Retrieval:** dense `{meta['embedding_model']}` + BM25 → RRF(k={meta['rrf_k']}) → reranker `{meta['reranker_model']}`, top-{k}",
        f"- **Loop settings:** thresholds F≥{meta['thresholds']['faithfulness']} R≥{meta['thresholds']['relevance']} "
        f"C≥{meta['thresholds']['completeness']}, fail on unsupported claims = {meta['thresholds']['fail_on_unsupported_claims']}, "
        f"max retries = {meta['max_retries']}",
        "",
        "> All numbers below were produced by `python -m eval.run_eval` in this run. Small n: treat differences of a few "
        "points as noise. LLM-judged metrics come from an evaluator model that is independent of the pipeline's judge, "
        "but LLM judges are still imperfect; fact recall and abstention are deterministic string checks.",
        "",
    ]

    if ablation:
        lines += [f"## Retrieval ablation (deterministic, {next(iter(ablation.values()))['n']} labelled questions)", "",
                  f"| System | hit@{k} | evidence recall@{k} | MRR | p50 latency | misses |", "|---|---|---|---|---|---|"]
        for name, r in ablation.items():
            lines.append(f"| {name} | {_f(r[f'hit@{k}'])} | {_f(r[f'evidence_recall@{k}'])} | {_f(r['mrr'])} | "
                         f"{_f(r['latency_p50_ms'], 0)} ms | {', '.join(r['misses']) or '—'} |")
        lines.append("")

    b, a = summary.get("baseline"), summary.get("adversarial")
    if b and a:
        rows = [
            ("Fact recall (answerable, deterministic)", "fact_recall", False),
            ("Evaluator faithfulness", "eval_faithfulness", False),
            ("Evaluator relevance", "eval_relevance", False),
            ("Evaluator completeness", "eval_completeness", False),
            ("Answers with ≥1 unsupported claim ↓", "answers_with_unsupported_claims", True),
            ("Unsupported claims per answer ↓", "unsupported_claims_per_answer", False),
            ("False abstention rate (answerable) ↓", "false_abstention_rate", True),
            ("Correct abstention rate (unanswerable)", "correct_abstention_rate", True),
            (f"Retrieval hit@{k}", f"retrieval_hit@{k}", False),
        ]
        lines += ["## Answer quality", "", "| Metric | Baseline | Adversarial | Δ |", "|---|---|---|---|"]
        for label, key, pct in rows:
            fmt = _pct if pct else _f
            lines.append(f"| {label} | {fmt(b[key])} | {fmt(a[key])} | {_delta(b[key], a[key])} |")
        lines += ["", "## Cost and latency", "", "| Metric | Baseline | Adversarial | Ratio |", "|---|---|---|---|"]
        for label, key in [("Mean latency, net of throttling (ms)", "net_latency_mean_ms"),
                           ("p50 latency, net of throttling (ms)", "net_latency_p50_ms"),
                           ("p95 latency, net of throttling (ms)", "net_latency_p95_ms"),
                           ("Mean latency incl. provider throttling (ms)", "latency_mean_ms"),
                           ("Mean time waiting on rate limits (ms)", "throttle_ms_mean"),
                           ("LLM calls / query", "llm_calls_mean"),
                           ("Tokens / query", "tokens_mean")]:
            ratio = f"{a[key] / b[key]:.2f}×" if a[key] and b[key] else "—"
            lines.append(f"| {label} | {_f(b[key], 0)} | {_f(a[key], 0)} | {ratio} |")
        lines += ["", "*Net latency* subtracts time the LLM client spent sleeping on provider rate limits "
                  "(Groq free tier: 8k tokens/min per model), which depends on the account tier, not on the system.", "",
                  f"- Retry rate (adversarial, answered questions): **{_pct(a['retry_rate'])}**",
                  f"- Answers where the loop improved the pipeline judge's score: **{_pct(a['improved_rate'])}**",
                  f"- Errors: baseline {b['errors']}, adversarial {a['errors']}", ""]

        lines += ["## By question category", "", "| Category | Mode | n | fact recall | evaluator faithfulness | answers w/ unsupported claims | abstained |",
                  "|---|---|---|---|---|---|---|"]
        for cat in ("lookup", "synthesis", "bait", "compound", "unanswerable"):
            for mode, s in (("baseline", b), ("adversarial", a)):
                c = s["by_category"].get(cat)
                if c:
                    lines.append(f"| {cat} | {mode} | {c['n']} | {_f(c['fact_recall'])} | {_f(c['eval_faithfulness'])} | "
                                 f"{_pct(c['answers_with_unsupported_claims'])} | {c['abstained']} |")
        lines.append("")

    if calibration:
        lines += ["## Pipeline judge calibration (seeded faults)", "",
                  f"Judge `{calibration['model']}`: detected **{_pct(calibration['fault_detection_rate'])}** of "
                  f"{calibration['n_faults']} seeded faults; false alarms on **{_pct(calibration['false_alarm_rate'])}** of "
                  f"{calibration['n_controls']} correct controls.", "",
                  "| Case | Faulty | Verdict | Failed checks |", "|---|---|---|---|"]
        for c in calibration["cases"]:
            lines.append(f"| {c['kind']} | {'yes' if c['faulty'] else 'no'} | {c['verdict']} | {', '.join(c['failed_checks']) or '—'} |")
        lines.append("")
    return "\n".join(lines)
