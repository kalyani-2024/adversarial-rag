# Evaluation report: baseline RAG vs adversarial RAG

- **Run:** 20260924T144213Z · dataset `golden.jsonl` (22 questions) · corpus: drift_detection_paper.txt
- **Models:** generator `openai/gpt-oss-120b`, pipeline judge `qwen/qwen3.8-27b`, critic `openai/gpt-oss-20b`, **independent evaluator** `openai/gpt-oss-120b`
- **Retrieval:** dense `sentence-transformers/all-MiniLM-L6-v2` + BM25 → RRF(k=60) → reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5
- **Loop settings:** thresholds F≥0.8 R≥0.7 C≥0.6, fail on unsupported claims = True, max retries = 2

> All numbers below were produced by `python -m eval.run_eval` in this run. Small n: treat differences of a few points as noise. LLM-judged metrics come from an evaluator model that is independent of the pipeline's judge, but LLM judges are still imperfect; fact recall and abstention are deterministic string checks.

## Retrieval ablation (deterministic, 19 labelled questions)

| System | hit@5 | evidence recall@5 | MRR | p50 latency | misses |
|---|---|---|---|---|---|
| dense_only | 0.842 | 0.816 | 0.653 | 14 ms | q05, q17, q19 |
| bm25_only | 0.947 | 0.947 | 0.860 | 0 ms | q19 |
| hybrid_rrf | 0.895 | 0.895 | 0.772 | 14 ms | q17, q19 |
| hybrid_rrf_rerank | 0.895 | 0.895 | 0.774 | 1014 ms | q17, q19 |

## Answer quality

| Metric | Baseline | Adversarial | Δ |
|---|---|---|---|
| Fact recall (answerable, deterministic) | 0.898 | 0.898 | +0.000 |
| Evaluator faithfulness | 1.000 | 0.994 | -0.006 |
| Evaluator relevance | 1.000 | 1.000 | +0.000 |
| Evaluator completeness | 1.000 | 1.000 | +0.000 |
| Answers with ≥1 unsupported claim ↓ | 0% | 6% | +0.059 |
| Unsupported claims per answer ↓ | 0.000 | 0.059 | +0.059 |
| False abstention rate (answerable) ↓ | 11% | 11% | +0.000 |
| Correct abstention rate (unanswerable) | 100% | 100% | +0.000 |
| Retrieval hit@5 | 0.895 | 0.895 | +0.000 |

## Cost and latency

| Metric | Baseline | Adversarial | Ratio |
|---|---|---|---|
| Mean latency, net of throttling (ms) | 1636 | 1914 | 1.17× |
| p50 latency, net of throttling (ms) | 1713 | 1978 | 1.15× |
| p95 latency, net of throttling (ms) | 2198 | 2634 | 1.20× |
| Mean latency incl. provider throttling (ms) | 2973 | 3374 | 1.13× |
| Mean time waiting on rate limits (ms) | 1337 | 1460 | 1.09× |
| LLM calls / query | 1 | 2 | 2.00× |
| Tokens / query | 992 | 2121 | 2.14× |

*Net latency* subtracts time the LLM client spent sleeping on provider rate limits (Groq free tier: 8k tokens/min per model), which depends on the account tier, not on the system.

- Retry rate (adversarial, answered questions): **0%**
- Answers where the loop improved the pipeline judge's score: **0%**
- Errors: baseline 0, adversarial 0

## By question category

| Category | Mode | n | fact recall | evaluator faithfulness | answers w/ unsupported claims | abstained |
|---|---|---|---|---|---|---|
| lookup | baseline | 9 | 1.000 | 1.000 | 0% | 0 |
| lookup | adversarial | 9 | 1.000 | 1.000 | 0% | 0 |
| synthesis | baseline | 7 | 0.881 | 1.000 | 0% | 0 |
| synthesis | adversarial | 7 | 0.881 | 0.986 | 14% | 0 |
| bait | baseline | 3 | 0.500 | 1.000 | 0% | 2 |
| bait | adversarial | 3 | 0.500 | 1.000 | 0% | 2 |
| unanswerable | baseline | 3 | — | — | — | 3 |
| unanswerable | adversarial | 3 | — | — | — | 3 |

## Pipeline judge calibration (seeded faults)

Judge `qwen/qwen3.8-27b`: detected **100%** of 6 seeded faults; false alarms on **0%** of 4 correct controls.

| Case | Faulty | Verdict | Failed checks |
|---|---|---|---|
| correct | no | PASS | — |
| wrong_number | yes | FAIL | faithfulness, unsupported_claims |
| correct | no | PASS | — |
| outside_knowledge | yes | FAIL | faithfulness, unsupported_claims |
| correct | no | PASS | — |
| contradiction | yes | FAIL | faithfulness, completeness, unsupported_claims |
| correct | no | PASS | — |
| overclaim | yes | FAIL | faithfulness, completeness, unsupported_claims |
| incomplete | yes | FAIL | faithfulness, relevance, completeness, unsupported_claims |
| off_topic | yes | FAIL | relevance, completeness |
