# Evaluation report: baseline RAG vs adversarial RAG

- **Run:** 20260924T145612Z · dataset `stress.jsonl` (6 questions) · corpus: drift_detection_paper.txt
- **Models:** generator `openai/gpt-oss-120b`, pipeline judge `qwen/qwen3.8-27b`, critic `openai/gpt-oss-20b`, **independent evaluator** `openai/gpt-oss-120b`
- **Retrieval:** dense `sentence-transformers/all-MiniLM-L6-v2` + BM25 → RRF(k=60) → reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-5
- **Loop settings:** thresholds F≥0.8 R≥0.7 C≥0.6, fail on unsupported claims = True, max retries = 2

> All numbers below were produced by `python -m eval.run_eval` in this run. Small n: treat differences of a few points as noise. LLM-judged metrics come from an evaluator model that is independent of the pipeline's judge, but LLM judges are still imperfect; fact recall and abstention are deterministic string checks.

## Retrieval ablation (deterministic, 6 labelled questions)

| System | hit@5 | evidence recall@5 | MRR | p50 latency | misses |
|---|---|---|---|---|---|
| dense_only | 1.000 | 1.000 | 0.528 | 16 ms | — |
| bm25_only | 0.833 | 0.833 | 0.700 | 0 ms | s01 |
| hybrid_rrf | 1.000 | 1.000 | 0.681 | 16 ms | — |
| hybrid_rrf_rerank | 1.000 | 1.000 | 0.833 | 718 ms | — |

## Answer quality

| Metric | Baseline | Adversarial | Δ |
|---|---|---|---|
| Fact recall (answerable, deterministic) | 0.750 | 0.917 | +0.167 |
| Evaluator faithfulness | 1.000 | 0.975 | -0.025 |
| Evaluator relevance | 0.880 | 0.900 | +0.020 |
| Evaluator completeness | 0.890 | 0.917 | +0.027 |
| Answers with ≥1 unsupported claim ↓ | 0% | 17% | +0.167 |
| Unsupported claims per answer ↓ | 0.000 | 0.167 | +0.167 |
| False abstention rate (answerable) ↓ | 17% | 0% | -0.167 |
| Correct abstention rate (unanswerable) | — | — | — |
| Retrieval hit@5 | 1.000 | 1.000 | +0.000 |

## Cost and latency

| Metric | Baseline | Adversarial | Ratio |
|---|---|---|---|
| Mean latency, net of throttling (ms) | 1757 | 2746 | 1.56× |
| p50 latency, net of throttling (ms) | 1592 | 2337 | 1.47× |
| p95 latency, net of throttling (ms) | 2138 | 5164 | 2.42× |
| Mean latency incl. provider throttling (ms) | 1953 | 3258 | 1.67× |
| Mean time waiting on rate limits (ms) | 196 | 512 | 2.61× |
| LLM calls / query | 1 | 2 | 2.50× |
| Tokens / query | 1245 | 3509 | 2.82× |

*Net latency* subtracts time the LLM client spent sleeping on provider rate limits (Groq free tier: 8k tokens/min per model), which depends on the account tier, not on the system.

- Retry rate (adversarial, answered questions): **17%**
- Answers where the loop improved the pipeline judge's score: **17%**
- Errors: baseline 0, adversarial 0

## By question category

| Category | Mode | n | fact recall | evaluator faithfulness | answers w/ unsupported claims | abstained |
|---|---|---|---|---|---|---|
| compound | baseline | 6 | 0.750 | 1.000 | 0% | 1 |
| compound | adversarial | 6 | 0.917 | 0.975 | 17% | 0 |
