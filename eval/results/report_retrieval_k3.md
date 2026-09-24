# Evaluation report: baseline RAG vs adversarial RAG

- **Run:** 20260924T151443Z · dataset `golden.jsonl` (22 questions) · corpus: drift_detection_paper.txt
- **Models:** generator `openai/gpt-oss-120b`, pipeline judge `qwen/qwen3.8-27b`, critic `openai/gpt-oss-20b`, **independent evaluator** `openai/gpt-oss-120b`
- **Retrieval:** dense `sentence-transformers/all-MiniLM-L6-v2` + BM25 → RRF(k=60) → reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, top-3
- **Loop settings:** thresholds F≥0.8 R≥0.7 C≥0.6, fail on unsupported claims = True, max retries = 2

> All numbers below were produced by `python -m eval.run_eval` in this run. Small n: treat differences of a few points as noise. LLM-judged metrics come from an evaluator model that is independent of the pipeline's judge, but LLM judges are still imperfect; fact recall and abstention are deterministic string checks.

## Retrieval ablation (deterministic, 19 labelled questions)

| System | hit@3 | evidence recall@3 | MRR | p50 latency | misses |
|---|---|---|---|---|---|
| dense_only | 0.789 | 0.763 | 0.640 | 38 ms | q05, q11, q17, q19 |
| bm25_only | 0.947 | 0.947 | 0.860 | 0 ms | q19 |
| hybrid_rrf | 0.895 | 0.868 | 0.772 | 28 ms | q17, q19 |
| hybrid_rrf_rerank | 0.842 | 0.842 | 0.763 | 1655 ms | q11, q17, q19 |
