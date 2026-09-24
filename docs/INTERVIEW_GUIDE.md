# Interview guide: RAG Reliability Lab

How to explain this project in a technical interview: what to say, what to show, and how to answer the follow-ups. Every number here comes from `eval/results/report.md` or from a test in `tests/`. If an interviewer asks "how do you know?", you can point at the file.

---

## 1. The 60-second explanation

> "I built a RAG system that doesn't take its own answers on trust. Retrieval is hybrid: dense FAISS search plus BM25, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder. If the best evidence is weak, it refuses to answer, without calling the LLM. Otherwise it generates an answer that has to cite numbered sources. Then a separate judge model scores faithfulness, relevance and completeness. **Only if the answer fails** does an adversarial critic list the unsupported claims, and the generator rewrites the answer. It retries at most twice and returns the best attempt, never a worse one.
>
> The design came from a measurement. My first version always ran a critique-and-rewrite step, and my own eval showed it made answers *less* faithful at 5× the latency. So v2 makes the loop conditional, puts the PASS/FAIL decision in code instead of the LLM, and I evaluate baseline against adversarial on a golden dataset with an independent evaluator model. The honest result: with a strong generator, most answers already pass, so the loop rarely fires and doesn't measurably change quality on my test sets. What it buys is a calibrated per-answer verification signal; the judge caught 6 of 6 seeded faults. It's FastAPI + LangGraph + Streamlit, with per-request tracing, token accounting, 111 tests, and a Docker image."

## 2. The 5-minute architecture walkthrough

Draw this (or open the README diagram):

```
Upload → parse (page-aware) → clean → chunk (recursive, no page crossing) → embed → SQLite + FAISS (+ BM25 rebuild)

Query → [rewrite only if follow-up/filler] → dense top-20 ∥ BM25 top-20 → RRF → cross-encoder top-k
      → evidence gate ──fail──► "I don't have enough evidence…" (0 LLM calls)
      → generate with [n] citations
      → judge (scores) → verdict in code
            PASS → return
            FAIL → critic → regenerate → judge  (≤ MAX_RETRIES, return best attempt)
```

Talk track, one breath per box:

1. **Ingestion.** "PDF pages are parsed separately, so every chunk keeps its page number for citations. Chunking is recursive on paragraph, then line, then sentence, then word, so chunks never cut words, and they never cross a page. SQLite is the source of truth; FAISS only holds vectors keyed by chunk id, so I can delete a document and rebuild the index if the file is lost. Duplicates are caught by SHA-256."
2. **Retrieval.** "Dense catches paraphrase, BM25 catches exact numbers and identifiers. Their scores aren't comparable, so I fuse *ranks* with RRF. The cross-encoder then rescores the top 20 pairs jointly."
3. **Evidence gate.** "Before any LLM call, I check the best rerank score against a threshold I calibrated on real data: on-topic questions scored above −3.2 and off-topic ones about −11. Below −5 we abstain. Cheapest and safest answer."
4. **Generation.** "Sources are numbered; every claim must cite. I parse the citations afterwards, drop ones pointing at sources that don't exist, and map the rest to document/page for the UI."
5. **Reliability loop.** "The judge returns numbers only; `apply_thresholds` decides. If FAIL, the critic returns a structured list of problems and the generator rewrites against that list. The loop is an explicit LangGraph conditional edge with a retry counter plus a recursion limit."
6. **Observability.** "Each response carries a trace: spans per stage, LLM calls, tokens, time lost to rate limiting, request id. `/metrics` aggregates p50/p95 per stage and the retry rate."
7. **Evaluation.** "A golden set of 22 questions across four categories (lookup, synthesis, bait, unanswerable). It measures retrieval recall deterministically and compares baseline with adversarial on fact recall, abstention, and faithfulness from an *independent* evaluator model."

## 2b. Results, and how to talk about them

Numbers from `eval/results/` (runs on 2026-09-24). Know these cold, *and* know their limits.

| Finding | Number | What to say |
|---|---|---|
| Judge calibration | 6/6 seeded faults caught, 0/4 false alarms | "The signal the loop depends on is real, on a small calibration set." |
| Loop activity | fired on 0/17 answered golden questions, 1/6 compound stress questions | "Conditional means it stays out of the way when the answer is already grounded." |
| Quality, baseline vs adversarial | fact recall 0.898 vs 0.898; evaluator faithfulness 1.000 vs 0.994 | "No measurable difference at this n. It no longer *hurts* like v1 did, but I can't claim it improves quality here." |
| Cost | ~2× LLM calls and tokens; net latency 1.17× mean, 1.20× p95 | "Every answer pays for one judge call; retries only when needed." |
| Retrieval | BM25 alone best (hit@5 0.947); hybrid 0.895; dense 0.842; rerank neutral at k=5, worse at k=3, +~1 s | "My questions reuse the paper's wording, which favors BM25. I'd re-measure with paraphrased questions before claiming hybrid wins." |
| Abstention | 3/3 unanswerable refused, with 0 LLM calls (evidence gate); 2/19 answerable wrongly refused | "The gate is cheap and safe; the false refusals are the generator being conservative on two 'bait' questions." |

How to frame it: *"My eval showed the always-on critic hurt, so I made it conditional and measured again. Now it doesn't hurt, it costs about 2× calls, and on my small test sets it rarely triggers. The next step would be a harder, paraphrased, multi-document eval, plus claim-level verification, to find where it pays off."* Interviewers value that more than a suspiciously large improvement number.

## 3. Components and what to say about each

| Component | File | One-liner |
|---|---|---|
| Config | `app/core/config.py` | Every knob is an env var; secrets are `SecretStr`; validated at startup. |
| LLM client | `app/core/llm.py` | Protocol + Groq implementation; own retry loop honoring `Retry-After` so throttle time is measured; errors mapped to domain types. |
| Parsers / cleaning / chunking | `app/ingestion/` | Page-aware parsing, NFKC + PDF hyphenation repair, recursive chunking with word-boundary overlap. |
| Document store | `app/storage/document_store.py` | SQLite, cascade deletes, content-hash uniqueness. |
| Dense index | `app/retrieval/dense.py` | `IndexIDMap2(IndexFlatIP)`, remove by id, atomic save. |
| BM25 | `app/retrieval/bm25.py` | Own implementation with Lucene IDF (the library's IDF breaks on small corpora). |
| Fusion | `app/retrieval/fusion.py` | RRF, deterministic tie-break. |
| Reranker | `app/retrieval/reranker.py` | Protocol; local cross-encoder; failure → fusion order. |
| Hybrid retriever + gate | `app/retrieval/hybrid.py` | Scores from every stage are kept per chunk; `evidence_check` decides abstention. |
| Generator | `app/generation/generator.py` | Numbered context, citation parsing, abstention detection, critique-guided regeneration. |
| Judge / critic | `app/generation/judge.py`, `critic.py` | Scores from the LLM, verdict in code; critic only on FAIL, falls back to judge findings. |
| Graph | `app/pipeline/graph.py` | LangGraph nodes + pure routing functions + best-attempt selection. |
| Services / container | `app/services/` | Orchestration and dependency wiring; routes stay thin. |
| Tracing / metrics | `app/observability/` | OTel-shaped spans, JSON logs, p50/p95. |
| Eval | `app/evaluation/`, `eval/` | Golden set, deterministic metrics, independent evaluator, judge calibration. |

## 4. Algorithms you must be able to explain

**BM25.** For query terms t: `Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, k1 = 1.5, b = 0.75. *tf saturation* (k1) stops a term repeated 50 times from dominating; *length normalization* (b) stops long chunks from winning by size. IDF = `ln(1 + (N − n + 0.5)/(n + 0.5))`, always positive. Why I wrote it myself: the classic Okapi IDF goes ≤ 0 when a term is in half the chunks, so on one small uploaded document BM25 returned nothing. A unit test pins that.

**Reciprocal Rank Fusion.** `RRF(d) = Σ_r 1/(k + rank_r(d))`, k = 60. Rank-based, so no score normalization; a document ranked moderately by *both* retrievers beats one ranked high by only one. k damps the advantage of rank 1 over rank 5.

**Bi-encoder vs cross-encoder.** A bi-encoder embeds query and document separately, so it's fast and cacheable; that's what retrieves. A cross-encoder reads them together with full attention, which is more accurate but costs O(candidates) forward passes, so it only reranks the shortlist.

**Cosine via inner product.** Embeddings are L2-normalized, so the inner product equals cosine; `IndexFlatIP` is exact search.

**Recursive chunking.** Split on the coarsest separator whose pieces fit, recurse on oversized pieces with finer separators, then pack pieces greedily into windows with a word-aligned overlap tail.

**Reliability loop as a state machine.** States: generate, judge, critic, regenerate, finalize. Transition `judge → critic` only if `verdict == FAIL ∧ retries < max ∧ no judge error`. It terminates because retries strictly increase, and there's a recursion limit as a backstop. Worst case 1 + 3·max_retries LLM calls.

**Best-attempt selection.** `max` over attempts of `(PASS, grounded, aggregate, −attempt)`. Grounded means no faithfulness or unsupported-claims failure. The aggregate is `0.5·F + 0.25·R + 0.25·C`: faithfulness is weighted highest because an unfaithful answer is worse than an incomplete one.

## 5. Why each technology

- **FAISS, not a hosted vector DB:** thousands of vectors; exact search takes about 1 ms in-process; no network or ops. The index is derivable from SQLite, so moving to pgvector or Qdrant later changes one class.
- **SQLite:** transactional metadata store with zero ops. Swapping to Postgres is what you'd do for multiple replicas.
- **LangGraph:** the loop *is* a state machine with conditional edges. LangGraph makes the edges explicit and inspectable and gives a recursion limit; I don't use LangChain's retrieval abstractions.
- **Groq:** low latency for multi-call pipelines; a free tier to demo with. Behind a protocol, so it's replaceable.
- **Three different LLMs:** generator `gpt-oss-120b`, judge `qwen3.8-27b`, critic `gpt-oss-20b`. That reduces self-preference bias and spreads per-model rate limits. The eval uses an evaluator that must differ from the pipeline judge.
- **Streamlit:** fastest way to ship a chat UI in Python; it shows only the answer with inline hover citations and talks to the API over HTTP, so it's replaceable (e.g. by a React front end). Diagnostics live in backend logs, not in the user's view.
- **FastAPI + Pydantic:** typed contracts, validation, OpenAPI for free, threadpool for sync handlers.

## 6. Engineering trade-offs to bring up yourself

1. **Reliability vs latency/cost.** The loop adds a judge call to every answer and roughly 2 calls per retry. It is conditional, so the cost is paid only when the judge finds a problem. See the cost table in the eval report.
2. **Strict vs lenient verdicts.** `FAIL_ON_UNSUPPORTED_CLAIMS` catches hallucinations the numeric score hides, but it raises the retry rate. It's configurable per request.
3. **Reranker.** It helped the abstention signal, not ranking, on this corpus, and costs about 1 s of CPU. I measured it and kept it, and I say so.
4. **Local models vs hosted.** No API cost and data stays local, but the API container uses ~650–800 MiB of RAM (measured) and the first query is slow without warm-up.
5. **Simplicity vs scale.** Single process, in-memory BM25, SQLite: right for one user; section 8 covers what changes.

## 7. Fifteen likely questions with strong answers

**1. Why not just use a bigger model instead of a judge loop?**
A bigger model hallucinates less often but still does it, and it tells you nothing about *when* it does. The judge makes reliability observable per answer (scores plus unsupported claims), which you can threshold, log, alert on and evaluate. The loop also only costs extra when the judge flags a problem.

**2. Isn't an LLM judging an LLM circular?**
Partly, and I mitigate it three ways. The judge is a different model family from the generator. The verdict is computed in code from scores. And I calibrated the judge on seeded faults (wrong numbers, outside knowledge, contradictions, over-claims, incomplete and off-topic answers): see the calibration table in the report. The evaluation also uses a fourth role, an independent evaluator, plus deterministic metrics (fact recall, abstention) that no LLM decides.

**3. Your v1 made answers worse. Why?**
The always-on synthesizer rewrote answers that were already correct. Rewriting has only downside when nothing is wrong: it adds paraphrase drift and invites new claims. The fix is to intervene only on detected failure and never return an attempt scored worse than the original.

**4. How did you choose the thresholds?**
The evidence gate is calibrated on real queries: on-topic rerank logits were at least −3.2 and off-topic ones about −11, so −5 sits in the gap. Judge thresholds (F ≥ 0.8, R ≥ 0.7, C ≥ 0.6) favor faithfulness over completeness. With labelled data I'd pick them by sweeping to trade retry rate against evaluator faithfulness, and they're overridable per request.

**5. What happens when the judge is wrong?**
A false FAIL costs a retry, but best-attempt selection means the user still gets the original if the retry is worse. A false PASS returns the answer, and the judge scores and trace are visible, so it's auditable. If the judge errors or returns bad JSON, the answer is returned flagged *unverified* rather than blocking.

**6. Why RRF instead of weighting scores?**
Cosine similarity and BM25 are on different, query-dependent scales. Score fusion needs normalization and tuned weights; RRF needs only ranks and one robust constant.

**7. How do you prevent infinite loops?**
A retry counter incremented on every FAIL edge, a routing function that stops at `max_retries` (validated to at most 5), and LangGraph's `recursion_limit` as an independent backstop. There's a test with a judge that always fails, which asserts exactly 3 judge calls at `max_retries = 2`.

**8. How do you handle a question the documents can't answer?**
In three layers. The evidence gate refuses before generation if retrieval is weak. The generator must output a fixed sentence when the sources lack the answer. And the judge doesn't penalize "the sources don't cover X", so honest partial answers pass. The eval has an *unanswerable* category measuring correct abstention and a false-abstention metric for answerable questions.

**9. How do citations work, and how do you stop fake ones?**
Chunks are numbered in the prompt. Afterwards a regex extracts `[n]`, `[1, 2]` and model-native markers like `【4†L1-L3】`, drops out-of-range numbers from the text, and maps valid ones to document/page/chunk. The judge also penalizes wrong citation numbers.

**10. What would you change for 10,000 users?**
Move state out of the process: Postgres with pgvector or a vector DB, OpenSearch for BM25, Prometheus for metrics, Redis for caches and rate-limit budgets. Put embedding and reranking in an inference service with batching, make ingestion asynchronous through a queue, add per-tenant quotas because the loop multiplies LLM spend, and stream the first answer while verification runs.

**11. How would you add multi-tenancy?**
A `tenant_id` on every row, filtered *inside* the retrieval query (filtering after top-k leaks and truncates), Postgres row-level security as defense in depth, tenant-scoped cache keys, and OIDC/JWT auth in a FastAPI dependency.

**12. Why is the reranker still on if it didn't improve recall?**
Because I measured two different things. Ranking: neutral at k = 5, slightly worse at k = 3 on this corpus. Abstention: its logits separate on-topic from off-topic far more clearly than cosine, and that's what drives the zero-LLM-call refusal. It's one flag to disable, and on a paraphrase-heavy corpus I'd expect the ranking result to change. I'd re-measure there.

**13. How do you test code that calls an LLM?**
Dependency injection: `LLMClient` is a protocol, and tests inject a scripted `FakeLLM` keyed by purpose (generate/judge/critic), plus a hashing embedder, so there's no network and no model download. Graph routing, retry limits, best-attempt selection, abstention and API error mapping are all deterministic tests. Real-model behavior is covered by the eval harness and judge calibration.

**14. What's in the trace, and how would it go to production observability?**
Spans for each stage (name, offset, duration, status, attributes), LLM calls, tokens, retries, and throttle time, keyed by the request id that is also in the `x-request-id` header and every JSON log line. The shape matches OpenTelemetry, so exporting means mapping `Span` to OTel spans with GenAI semantic-convention attributes; nothing in the pipeline changes.

**Bonus, and very likely: "Did the adversarial loop actually improve results?"**
Answer with the numbers, not a sales pitch. "On my 22-question golden set it never triggered, and quality was identical to baseline: fact recall 0.898 in both, evaluator faithfulness 1.00 vs 0.99. On six compound questions designed to bait outside knowledge, it triggered once and fixed the pipeline judge's objections, but the independent evaluator still flagged one claim. So on this data it's insurance plus a verification signal, at about 2× LLM calls. Compared with v1, where the always-on version measurably hurt, that's the right direction. To show real gains I'd need harder data: paraphrased and multi-document questions, and a weaker or cheaper generator where hallucinations are more common. That's also where a conditional loop saves the most money versus always using a bigger model."

**15. What was the hardest bug?**
There are two good stories. First: BM25 returned nothing on a one-document corpus because the library's IDF goes negative; I found it with a unit test and reimplemented it with Lucene's IDF. Second: the judge listed unsupported claims but still scored faithfulness 0.95, so hallucinated answers passed. Worse, when every attempt failed, best-attempt selection preferred the fluent hallucination over the honest answer. The fixes: any unsupported claim fails the verdict, and grounding became a hard ranking key.

## 8. Scaling discussion (short form)

1. **State:** SQLite/FAISS/BM25/metrics are per process. Move them to Postgres + pgvector (or Qdrant), OpenSearch and Prometheus.
2. **Compute:** CPU embedding and reranking move to a batched inference service or GPU.
3. **LLM budget:** per-tenant quotas, a Redis token bucket shared across replicas, answer caching keyed by `(query, corpus version, settings)`, streaming.
4. **Ingestion:** object storage, then a queue, then workers; `202 Accepted` with polling; idempotent via content hash.
5. **Evaluation in CI:** run the golden set on every prompt or model change; fail the build if faithfulness or abstention regresses.

## 9. Failure-mode discussion

Walk through the table in `docs/ARCHITECTURE.md` §3.7. Key phrases: "degrade, don't die"; "abstain before you hallucinate"; "unverified is a state, not an error"; "a retry can never make the answer worse".

## 10. What I would improve with more time

- A bigger, more varied golden set (several documents, paraphrased questions, adversarial phrasing) and confidence intervals via bootstrap.
- Threshold tuning from data (sweep vs retry rate and evaluator faithfulness).
- Claim-level verification: split the answer into atomic claims and check each against its cited chunk (NLI model or LLM), instead of one holistic judge call.
- Streaming the draft answer while verification runs, with the badge updating in place.
- Async ingestion queue, Postgres/pgvector, auth and multi-tenancy.
- OpenTelemetry exporter and a Grafana dashboard for retry rate, throttle time and verdict distribution.
- OCR for scanned PDFs; table-aware chunking.

## 11. Live code walkthrough: open these files

1. **`app/pipeline/graph.py`: `route_after_judge` and `select_best_attempt`.**
   Say: "This is the core. Routing is a pure function of the state, so it's unit-tested. Note the three exits from judge: PASS, retries exhausted, judge error. Selection ranks by PASS, then grounded, then score, and ties go to the earliest attempt, so a retry can't make things worse."
2. **`app/generation/judge.py`: `apply_thresholds`.**
   Say: "The LLM measures; code decides. The unsupported-claims rule exists because I saw the judge list hallucinations and still score 0.95."
3. **`app/retrieval/hybrid.py`: `retrieve` and `evidence_check`.**
   Say: "Two candidate generators, RRF, then the reranker on a shortlist. Every stage's score survives into the response. Each failure has a fallback. The evidence gate runs before any LLM call."
4. **`app/retrieval/bm25.py`: `_BM25`.**
   Say: "Thirty lines of BM25 with Lucene's IDF: here's the bug it fixes and the test that pins it."
5. **`app/services/document_service.py`: `ingest` and `sync_index`.**
   Say: "Embed before the transaction so a model failure leaves no partial state; compensating delete if the FAISS write fails; self-heal on startup, which makes the index disposable on ephemeral disks."
6. **`app/core/llm.py`: `GroqLLM.complete`.**
   Say: "The provider is behind a protocol. I own the retry loop so I can honor `Retry-After` and *measure* throttle time. That separated our latency from free-tier rate limits in the eval."
7. **`app/evaluation/harness.py`: module docstring and `evaluate_item`.**
   Say: "Isolated index, same retrieval for both modes, an independent evaluator so the loop isn't graded by its own judge, deterministic metrics first, and resumable runs."
8. **`tests/test_pipeline.py`.**
   Say: "Scripted LLM by purpose. These tests prove the call sequences: PASS means exactly generate + judge; always-FAIL means exactly three judges at max_retries = 2; and a worse retry keeps the original."
