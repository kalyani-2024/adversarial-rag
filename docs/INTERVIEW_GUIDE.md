# Interview guide: RAG Reliability Lab

How to explain this project in a technical interview: what to say, what to show, and how to answer the follow-ups. Every number here comes from `eval/results/`, from a test in `tests/`, or from a measurement described in the README. If an interviewer asks "how do you know?", you can point at the file.

---

## 1. The 60-second explanation

> "I built a RAG system that doesn't take its own answers on trust. Retrieval is hybrid: dense FAISS search plus BM25, fused with Reciprocal Rank Fusion, then reranked by a cross-encoder. If the best evidence is weak, it refuses to answer, without calling the LLM. Otherwise it streams an answer that has to cite numbered sources, and a separate judge model scores faithfulness, relevance and completeness. **Only if the answer fails** does an adversarial critic list the unsupported claims; the draft is replaced by a regenerated answer. It retries at most twice and returns the best attempt, never a worse one.
>
> The design came from a measurement. My first version always ran a critique-and-rewrite step, and my own eval showed it made answers *less* faithful at 5× the latency. So v2 makes the loop conditional, puts the PASS/FAIL decision in code instead of the LLM, and I evaluate baseline against adversarial on a golden dataset with an independent evaluator model. The honest result: with a strong generator, most answers already pass, so the loop rarely fires and doesn't measurably change quality on my test sets. What it buys is a calibrated per-answer verification signal; the judge caught 6 of 6 seeded faults.
>
> On the engineering side, it takes 100 MB uploads, indexed in the background with progress while chat keeps priority. BM25 is a segmented inverted index, 1 ms at 100k chunks. Everything is traced and logged per request. It's FastAPI + LangGraph + Streamlit, with 127 tests and a Docker image."

## 2. The 5-minute architecture walkthrough

Draw this (or open the README diagram):

```
Upload ≤100 MB → validate (type, size, SHA-256 dup) → 202 → background worker:
   parse (page-aware) → clean → chunk (recursive, no page crossing) → embed in batches (progress)
   → SQLite (source of truth) + FAISS + BM25 segment for this document
   (pauses between batches while any query runs)

Query → [rewrite only if follow-up/filler] → dense top-20 ∥ BM25 top-20 → RRF → cross-encoder top-10 → top-k
      → evidence gate ──fail──► "I don't have enough evidence…" (0 LLM calls)
      → send numbered sources → stream draft tokens with [n] citations
      → judge (scores) → verdict in code
            PASS → final
            FAIL → critic → reset draft → stream regenerated answer → judge  (≤ MAX_RETRIES, best attempt wins)
```

Talk track, one breath per box:

1. **Ingestion.** "Uploads return 202 immediately; one background worker indexes them and reports stage and progress. PDF pages are parsed separately, so every chunk keeps its page number for citations. Chunking is recursive on paragraph, then line, then sentence, then word, so chunks never cut words or cross a page. SQLite is the source of truth; FAISS only holds vectors keyed by chunk id, so I can delete a document and rebuild the index if the file is lost. Duplicates are caught by SHA-256 with a unique constraint."
2. **Retrieval.** "Dense catches paraphrase, BM25 catches exact numbers and identifiers. Their scores aren't comparable, so I fuse *ranks* with RRF. The cross-encoder then rescores the top 10 pairs jointly, capped at 256 tokens, which I measured to be as accurate as 20 × 512 and 3× cheaper."
3. **Evidence gate.** "Before any LLM call, I check the best rerank score against a threshold I calibrated on real data: on-topic questions scored above −3.2 and off-topic ones −9 to −11. Below −5 we abstain. Cheapest and safest answer."
4. **Generation and streaming.** "Sources are numbered and sent to the client first; then tokens stream, so citation badges appear while the answer is written. Afterwards I parse the citations, drop ones pointing at sources that don't exist, and map the rest to document/page."
5. **Reliability loop.** "The judge returns numbers only; `apply_thresholds` decides. If FAIL, the critic returns a structured list of problems, the client gets a `reset` event, and the rewritten answer streams in. The loop is an explicit LangGraph conditional edge with a retry counter plus a recursion limit."
6. **Observability.** "The user sees only the chat. Operators get JSON log lines per `request_id`: retrieval scores per chunk, each attempt's verdict and failed checks, per-stage latency, tokens, and time lost to rate limiting. `/metrics` aggregates p50/p95 and the retry rate."
7. **Evaluation.** "A golden set of 22 questions across four categories plus 6 compound stress questions. It measures retrieval recall deterministically and compares baseline with adversarial on fact recall, abstention, and faithfulness from an *independent* evaluator model."

## 3. Results, and how to talk about them

Numbers from `eval/results/` (runs on 2026-09-24) and the README's measurements. Know these cold, *and* know their limits.

| Finding | Number | What to say |
|---|---|---|
| Judge calibration | 6/6 seeded faults caught, 0/4 false alarms | "The signal the loop depends on is real, on a small calibration set." |
| Loop activity | fired on 0/17 answered golden questions, 1/6 compound stress questions | "Conditional means it stays out of the way when the answer is already grounded." |
| Quality, baseline vs adversarial | fact recall 0.898 vs 0.898; evaluator faithfulness 1.000 vs 0.994 | "No measurable difference at this n. It no longer *hurts* like v1 did, but I can't claim it improves quality here." |
| Cost | ~2× LLM calls and tokens; net latency 1.17× mean, 1.20× p95 | "Every answer pays for one judge call; retries only when needed. Streaming hides most of it: the text shows before the judge finishes." |
| Retrieval | BM25 alone best (hit@5 0.947); hybrid 0.895; dense 0.842; rerank doesn't change hit rates | "My questions reuse the paper's wording, which favors BM25. I'd re-measure with paraphrased questions before claiming hybrid wins." |
| Reranker cost | 20 × 512: ~2 s; 10 × 256: ~0.6 s, same hit@k, MRR 0.800 vs 0.774 | "I cut its cost 3× by measuring what actually mattered." |
| Abstention | 3/3 unanswerable refused with 0 LLM calls; 2/19 answerable wrongly refused | "The gate is cheap and safe; the false refusals are the generator being conservative on two 'bait' questions." |
| Scale | BM25 1.1 ms at 100k chunks; small add next to 100k chunks 30 ms (was a 33 s rebuild) | "Query cost follows the postings, not the corpus; updates touch one segment." |
| Chat during indexing | first token 2.1 s while indexing 1,710 chunks vs 3.0 s idle | "Queries take priority; the indexer yields between batches." |
| Indexing throughput | ~21 chunks/s on the laptop CPU; 1 MB of text ≈ 1,700 chunks ≈ 80 s | "Cost depends on text volume, not megabytes. For big text corpora you need a GPU or hosted embeddings." |

How to frame it: *"My eval showed the always-on critic hurt, so I made it conditional and measured again. Now it doesn't hurt, it costs about 2× calls, and on my small test sets it rarely triggers. The next step would be a harder, paraphrased, multi-document eval, plus claim-level verification, to find where it pays off."* Interviewers value that more than a suspiciously large improvement number.

## 4. Components and what to say about each

| Component | File | One-liner |
|---|---|---|
| Config | `app/core/config.py` | Every knob is an env var; secrets are `SecretStr`; validated at startup. |
| LLM client | `app/core/llm.py` | Protocol + Groq implementation; `complete` and `stream`; own retry loop honoring `Retry-After`, so throttle time is measured; retries only while opening a stream. |
| Parsers / cleaning / chunking | `app/ingestion/` | Page-aware parsing, NFKC + PDF hyphenation repair, recursive chunking with word-boundary overlap. |
| Document store | `app/storage/document_store.py` | SQLite (WAL): documents with status/error, chunks, cascade deletes, content-hash uniqueness, restart recovery. |
| Document service | `app/services/document_service.py` | `submit` (202, background worker, batched embedding, progress, cancel) and `ingest` (synchronous); compensating rollback; index self-heal. |
| Priority gate | `app/services/priority.py` | Counts running queries; the indexer waits for zero before each batch (max 10 s). |
| Dense index | `app/retrieval/dense.py` | `IndexIDMap2(IndexFlatIP)`, remove by id, atomic save. |
| BM25 | `app/retrieval/bm25.py` | Own BM25 with Lucene IDF; inverted index in per-document segments with global stats; copy-on-write snapshots. |
| Fusion | `app/retrieval/fusion.py` | RRF, deterministic tie-break. |
| Reranker | `app/retrieval/reranker.py` | Protocol; local cross-encoder (10 candidates × 256 tokens); failure → fusion order. |
| Hybrid retriever + gate | `app/retrieval/hybrid.py` | Scores from every stage kept per chunk; incremental BM25 refresh; `evidence_check` decides abstention. |
| Generator | `app/generation/generator.py` | Numbered context, optional token streaming, citation parsing, abstention detection, critique-guided regeneration. |
| Judge / critic | `app/generation/judge.py`, `critic.py` | Scores from the LLM, verdict in code; critic only on FAIL, falls back to judge findings. |
| Graph | `app/pipeline/graph.py` | LangGraph nodes, pure routing functions, best-attempt selection, streaming events via an `emit` callback. |
| API | `app/api/routes/` | Thin routes; `/query/stream` runs the same pipeline in a worker thread and drains an event queue as SSE. |
| Logs / tracing / metrics | `app/observability/` | OTel-shaped spans, per-query JSON log records, p50/p95, throttle time. |
| UI | `ui/streamlit_app.py`, `ui/client.py` | Streaming chat, inline citation popovers with full passages, settings sidebar, live indexing progress; HTTP only. |
| Eval | `app/evaluation/`, `eval/` | Golden + stress sets, deterministic metrics, independent evaluator, judge calibration, rescoring. |

## 5. Algorithms you must be able to explain

**BM25.** For query terms t: `Σ IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, k1 = 1.5, b = 0.75. *tf saturation* (k1) stops a term repeated 50 times from dominating; *length normalization* (b) stops long chunks from winning by size. IDF = `ln(1 + (N − n + 0.5)/(n + 0.5))`, always positive. Why I wrote it myself: the classic Okapi IDF goes ≤ 0 when a term is in half the chunks, so on one small uploaded document BM25 returned nothing. A unit test pins that.

**Inverted index with segments.** Each term maps to posting arrays (chunk positions, term frequencies); a query adds `idf · tf·(k1+1)/(tf + norm)` into an accumulator only for chunks in its terms' postings, then takes the top k with `argpartition`. Each document is its own segment; document frequencies, chunk count and total length are summed globally, so IDF and avgdl are identical to a single index (a test compares them). Add or delete a document = build or drop one segment + update the global counts, published as a new immutable snapshot, so readers never lock.

**Reciprocal Rank Fusion.** `RRF(d) = Σ_r 1/(k + rank_r(d))`, k = 60. Rank-based, so no score normalization; a document ranked moderately by *both* retrievers beats one ranked high by only one. k damps the advantage of rank 1 over rank 5.

**Bi-encoder vs cross-encoder.** A bi-encoder embeds query and document separately, so it's fast and cacheable; that's what retrieves. A cross-encoder reads them together with full attention, which is more accurate but costs O(candidates) forward passes, so it only reranks the shortlist. Its cost also grows with sequence length, which is why capping at 256 tokens halved the time.

**Cosine via inner product.** Embeddings are L2-normalized, so the inner product equals cosine; `IndexFlatIP` is exact search.

**Recursive chunking.** Split on the coarsest separator whose pieces fit, recurse on oversized pieces with finer separators, then pack pieces greedily into windows with a word-aligned overlap tail.

**Reliability loop as a state machine.** States: generate, judge, critic, regenerate, finalize. Transition `judge → critic` only if `verdict == FAIL ∧ retries < max ∧ no judge error`. It terminates because retries strictly increase, and there's a recursion limit as a backstop. Worst case 1 + 3·max_retries LLM calls.

**Best-attempt selection.** `max` over attempts of `(PASS, grounded, aggregate, −attempt)`. Grounded means no faithfulness or unsupported-claims failure. The aggregate is `0.5·F + 0.25·R + 0.25·C`: faithfulness is weighted highest because an unfaithful answer is worse than an incomplete one.

**Streaming protocol.** `status` → `sources` → `token`… → (`reset` → `token`…)* → `final`. The client renders drafts with citation badges from `sources`, clears on `reset`, and always replaces the text with `final`, which may be an earlier attempt because of best-attempt selection.

**Query-priority gate.** A counter under a condition variable: queries increment/decrement it; the indexer calls `wait_until_idle(10 s)` before each embedding batch. That's a simple form of priority scheduling with a starvation bound.

## 6. Why each technology

- **FAISS, not a hosted vector DB:** thousands to ~10⁵ vectors; exact search takes about 1 ms in-process; no network or ops. The index is derivable from SQLite, so moving to pgvector or Qdrant later changes one class.
- **SQLite:** transactional metadata store with zero ops; WAL lets the API read while the worker writes. Swap to Postgres for multiple replicas.
- **LangGraph:** the loop *is* a state machine with conditional edges. LangGraph makes the edges explicit and inspectable, gives a recursion limit, and its `configurable` let me thread the streaming callback through without changing node signatures. I don't use LangChain's retrieval abstractions.
- **Groq:** low latency for multi-call pipelines (the whole answer arrives in ~0.3 s after the first token); a free tier to demo with. Behind a protocol, so it's replaceable.
- **Three different LLMs:** generator `gpt-oss-120b`, judge `qwen3.8-27b`, critic `gpt-oss-20b`. That reduces self-preference bias and spreads per-model rate limits. The eval uses an evaluator that must differ from the pipeline judge.
- **Server-Sent Events, not WebSockets:** the stream is one-directional, SSE works over plain HTTP through proxies, and it fits a normal POST handler.
- **Streamlit:** fastest way to ship a chat UI in Python; it shows only the answer with inline citations and talks to the API over HTTP, so it's replaceable (e.g. by a React front end). Diagnostics live in backend logs, not in the user's view.
- **FastAPI + Pydantic:** typed contracts, validation, OpenAPI for free, threadpool for sync handlers, `StreamingResponse` for SSE.

## 7. Engineering trade-offs to bring up yourself

1. **Reliability vs latency/cost.** The loop adds a judge call to every answer and roughly 2 calls per retry. It is conditional, so the cost is paid only when the judge finds a problem, and streaming shows the draft before the judge finishes.
2. **Streaming a draft that might change.** The alternative is showing nothing until verified. I chose visible progress plus an explicit "Checking…/Refining…" status, and the final event is always authoritative.
3. **Strict vs lenient verdicts.** `FAIL_ON_UNSUPPORTED_CLAIMS` catches hallucinations the numeric score hides, but it raises the retry rate. It's configurable per request and in the sidebar.
4. **Reranker.** On this corpus it helped the abstention signal, not ranking. I measured it, cut its cost from ~2 s to ~0.6 s without losing accuracy, and kept it, and I say so.
5. **Background indexing in-process vs a queue.** In-process is zero infrastructure and keeps chat responsive via the priority gate, but it's tied to the API's CPU and loses in-flight uploads on restart (marked failed). A queue with workers is the next step.
6. **Local models vs hosted.** No API cost and data stays local, but the API container uses ~650–800 MiB of RAM (measured) and CPU embedding is ~21 chunks/s, so big text uploads are slow.
7. **Simplicity vs scale.** Single process, in-memory BM25, SQLite: right for one user; section 9 covers what changes.

## 8. Likely questions with strong answers

**1. Why not just use a bigger model instead of a judge loop?**
A bigger model hallucinates less often but still does it, and it tells you nothing about *when* it does. The judge makes reliability observable per answer (scores plus unsupported claims), which you can threshold, log, alert on and evaluate. The loop also only costs extra when the judge flags a problem.

**2. Isn't an LLM judging an LLM circular?**
Partly, and I mitigate it three ways. The judge is a different model family from the generator. The verdict is computed in code from scores. And I calibrated the judge on seeded faults (wrong numbers, outside knowledge, contradictions, over-claims, incomplete and off-topic answers). The evaluation also uses a fourth role, an independent evaluator, plus deterministic metrics (fact recall, abstention) that no LLM decides.

**3. Your v1 made answers worse. Why?**
The always-on synthesizer rewrote answers that were already correct. Rewriting has only downside when nothing is wrong: it adds paraphrase drift and invites new claims. The fix is to intervene only on detected failure and never return an attempt scored worse than the original.

**4. Did the adversarial loop actually improve results?** *(very likely)*
Answer with the numbers, not a sales pitch. "On my 22-question golden set it never triggered, and quality was identical to baseline: fact recall 0.898 in both, evaluator faithfulness 1.00 vs 0.99. On six compound questions designed to bait outside knowledge, it triggered once and fixed the pipeline judge's objections, but the independent evaluator still flagged one claim. So on this data it's insurance plus a verification signal, at about 2× LLM calls. Compared with v1, where the always-on version measurably hurt, that's the right direction. To show real gains I'd need harder data: paraphrased and multi-document questions, and a weaker or cheaper generator where hallucinations are more common. That's also where a conditional loop saves the most money versus always using a bigger model."

**5. How did you choose the thresholds?**
The evidence gate is calibrated on real queries: on-topic rerank logits were at least −3.2 and off-topic ones −9 to −11, so −5 sits in the gap. I re-checked this after changing the reranker settings. Judge thresholds (F ≥ 0.8, R ≥ 0.7, C ≥ 0.6) favor faithfulness over completeness. With labelled data I'd pick them by sweeping to trade retry rate against evaluator faithfulness, and they're overridable per request.

**6. What happens when the judge is wrong?**
A false FAIL costs a retry, but best-attempt selection means the user still gets the original if the retry is worse. A false PASS returns the answer, and the judge scores are logged per request, so it's auditable. If the judge errors or returns bad JSON, the answer is returned flagged *unverified* rather than blocking.

**7. How can you stream an answer you haven't verified yet?**
I stream the *draft* and make verification able to overwrite it. Sources go out first so citation badges render live. If the judge fails the draft, a `reset` event clears it and the regenerated answer streams in. The `final` event carries the answer the pipeline actually selected, which may be an earlier attempt, and the client always renders that. Both endpoints run the same graph; the streaming one just passes an `emit` callback, so there's no duplicated logic.

**8. How do you handle a 100 MB upload without freezing the app?**
Validate type, size and duplicates synchronously, then register the document as `processing` and return 202. One background worker parses, chunks and embeds in batches, reporting progress the UI polls. Chat keeps priority: the worker pauses between batches while any query runs, and I measured no slowdown in time-to-first-token during indexing. Failures stay visible as `failed` with the error; deleting mid-way cancels; a restart marks interrupted uploads failed. And the honest caveat: embedding is ~21 chunks/s on my CPU, so what matters is the amount of text. 1 MB of text is about 80 s; a huge text corpus needs a GPU or a hosted embedding API.

**9. How did you make it feel responsive?**
Measure first, then fix the biggest item. Timing the stream showed ~2 s of every UI request was Windows resolving `localhost` via IPv6 before falling back to IPv4, so the client uses `127.0.0.1` (6 ms). The CPU reranker took ~2 s; 10 candidates × 256 tokens cut it to ~0.6 s with no loss in hit@k. Streaming puts text on screen after the model's ~1–2 s reasoning phase instead of after verification. BM25 moved from scanning every chunk to a segmented inverted index.

**10. Why RRF instead of weighting scores?**
Cosine similarity and BM25 are on different, query-dependent scales. Score fusion needs normalization and tuned weights; RRF needs only ranks and one robust constant.

**11. How do you prevent infinite loops?**
A retry counter incremented on every FAIL edge, a routing function that stops at `max_retries` (validated to at most 5), and LangGraph's `recursion_limit` as an independent backstop. There's a test with a judge that always fails, which asserts exactly 3 judge calls at `max_retries = 2`.

**12. How do you handle a question the documents can't answer?**
In three layers. The evidence gate refuses before generation if retrieval is weak. The generator must output a fixed sentence when the sources lack the answer. And the judge doesn't penalize "the sources don't cover X", so honest partial answers pass. The eval has an *unanswerable* category measuring correct abstention and a false-abstention metric for answerable questions.

**13. How do citations work, and how do you stop fake ones?**
Chunks are numbered in the prompt. Afterwards a regex extracts `[n]`, `[1, 2]` and model-native markers like `【4†L1-L3】`, drops out-of-range numbers from the text, and maps valid ones to document/page/chunk. In the UI each marker is a badge whose popover shows the full source passage. The judge also penalizes wrong citation numbers.

**14. What would you change for 10,000 users?**
Move state out of the process: Postgres with pgvector or a vector DB, OpenSearch for BM25, Prometheus for metrics, Redis for caches, rate-limit budgets and ingestion progress. Put embedding and reranking in an inference service with batching, move ingestion to object storage + a queue + GPU workers (the 202/poll API stays the same), and add per-tenant quotas because the loop multiplies LLM spend.

**15. How would you add multi-tenancy?**
A `tenant_id` on every row, filtered *inside* the retrieval query (filtering after top-k leaks and truncates), Postgres row-level security as defense in depth, tenant-scoped cache keys, and OIDC/JWT auth in a FastAPI dependency.

**16. Why is the reranker still on if it didn't improve recall?**
Because I measured two different things. Ranking: no change in hit rates on this corpus, with slightly better MRR at the current settings. Abstention: its logits separate on-topic from off-topic far more clearly than cosine, and that drives the zero-LLM-call refusal. It's one flag to disable, and on a paraphrase-heavy corpus I'd expect the ranking result to change. I'd re-measure there.

**17. How do you test code that calls an LLM?**
Dependency injection: `LLMClient` is a protocol, and tests inject a scripted `FakeLLM` keyed by purpose (generate/judge/critic, including a streaming variant), plus a hashing embedder, so there's no network and no model download. Graph routing, retry limits, best-attempt selection, abstention, SSE event order, background ingestion, cancellation and API error mapping are all deterministic tests. Real-model behavior is covered by the eval harness and judge calibration.

**18. What's in the logs, and how would it go to production observability?**
Per request: the query (optional), retrieval scores per chunk, each attempt's verdict and failed checks, per-stage latency, LLM calls, tokens and throttle time, all keyed by the `request_id` that is also in the `x-request-id` header. Spans have OpenTelemetry's shape, so exporting means mapping `Span` to OTel spans with GenAI semantic-convention attributes; nothing in the pipeline changes.

**19. What was the hardest bug?**
There are two good stories. First: BM25 returned nothing on a one-document corpus because the library's IDF goes negative; I found it with a unit test and reimplemented it with Lucene's IDF. Second: the judge listed unsupported claims but still scored faithfulness 0.95, so hallucinated answers passed. Worse, when every attempt failed, best-attempt selection preferred the fluent hallucination over the honest answer. The fixes: any unsupported claim fails the verdict, and grounding became a hard ranking key.

## 9. Scaling discussion (short form)

1. **State:** SQLite/FAISS/BM25/metrics/ingestion worker are per process. Move them to Postgres + pgvector (or Qdrant), OpenSearch, Prometheus, and Redis for progress and priority signals.
2. **Compute:** CPU embedding (~21 chunks/s) and reranking (~0.6 s) move to a batched inference service or GPU, or to hosted models.
3. **LLM budget:** per-tenant quotas, a Redis token bucket shared across replicas, answer caching keyed by `(query, corpus version, settings)`; streaming already hides first-answer latency.
4. **Ingestion:** object storage → queue → GPU workers; the 202 + `GET /documents/{id}` contract is already in place; idempotent via content hash.
5. **Evaluation in CI:** run the golden set on every prompt or model change; fail the build if faithfulness or abstention regresses.

## 10. Failure-mode discussion

Walk through the table in `docs/ARCHITECTURE.md` §3.11. Key phrases: "degrade, don't die"; "abstain before you hallucinate"; "unverified is a state, not an error"; "a retry can never make the answer worse"; "a failed upload is visible and retryable"; "never retry a stream after tokens went out".

## 11. What I would improve with more time

- A bigger, more varied golden set (several documents, paraphrased questions, adversarial phrasing) and confidence intervals via bootstrap.
- Threshold tuning from data (sweep vs retry rate and evaluator faithfulness).
- Claim-level verification: split the answer into atomic claims and check each against its cited chunk (NLI model or LLM), instead of one holistic judge call. That would also allow streaming verification sentence by sentence.
- Out-of-process ingestion (object storage + queue + GPU embedding workers), Postgres/pgvector, auth and multi-tenancy.
- OCR for scanned PDFs; table-aware chunking; a better PDF text extractor (pypdf drops spaces in some files).
- OpenTelemetry exporter and a Grafana dashboard for retry rate, throttle time and verdict distribution.

## 12. Live code walkthrough: open these files

1. **`app/pipeline/graph.py`: `route_after_judge` and `select_best_attempt`.**
   Say: "This is the core. Routing is a pure function of the state, so it's unit-tested. Note the three exits from judge: PASS, retries exhausted, judge error. Selection ranks by PASS, then grounded, then score, and ties go to the earliest attempt, so a retry can't make things worse. The nodes also emit streaming events through an optional callback."
2. **`app/generation/judge.py`: `apply_thresholds`.**
   Say: "The LLM measures; code decides. The unsupported-claims rule exists because I saw the judge list hallucinations and still score 0.95."
3. **`app/retrieval/hybrid.py`: `retrieve` and `evidence_check`.**
   Say: "Two candidate generators, RRF, then the reranker on a shortlist of 10. Every stage's score survives into the logs. Each failure has a fallback. The evidence gate runs before any LLM call."
4. **`app/retrieval/bm25.py`: `_Segment.build` and `BM25Index.search`.**
   Say: "BM25 with Lucene's IDF, as an inverted index split into per-document segments with global statistics. Here's the small-corpus bug it fixes, and the test that shows segments score exactly like one big index. 1 ms at 100k chunks; adding a document touches only its segment."
5. **`app/services/document_service.py`: `submit`, `_process` and `sync_index`, plus `app/services/priority.py`.**
   Say: "Register and return 202, then one worker embeds in batches with progress and cancellation checkpoints, and yields to live queries through the priority gate. Vectors are computed before writing, with a compensating rollback if FAISS fails, and a self-heal on startup that makes the index disposable on ephemeral disks."
6. **`app/api/routes/query.py`: `query_stream`.**
   Say: "The same pipeline in a worker thread, an event queue, and a generator that yields SSE frames. Validation errors are plain 422s before the stream; pipeline errors become an `error` event. `X-Accel-Buffering: no` keeps proxies from buffering."
7. **`app/core/llm.py`: `GroqLLM.complete` / `stream` / `_create_with_retry`.**
   Say: "The provider is behind a protocol. I own the retry loop so I can honor `Retry-After` and *measure* throttle time, which separated our latency from free-tier rate limits in the eval. Streams are retried only before the first token."
8. **`tests/test_pipeline.py` and `tests/test_streaming.py`.**
   Say: "Scripted LLM by purpose. These tests prove the call sequences: PASS means exactly generate + judge; always-FAIL means exactly three judges at max_retries = 2; a worse retry keeps the original; and the SSE stream emits status → sources → tokens → reset → final in order."
