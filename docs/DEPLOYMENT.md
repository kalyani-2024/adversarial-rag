# Deployment

## What the service needs

| Resource | Why |
|---|---|
| ~650–710 MiB RAM for the API (measured, see below); UI ~50–60 MiB | PyTorch + MiniLM embedder + MiniLM cross-encoder in memory |
| 1–2+ vCPU | embedding + reranking are CPU-bound: rerank ~0.6 s per query and embedding ~21 chunks/s on the test laptop; expect slower on 1–2 vCPU. Indexing speed decides how long large uploads take. |
| A writable `DATA_DIR` | SQLite (documents, status, chunks) + FAISS index |
| Outbound HTTPS to Groq | LLM calls; set `GROQ_API_KEY` as a runtime secret |
| Request body limit ≥ 100 MB | uploads up to `MAX_UPLOAD_MB` (default 100) must pass any reverse proxy / platform limit |
| Unbuffered responses | `POST /query/stream` uses Server-Sent Events; proxies must not buffer `text/event-stream` (the API sends `X-Accel-Buffering: no`) |

The image bakes the two small models in (no model-hub access needed at runtime, `HF_HUB_OFFLINE=1`) and runs as non-root uid 1000.

## Persistence: the important caveat

The API is **stateful on disk**: uploaded documents live in `DATA_DIR`. On a platform with an ephemeral filesystem, every restart or redeploy wipes them. What the code does about it:

- **FAISS is disposable.** On startup `sync_index()` rebuilds the vector index from SQLite if it is missing, stale, or was built with another embedding model.
- **Interrupted uploads are visible.** Documents still indexing when the process stops are marked `failed` ("interrupted by a server restart") on the next start; re-uploading the file retries it.
- **Schema migrations are automatic.** Older databases get the `status`/`error` columns on startup (tested against a volume created before they existed).
- **Seed corpus.** `SEED_DIR` (the image ships `eval/corpus` at `/app/seed_docs`) is ingested synchronously at startup; content-hash de-duplication makes this idempotent. A demo therefore always comes up with the sample paper indexed, even on an ephemeral disk.
- **For durable user uploads** you need a persistent volume (Docker named volume, Render disk, HF persistent storage) or, at larger scale, Postgres + object storage (see ARCHITECTURE.md §5).

Background indexing runs inside the API process, so a platform that restarts or sleeps the container mid-upload loses that upload (it is marked failed, not silently lost).

## Option A: Docker Compose (local or any VM) ✅ tested

```bash
cp .env.example .env          # set GROQ_API_KEY
docker compose up --build
# UI  http://localhost:8501
# API http://localhost:8000/docs  (use 127.0.0.1 on Windows to avoid a ~2 s localhost/IPv6 delay)
```

Two containers from one image (`APP_ROLE=api` / `APP_ROLE=ui`), a named volume `rag-data` for `/data`, the UI waits for the API health check. Streamlit's upload limit is set to 100 MB in `.streamlit/config.toml`.

## Option B: Hugging Face Spaces (free CPU tier), recommended for a public demo

Why: the free CPU Space has enough RAM for PyTorch plus both models, runs any Dockerfile, and exposes one port, which `APP_ROLE=all` serves (API on 127.0.0.1:8000, Streamlit on the public `$PORT`). Because the UI calls the API inside the container, streaming and uploads don't pass through an extra proxy hop between them.

1. Create a Space → SDK **Docker** → hardware **CPU basic**.
2. Push this repository to the Space's git remote. The Space's `README.md` needs this front matter at the top (keep the rest of the README below it):
   ```yaml
   ---
   title: RAG Reliability Lab
   sdk: docker
   app_port: 7860
   ---
   ```
3. In *Settings → Variables and secrets* add:
   - secret `GROQ_API_KEY`
   - variable `APP_ROLE=all`
   - variable `SEED_DIR=/app/seed_docs`
4. The Space builds the Dockerfile and starts `scripts/entrypoint.sh`.

Caveats:

- **Ephemeral disk and sleep.** Free Spaces sleep when idle and the disk is ephemeral: uploads vanish on restart, and an upload being indexed when the Space sleeps is marked failed. The seed paper is re-indexed automatically.
- **Public by default.** Anyone can spend your Groq quota, so consider making the Space private or adding a shared-secret check.
- **Upload size.** Upload limits in front of Streamlit on Spaces are outside this repo's control. Check that 100 MB uploads work there before relying on them.

**Status:** the `APP_ROLE=all` container was tested locally with `docker run` (see below). I have not deployed it to a Space from this repository.

## Option C: Render (documented, not tested)

`render.yaml` defines an API service with a 1 GB persistent disk and a UI service. **The free Render instance (512 MB, no disk) is not viable**: the API measured ~650–710 MiB, so it would run out of memory, and uploads would be lost. Use a plan with at least 1–2 GB for the API. Set `GROQ_API_KEY` and the UI's `API_BASE_URL` in the dashboard. Also check Render's request-size and response-buffering behaviour for 100 MB uploads and SSE.

## Option D: Split UI (Streamlit Community Cloud) + API elsewhere

The UI only needs `streamlit` and `requests` and talks to the API over HTTP (JSON + SSE). Deploy `app.py` on Streamlit Community Cloud with the secret `API_BASE_URL` pointing at a hosted API. Enable CORS on the API (`CORS_ORIGINS='["https://your-app.streamlit.app"]'`). Uploads then travel UI → API over the internet, so both sides need the 100 MB limit.

## Configuration that matters in production

| Variable | Default | Why you'd change it |
|---|---|---|
| `MAX_UPLOAD_MB` | 100 | match your proxy/platform limit |
| `RERANK_CANDIDATES` / `RERANKER_MAX_LENGTH` | 10 / 256 | CPU budget per query (measured: same hit@k as 20 / 512 on the golden set) |
| `RERANKER_ENABLED` | true | saves ~0.6 s per query at the cost of a weaker abstention signal |
| `MAX_RETRIES` | 2 | caps LLM calls per question (worst case `1 + 3 × MAX_RETRIES`) |
| `LLM_MAX_RETRIES` / `LLM_MAX_WAIT_S` | 3 / 45 | how long to ride out provider rate limits before failing |
| `LOG_JSON` / `LOG_CONTENT` | true / true | set `LOG_CONTENT=false` where logs must not contain user text |
| `CORS_ORIGINS` | `["*"]` | restrict to your UI origin |
| `SEED_DIR` | unset | demo corpus to index at startup |
| `WARMUP_MODELS` | true | load models at startup so the first query isn't slow |

## Local verification log

### Run 2 (2026-09-24, after streaming + background ingestion): Docker Desktop 29.1.3, Windows 11, 8 CPU / 4 GB VM

| Check | Result |
|---|---|
| `docker build` | OK with the current code |
| `docker compose up` on the **existing** volume from run 1 | API healthy; old database migrated (new `status` column); seed document reported `ready`, 29 chunks, index consistent |
| Background upload (1.4 MB PDF) | `202 processing` in 0.35 s → stages `parsing` → `embedding` → `ready`, 102 chunks in 9.1 s |
| Streaming query through the container | `POST /query/stream`: events `status → sources → token (123) → final`; first token 3.3 s, total 3.9 s; verdict PASS; citations from both documents (`… — page 5`, `… — chunk 13`) |
| Delete | test document removed |
| Memory (`docker stats`) | API peak ~709 MiB during upload + query, ~624 MiB after; UI ~50 MiB |
| UI | `/_stcore/health` 200 |

### Run 1 (2026-09-24, first containerized version)

| Check | Result |
|---|---|
| `docker build` | OK in ~160 s (cached base); image reported as 3 GB by Docker Desktop: torch CPU 769 MB, pyarrow/pandas (Streamlit deps) ~230 MB, baked models 184 MB |
| Runs as non-root | `uid=1000(app)` |
| Secrets not in image | no `.env` in `/app`; no `GROQ*` env baked in |
| `docker compose up` | API healthy in ~30 s; seed paper indexed (29 chunks, index consistent); UI `/_stcore/health` 200 |
| Real query through the container | `answered`, judge PASS, 2 LLM calls, citations `drift_detection_paper.txt — chunk 2/5` |
| Memory (`docker stats`) | API ~654 MiB idle after a query; ~800 MiB peak while ingesting a 1.4 MB, 102-chunk PDF (synchronous ingestion at the time); UI ~60 MiB |
| Persistence | `docker compose restart api` → both documents (131 chunks) still indexed; seed step skipped the duplicate |
| Delete | `DELETE /documents/{id}` removed 102 chunks; index consistent |
| `APP_ROLE=all` single container | UI 200 on :7860, internal API healthy, seed indexed, query answered, container HEALTHCHECK `healthy`, ~490 MiB before first query |

Gotcha found while testing: in Git Bash on Windows, `docker run -e SEED_DIR=/app/seed_docs` is rewritten to a Windows path by MSYS. Use `MSYS_NO_PATHCONV=1 docker run ...` (not an issue with compose, PowerShell, or Linux/macOS shells).

Not re-tested in run 2: the single-container `APP_ROLE=all` mode (unchanged scripts; the UI inside it now uses streaming).

Possible image-size optimization: build separate API and UI images (the UI needs only `streamlit` + `requests`; the API does not need Streamlit/pyarrow).
