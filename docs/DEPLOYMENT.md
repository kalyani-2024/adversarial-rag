# Deployment

## What the service needs

| Resource | Why |
|---|---|
| ~650 MiB RAM for the API idle, ~800 MiB peak while ingesting (measured, see below) | PyTorch + MiniLM embedder + MiniLM cross-encoder in memory |
| 1–2 vCPU | embedding + reranking are CPU-bound (rerank of 20 candidates took ~1 s per query on the test laptop; expect more on 1–2 vCPU) |
| A writable `DATA_DIR` | SQLite (documents/chunks) + FAISS index |
| Outbound HTTPS to Groq | LLM calls; set `GROQ_API_KEY` as a runtime secret |

The image bakes the two small models in (no model-hub access needed at runtime, `HF_HUB_OFFLINE=1`) and runs as non-root uid 1000.

## Persistence: the important caveat

The API is **stateful on disk**: uploaded documents live in `DATA_DIR`. On a platform with an ephemeral filesystem, every restart or redeploy wipes them. What the code does about it:

- **FAISS is disposable.** On startup `sync_index()` rebuilds the vector index from SQLite if it is missing, stale, or was built with another embedding model.
- **Seed corpus.** `SEED_DIR` (the image ships `eval/corpus` at `/app/seed_docs`) is ingested at startup; content-hash de-duplication makes this idempotent. A demo therefore always comes up with the sample paper indexed, even on an ephemeral disk.
- **For durable user uploads** you need a persistent volume (Docker named volume, Render disk, HF persistent storage) or, at larger scale, Postgres + object storage (see ARCHITECTURE.md §5).

## Option A: Docker Compose (local or any VM) ✅ tested

```bash
cp .env.example .env          # set GROQ_API_KEY
docker compose up --build
# UI  http://localhost:8501
# API http://localhost:8000/docs
```

Two containers from one image (`APP_ROLE=api` / `APP_ROLE=ui`), a named volume `rag-data` for `/data`, the UI waits for the API health check.

## Option B: Hugging Face Spaces (free CPU tier), recommended for a public demo

Why: the free CPU Space has enough RAM for PyTorch plus both models, runs any Dockerfile, and exposes one port, which `APP_ROLE=all` serves (API on 127.0.0.1:8000, Streamlit on the public `$PORT`).

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

Caveats: free Spaces sleep when idle and the disk is ephemeral (uploads vanish on restart; the seed paper is re-indexed automatically). The Space is public: anyone can spend your Groq quota, so consider making it private or adding a shared-secret check.

**Status:** the `APP_ROLE=all` container was tested locally with `docker run` (see below). I have not deployed it to a Space from this repository.

## Option C: Render (documented, not tested)

`render.yaml` defines an API service with a 1 GB persistent disk and a UI service. **The free Render instance (512 MB, no disk) is not viable**: the API measured ~650 MiB idle and ~800 MiB while ingesting, so it would run out of memory, and uploads would be lost. Use a plan with at least 1–2 GB for the API. Set `GROQ_API_KEY` and the UI's `API_BASE_URL` in the dashboard.

## Option D: Split UI (Streamlit Community Cloud) + API elsewhere

The UI only needs `streamlit` and `requests` and talks to the API over HTTP. Deploy `app.py` on Streamlit Community Cloud with the secret `API_BASE_URL` pointing at a hosted API. Enable CORS on the API (`CORS_ORIGINS`) for that origin.

## Local verification log (2026-09-24, Docker Desktop 29.1.3, Windows 11, 8 CPU / 4 GB VM)

| Check | Result |
|---|---|
| `docker build` | OK in ~160 s (cached base); image reported as 3 GB by Docker Desktop: torch CPU 769 MB, pyarrow/pandas (Streamlit deps) ~230 MB, baked models 184 MB |
| Runs as non-root | `uid=1000(app)` |
| Secrets not in image | no `.env` in `/app`; no `GROQ*` env baked in |
| `docker compose up` | API healthy in ~30 s; seed paper indexed (29 chunks, index consistent); UI `/_stcore/health` 200 |
| Real query through the container | `answered`, judge PASS, 2 LLM calls, citations `drift_detection_paper.txt — chunk 2/5` |
| Memory (`docker stats`) | API ~654 MiB idle after a query; ~800 MiB peak while ingesting a 1.4 MB, 102-chunk PDF (7.4 s); UI ~60 MiB |
| Persistence | `docker compose restart api` → both documents (131 chunks) still indexed; seed step skipped the duplicate |
| Delete | `DELETE /documents/{id}` removed 102 chunks; index consistent |
| `APP_ROLE=all` single container | UI 200 on :7860, internal API healthy, seed indexed, query answered, container HEALTHCHECK `healthy`, ~490 MiB before first query |

Gotcha found while testing: in Git Bash on Windows, `docker run -e SEED_DIR=/app/seed_docs` is rewritten to a Windows path by MSYS. Use `MSYS_NO_PATHCONV=1 docker run ...` (not an issue with compose, PowerShell, or Linux/macOS shells).

Possible image-size optimization: build separate API and UI images (the UI needs only `streamlit` + `requests`; the API does not need Streamlit/pyarrow).
