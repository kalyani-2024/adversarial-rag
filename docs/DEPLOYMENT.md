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

## Option S: Streamlit Community Cloud (free), recommended for a public demo

Streamlit Community Cloud runs one Streamlit process per app, so the app uses **embedded mode**: with `RAG_BACKEND=embedded` the UI calls the same services (`DocumentService`, `QueryService`, LangGraph pipeline, background ingestion, streaming events, logging) in-process instead of over HTTP (`ui/embedded.py`). The backend is created once per server process (`st.cache_resource`) and shared by all visitors. Local development and Docker are unchanged and still use the FastAPI server.

**Steps**

1. **Push the code to GitHub** (Community Cloud deploys from a GitHub repository; the branch can be `feature/reliability-platform` or `master` after merging). `.env` and `data/` are git-ignored, so no secrets or indexes are pushed. Note that the repository includes the sample paper in `eval/corpus/`.
2. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → *Deploy a public app from GitHub*:
   - Repository: `kalyani-2024/adversarial-rag`, branch: the one you pushed
   - Main file path: **`ui/streamlit_app.py`**
   - *Advanced settings* → Python **3.12**
3. In *Advanced settings → Secrets*, paste (TOML):
   ```toml
   RAG_BACKEND = "embedded"
   GROQ_API_KEY = "gsk_..."
   APP_PASSWORD = "choose-a-password"
   ```
   Any other setting from `.env.example` can be added the same way (e.g. `MAX_UPLOAD_MB = "50"`).
4. Deploy. The first build installs dependencies from **`ui/requirements.txt`** (Community Cloud reads the requirements file next to the entrypoint before the root one). That file pins CPU-only PyTorch; the default Linux wheel would pull ~2 GB of unused CUDA libraries. The first visit then downloads the two small models (~110 MB) and indexes the sample paper automatically (`eval/corpus`), which takes around a minute.
5. Open the app URL, enter the password, ask a question. Share the URL and password with interviewers.

**Caveats**

- **Memory.** Everything (Streamlit, PyTorch, both models, FAISS, the LLM pipeline) runs in one process. Measured locally, the API alone used ~620–710 MiB under Docker. Community Cloud's per-app memory limit is not under this repo's control, so check it. If the app is killed while indexing a big upload, use smaller files or lower `MAX_UPLOAD_MB`.
- **Ephemeral storage and sleep.** Apps sleep after inactivity, and the filesystem resets on restart or redeploy: uploads disappear, and an upload being indexed is lost. The sample paper is re-indexed on every start.
- **Shared state.** All visitors share one backend and one document collection (it's a demo, not multi-tenant). Anyone with the password can upload or delete documents.
- **CPU.** Indexing and reranking run on shared CPU; expect them to be slower than on a laptop.
- **Logs.** The structured JSON logs appear in the app's *Manage app → Logs* panel.
- **`ModuleNotFoundError: No module named 'torchvision'` in the logs.** Seen on a deployment that installed transformers 5.18.0 (and ran on the default Python 3.14 because step 2's Python 3.12 setting was skipped). The pipeline does not use torchvision. Streamlit's file watcher inspects every loaded module, and transformers 5.18.0 registers lazy image-processor modules that import torchvision when touched, so each page load logged about 100 of these tracebacks as warnings; the app itself kept working. Reproduced locally (103 warnings per page load on 5.18.0, none on 5.17.0), so `ui/requirements.txt` pins `transformers==5.17.0` and `sentence-transformers==6.1.0`. After pulling this change, use *Manage app → Reboot app* so the dependencies are reinstalled. The Python version can only be chosen when an app is created, so moving to 3.12 means deleting and redeploying the app.

**Status:** embedded mode was tested locally in the configuration Community Cloud uses: no API server, fresh data directory, password on. The sample paper was auto-indexed, and a streamed answer with citations came back from the in-process pipeline (plus unit tests in `tests/test_embedded.py`). The build from `ui/requirements.txt` on Community Cloud itself, and the deployment, have not been done yet; they have to be done from your GitHub and Streamlit accounts.

## Option B: Hugging Face Spaces (Docker)

> On the author's account (September 2026), Docker Spaces were offered only as a paid option and the free tier allowed only *Static* Spaces, which cannot run this Python backend. Check the current Hugging Face offering; with a paid CPU Space the steps below apply unchanged.

Why: the free CPU Space has enough RAM for PyTorch plus both models, runs any Dockerfile, and exposes one port, which `APP_ROLE=all` serves (API on 127.0.0.1:8000, Streamlit on the public `$PORT`). Because the UI calls the API inside the container, streaming and uploads don't pass through an extra proxy hop between them.

**Steps**

1. **Create an access token.** Go to huggingface.co → *Settings → Access Tokens* → new token with **Write** role.
2. **Create the Space.** huggingface.co/new-space → any name (e.g. `rag-reliability-lab`) → SDK **Docker** → template *Blank* → hardware **CPU basic (free)** → visibility *Public* (or *Private* while testing).
3. **Add secrets and variables** (Space → *Settings → Variables and secrets*):

   | Name | Kind | Value |
   |---|---|---|
   | `GROQ_API_KEY` | **Secret** | your Groq key |
   | `APP_PASSWORD` | **Secret** | a password for the demo (strongly recommended on a public Space) |
   | `APP_ROLE` | Variable | `all` |
   | `SEED_DIR` | Variable | `/app/seed_docs` |

4. **Build the Space folder** from this repo. It copies only what the container needs, writes the Space README with the YAML header Spaces require, normalizes line endings, and refuses to run if a `.env` or index file would be included:
   ```bash
   python scripts/prepare_hf_space.py          # -> dist/hf-space/
   ```
5. **Upload it** (log in once; the token is entered in your terminal, never stored in the repo):
   ```bash
   pip install -U huggingface_hub
   hf auth login                                # paste the Write token
   hf upload <your-username>/<space-name> dist/hf-space . --repo-type space
   ```
   (Alternatively `git clone` the Space repo, copy `dist/hf-space/*` into it, commit and push.)
6. **Wait for the build** (*Logs → Build*, typically several minutes: PyTorch CPU + models are baked into the image). When the container log shows `Uvicorn running on http://127.0.0.1:8000` and Streamlit's URL, open the Space, enter the password, and ask a question about the sample paper.
7. **Redeploy after changes:** rerun steps 4–5.

How the container behaves on Spaces:

- **One container, one public port.** `APP_ROLE=all` starts the API on 127.0.0.1:8000 (not reachable from outside) and Streamlit on port 7860 (`app_port`).
- **Password.** `APP_PASSWORD` gates the UI, and the UI is the only public entry point, so it also protects your Groq quota.
- **Uploads inside an iframe.** Spaces display the app in an iframe on another domain, where Streamlit's XSRF cookie is not sent back, so uploads fail with HTTP 403. The single-container mode therefore defaults `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false`, with the password gate as the access control.

Caveats:

- **Ephemeral disk and sleep.** Free Spaces sleep after a period of inactivity and the disk is ephemeral: uploads vanish on restart, and an upload being indexed when the Space sleeps is marked failed. The seed paper is re-indexed automatically on every start. Persistent storage is a paid Space add-on (mount it and set `DATA_DIR` to it).
- **Cold start.** The first visit after sleep waits for the container to boot and the models to warm up (roughly 30–60 s).
- **Upload size.** Upload limits in front of Streamlit on Spaces are outside this repo's control. Check that large uploads work there before relying on 100 MB.
- **CPU.** Indexing and reranking run on the Space's shared CPU; expect them to be slower than on a laptop.

**Status:** the Space folder produced by `prepare_hf_space.py` **builds** successfully with Docker. The password gate and XSRF setting were tested locally without Docker. The single-container mode itself was tested in run 1 of the verification log. The deployment to huggingface.co has not been done yet; it has to be done from your account.

## Option C: Render (documented, not tested)

`render.yaml` defines an API service with a 1 GB persistent disk and a UI service. **The free Render instance (512 MB, no disk) is not viable**: the API measured ~650–710 MiB, so it would run out of memory, and uploads would be lost. Use a plan with at least 1–2 GB for the API. Set `GROQ_API_KEY` and the UI's `API_BASE_URL` in the dashboard. Also check Render's request-size and response-buffering behaviour for 100 MB uploads and SSE.

## Option D: Streamlit Community Cloud UI + API hosted elsewhere

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
