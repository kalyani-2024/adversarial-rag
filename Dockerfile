# syntax=docker/dockerfile:1
# One image, three roles chosen by APP_ROLE (api | ui | all); see scripts/entrypoint.sh.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DATA_DIR=/data \
    API_BASE_URL=http://localhost:8000

# Non-root user with uid 1000 (required by Hugging Face Spaces, good practice elsewhere).
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA (~2 GB) we never use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

USER app

# Bake the embedding + reranker weights into the image so containers start
# without network access to the model hub and the first query is not slow.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu'); CrossEncoder('${RERANKER_MODEL}', device='cpu')"
ENV HF_HUB_OFFLINE=1

COPY --chown=app:app app ./app
COPY --chown=app:app ui ./ui
COPY --chown=app:app api.py app.py ./
COPY --chown=app:app .streamlit ./.streamlit
COPY --chown=app:app scripts ./scripts
# Strip Windows line endings so a CRLF checkout can never break `sh` in the container.
RUN sed -i 's/\r$//' scripts/*.sh
COPY --chown=app:app eval/corpus ./seed_docs

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# APP_ROLE=api (default) | ui | all. Secrets (GROQ_API_KEY) are provided at runtime, never baked in.
CMD ["sh", "scripts/entrypoint.sh"]
