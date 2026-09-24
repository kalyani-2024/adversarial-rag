"""Assemble a Hugging Face Space (Docker SDK) from this repository.

    python scripts/prepare_hf_space.py                 # writes dist/hf-space/
    hf upload <user>/<space> dist/hf-space . --repo-type space

Only what the container needs is copied: the Dockerfile builds the same image
as `docker compose`, and the Space runs it with APP_ROLE=all (API on
127.0.0.1:8000, Streamlit public on port 7860). The Space's README carries the
YAML header Spaces require; the project README is not modified.
Secrets are never copied: set GROQ_API_KEY (and APP_PASSWORD) in the Space settings.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "hf-space"

INCLUDE = [
    "Dockerfile", ".dockerignore", "requirements.txt", "api.py", "app.py",
    "app", "ui", "scripts", ".streamlit", "eval/corpus",
]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
CRLF, LF = bytes([13, 10]), bytes([10])

SPACE_README = """---
title: RAG Reliability Lab
emoji: 🧪
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid RAG with a conditional adversarial reliability loop
---

# RAG Reliability Lab

Ask questions about uploaded documents. Answers stream in with inline citations
(hover a number to see the source passage). Each answer is checked by an
independent judge model, and only when it fails is it critiqued and regenerated.

Retrieval: dense (FAISS, MiniLM) + BM25, Reciprocal Rank Fusion, cross-encoder rerank,
and an evidence gate that refuses to answer when nothing relevant was found.

**Space configuration** (Settings → Variables and secrets):

| Name | Type | Value |
|---|---|---|
| `GROQ_API_KEY` | secret | your Groq key |
| `APP_PASSWORD` | secret | password for the demo (recommended) |
| `APP_ROLE` | variable | `all` |
| `SEED_DIR` | variable | `/app/seed_docs` (indexes the sample paper at startup) |

Notes: the free CPU Space sleeps when idle and its disk is ephemeral, so uploaded
documents disappear on restart (the sample paper is re-indexed automatically).
"""


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for rel in INCLUDE:
        src = ROOT / rel
        dst = OUT / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (OUT / "README.md").write_text(SPACE_README, encoding="utf-8")

    # Normalize text files to LF: a Windows checkout may have CRLF, which breaks shell scripts on Linux.
    for path in OUT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".txt", ".toml", ".md", ""}:
            data = path.read_bytes()
            if CRLF in data:
                path.write_bytes(data.replace(CRLF, LF))

    leaked = [p for p in OUT.rglob("*") if p.name == ".env" or p.suffix in {".sqlite3", ".faiss"}]
    if leaked:
        raise SystemExit(f"Refusing: secrets or data found in the Space folder: {leaked}")
    files = [p for p in OUT.rglob("*") if p.is_file()]
    size_kb = sum(p.stat().st_size for p in files) / 1024
    print(f"Space folder ready: {OUT}  ({len(files)} files, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
