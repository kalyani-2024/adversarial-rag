"""
FastAPI server for the Adversarial RAG pipeline.

Endpoints:
  POST /ingest  — Upload a .txt/.pdf/.docx file → extract text → chunk → embed → index
  POST /query   — Run the full LangGraph pipeline, return all state fields
"""

import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from core.vector_store import vector_store
from core.graph import rag_graph

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Adversarial RAG API",
    description="RAG pipeline with adversarial critique loop",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    chunks: list[str]
    initial_answer: str
    critique: str
    final_answer: str


class IngestResponse(BaseModel):
    message: str
    chunks_added: int


# ---------------------------------------------------------------------------
# File text extraction helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".docx")


def _extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from an uploaded file based on its extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "txt":
        return content.decode("utf-8", errors="ignore")

    if ext == "pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    if ext == "docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '.{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a .txt, .pdf, or .docx file into the vector store."""
    if not file.filename or not file.filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    text = _extract_text(file.filename, content)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded file.")

    num_chunks = vector_store.ingest(text)
    return IngestResponse(
        message=f"Successfully ingested '{file.filename}'.",
        chunks_added=num_chunks,
    )


@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Run the full adversarial RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if vector_store.index.ntotal == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload documents via /ingest first.",
        )

    # Execute the LangGraph pipeline
    result = rag_graph.invoke({"query": request.query})

    return QueryResponse(
        query=result.get("query", request.query),
        chunks=result.get("chunks", []),
        initial_answer=result.get("initial_answer", ""),
        critique=result.get("critique", ""),
        final_answer=result.get("final_answer", ""),
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "index_size": vector_store.index.ntotal,
        "total_chunks": len(vector_store.chunks),
    }
