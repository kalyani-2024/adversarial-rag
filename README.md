# Adversarial RAG Pipeline

A **Retrieval-Augmented Generation** system with an adversarial critique loop.  
Built with **LangGraph**, **FAISS**, **Sentence-Transformers**, **Groq**, **FastAPI**, and **Streamlit**.

---

## Architecture

```
Query → Retriever → Generator → Adversary → Synthesizer → Final Answer
```

| Node | Role |
|------|------|
| **Retriever** | Embeds the query with `all-MiniLM-L6-v2`, searches FAISS, returns top-3 chunks |
| **Generator** | Sends query + chunks to Groq (`llama-3.3-70b-versatile`) for a grounded answer |
| **Adversary** | Critiques the answer: weaknesses, unsupported claims, contradictions |
| **Synthesizer** | Merges the initial answer with the critique to produce a corrected final answer |

## Project Structure

```
adversarial-rag/
├── agents/
│   ├── retriever.py      # FAISS retrieval node
│   ├── generator.py      # Grounded answer generation
│   ├── adversary.py      # Adversarial critique
│   └── synthesizer.py    # Final corrected answer
├── core/
│   ├── vector_store.py   # FAISS index management
│   └── graph.py          # LangGraph pipeline definition
├── api.py                # FastAPI server
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Start the FastAPI server

```bash
uvicorn api:app --reload --port 8000
```

### 4. Start the Streamlit UI

```bash
streamlit run app.py
```

## API Endpoints

### `POST /ingest`

Upload a `.txt` file to be chunked and indexed in FAISS.

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@document.txt"
```

### `POST /query`

Run the full adversarial RAG pipeline.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?"}'
```

**Response:**

```json
{
  "query": "...",
  "chunks": ["...", "...", "..."],
  "initial_answer": "...",
  "critique": "...",
  "final_answer": "..."
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Your Groq API key ([console.groq.com](https://console.groq.com)) |

## License

MIT
