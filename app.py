"""
Streamlit frontend for the Adversarial RAG pipeline.

Features:
  - Sidebar: document upload (.txt)
  - Main: query input
  - 3-column output: Initial Answer | Critique | Final Answer
  - Expander showing retrieved chunks
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Adversarial RAG",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished, premium look
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #8b8fa3;
        font-size: 1rem;
        margin-top: 0;
    }

    /* Cards */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        min-height: 200px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .card-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-title.initial { color: #667eea; }
    .card-title.critique { color: #f093fb; }
    .card-title.final    { color: #4facfe; }
    .card-body {
        font-size: 0.92rem;
        line-height: 1.7;
        color: #c9cde0;
    }

    /* Chunks expander */
    .chunk-badge {
        display: inline-block;
        background: rgba(102, 126, 234, 0.12);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #b0b8d4;
        width: 100%;
    }
    .chunk-label {
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16162a 0%, #1a1a2e 100%);
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    /* Status badges */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-success {
        background: rgba(76, 175, 80, 0.15);
        color: #66bb6a;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .status-error {
        background: rgba(244, 67, 54, 0.15);
        color: #ef5350;
        border: 1px solid rgba(244, 67, 54, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — document upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-title">📄 Document Upload</div>', unsafe_allow_html=True)
    st.caption("Upload `.txt`, `.pdf`, or `.docx` files to build the knowledge base.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if st.button("⬆️ Ingest Document", use_container_width=True):
            with st.spinner("Chunking & indexing…"):
                try:
                    mime_map = {
                        "txt": "text/plain",
                        "pdf": "application/pdf",
                        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
                    mime = mime_map.get(ext, "application/octet-stream")
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime)}
                    resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.markdown(
                            f'<span class="status-pill status-success">✓ {data["chunks_added"]} chunks added</span>',
                            unsafe_allow_html=True,
                        )
                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.markdown(
                            f'<span class="status-pill status-error">✗ {detail}</span>',
                            unsafe_allow_html=True,
                        )
                except requests.ConnectionError:
                    st.error("Cannot reach API. Is the FastAPI server running on port 8000?")

    st.divider()

    # Health check
    st.markdown('<div class="sidebar-title">📊 Index Status</div>', unsafe_allow_html=True)
    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        col1, col2 = st.columns(2)
        col1.metric("Vectors", health.get("index_size", 0))
        col2.metric("Chunks", health.get("total_chunks", 0))
    except Exception:
        st.caption("⚠️ API unavailable")

# ---------------------------------------------------------------------------
# Main — header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>⚔️ Adversarial RAG</h1>
        <p>Retrieve · Generate · Critique · Synthesize</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What are the main contributions of the paper?",
    label_visibility="collapsed",
)

run_button = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button and query.strip():
    with st.spinner("Running adversarial RAG pipeline…"):
        try:
            resp = requests.post(
                f"{API_BASE}/query",
                json={"query": query},
                timeout=180,
            )

            if resp.status_code != 200:
                detail = resp.json().get("detail", resp.text)
                st.error(f"API error: {detail}")
            else:
                data = resp.json()

                # --- Retrieved chunks expander ---
                with st.expander("📚 Retrieved Chunks", expanded=False):
                    chunks = data.get("chunks", [])
                    if chunks:
                        for i, chunk in enumerate(chunks):
                            st.markdown(
                                f'<div class="chunk-badge">'
                                f'<div class="chunk-label">Chunk {i + 1}</div>'
                                f"{chunk}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No chunks retrieved.")

                # --- 3-column output ---
                col_initial, col_critique, col_final = st.columns(3)

                with col_initial:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="card-title initial">💡 Initial Answer</div>'
                        f'<div class="card-body">{data.get("initial_answer", "—")}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with col_critique:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="card-title critique">🔍 Adversarial Critique</div>'
                        f'<div class="card-body">{data.get("critique", "—")}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with col_final:
                    st.markdown(
                        f'<div class="result-card">'
                        f'<div class="card-title final">✅ Final Answer</div>'
                        f'<div class="card-body">{data.get("final_answer", "—")}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        except requests.ConnectionError:
            st.error(
                "🔌 Cannot connect to the API server. "
                "Make sure to run `uvicorn api:app --reload --port 8000` first."
            )

elif run_button and not query.strip():
    st.warning("Please enter a query first.")
