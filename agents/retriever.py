"""
Retriever agent node.

Embeds the user query and performs FAISS similarity search to retrieve
the top-3 most relevant document chunks.
"""

from core.vector_store import vector_store


def retriever_node(state: dict) -> dict:
    """Retrieve top-3 chunks from FAISS for the user query."""
    query: str = state["query"]
    chunks = vector_store.search(query, top_k=3)
    return {"chunks": chunks}
