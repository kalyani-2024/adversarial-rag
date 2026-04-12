"""
Generator agent node.

Takes the user query and retrieved chunks, then calls Groq
(llama-3.3-70b-versatile) to produce a grounded answer that cites chunks.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the provided context chunks. Cite the chunk number (e.g. [1], [2]) for every "
    "claim you make. If the context does not contain enough information, say so."
)


def generator_node(state: dict) -> dict:
    """Generate a grounded answer from query + retrieved chunks."""
    query: str = state["query"]
    chunks: list[str] = state.get("chunks", [])

    if not chunks:
        return {"initial_answer": "No relevant documents found to answer the query."}

    # Format chunks for the prompt
    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
    )

    user_message = (
        f"Context:\n{chunks_text}\n\n"
        f"Question: {query}\n\n"
        "Provide a thorough, well-cited answer."
    )

    response = _llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    return {"initial_answer": response.content}
