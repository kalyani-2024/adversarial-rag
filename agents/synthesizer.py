"""
Synthesizer agent node.

Takes the initial answer and the adversary's critique, then calls Groq
to produce a final corrected and improved answer.
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
    "You are a senior research synthesizer. You are given:\n"
    "1. An initial AI-generated answer.\n"
    "2. A structured critique of that answer.\n"
    "3. The original source chunks.\n\n"
    "Your task is to produce a **final, corrected answer** that:\n"
    "- Addresses every weakness and contradiction raised in the critique.\n"
    "- Removes or fixes unsupported claims.\n"
    "- Remains grounded in the source chunks (cite with [1], [2], etc.).\n"
    "- Is clear, precise, and well-structured.\n\n"
    "Do NOT mention the critique process itself — just provide the best possible answer."
)


def synthesizer_node(state: dict) -> dict:
    """Synthesize a final corrected answer from initial answer + critique."""
    initial_answer: str = state.get("initial_answer", "")
    critique: str = state.get("critique", "")
    chunks: list[str] = state.get("chunks", [])

    chunks_text = "\n\n".join(
        f"[Chunk {i + 1}]: {chunk}" for i, chunk in enumerate(chunks)
    )

    user_message = (
        f"Source Chunks:\n{chunks_text}\n\n"
        f"Initial Answer:\n{initial_answer}\n\n"
        f"Critique:\n{critique}\n\n"
        "Produce the final, corrected answer."
    )

    response = _llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
    )

    return {"final_answer": response.content}
