"""
LangGraph pipeline definition.

Builds a sequential 4-node graph:
  Retriever → Generator → Adversary → Synthesizer

State is a TypedDict with: query, chunks, initial_answer, critique, final_answer.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents.retriever import retriever_node
from agents.generator import generator_node
from agents.adversary import adversary_node
from agents.synthesizer import synthesizer_node


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    query: str
    chunks: list[str]
    initial_answer: str
    critique: str
    final_answer: str


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the adversarial RAG graph."""
    graph = StateGraph(PipelineState)

    # Add the four sequential nodes
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("adversary", adversary_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Wire them sequentially
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "adversary")
    graph.add_edge("adversary", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# Pre-compiled graph singleton
rag_graph = build_graph()
