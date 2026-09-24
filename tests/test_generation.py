from app.generation.generator import extract_citations, format_context, generate_answer, is_abstention
from app.generation.rewrite import decide_rewrite, rewrite_query
from app.schemas.query import INSUFFICIENT_EVIDENCE_ANSWER, ChatTurn
from app.schemas.reliability import Critique
from app.schemas.retrieval import RetrievedChunk
from tests.fakes import FakeLLM


def _chunks():
    return [
        RetrievedChunk(chunk_id="a:0", document_id="a", document_name="paper.pdf", chunk_index=0, page=4, text="AUC is 0.959."),
        RetrievedChunk(chunk_id="b:8", document_id="b", document_name="arch.md", chunk_index=8, text="Uses FAISS."),
    ]


# --- citations ----------------------------------------------------------------------
def test_extract_citations_valid_invalid_and_groups():
    text = "AUC is 0.959 [1]. It uses FAISS [2, 7]. Made up [9]."
    cleaned, valid, invalid = extract_citations(text, num_sources=2)
    assert valid == [1, 2]
    assert invalid == [7, 9]
    assert cleaned == "AUC is 0.959 [1]. It uses FAISS [2]. Made up."


def test_format_context_labels():
    ctx = format_context(_chunks())
    assert "[1] (paper.pdf — page 4)" in ctx
    assert "[2] (arch.md — chunk 8)" in ctx


def test_generate_answer_builds_citations():
    llm = FakeLLM({"generate": ["The AUC is 0.959 [1]. It uses FAISS [2]."]})
    out = generate_answer(llm, "q?", _chunks(), temperature=0.0)
    assert not out.abstained
    assert [c.index for c in out.citations] == [1, 2]
    assert out.citations[0].label == "paper.pdf — page 4"
    assert out.citations[1].label == "arch.md — chunk 8"
    assert out.citations[0].text == "AUC is 0.959."


def test_generate_detects_abstention():
    llm = FakeLLM({"generate": [INSUFFICIENT_EVIDENCE_ANSWER]})
    out = generate_answer(llm, "q?", _chunks(), temperature=0.0)
    assert out.abstained and out.citations == [] and out.answer == INSUFFICIENT_EVIDENCE_ANSWER


def test_is_abstention_keeps_partial_answers():
    assert is_abstention("  " + INSUFFICIENT_EVIDENCE_ANSWER.upper())
    long_partial = "The AUC is 0.959 [1]. " * 10 + INSUFFICIENT_EVIDENCE_ANSWER
    assert not is_abstention(long_partial)


def test_regenerate_includes_critique_and_previous_answer():
    llm = FakeLLM({"regenerate": ["Fixed [1]."]})
    critique = Critique(unsupported_claims=["It was trained on ImageNet"], instructions="Remove the ImageNet claim.")
    out = generate_answer(llm, "q?", _chunks(), temperature=0.0, critique=critique, previous_answer="Old answer.")
    prompt = llm.calls[0][1][1]["content"]
    assert "Old answer." in prompt and "ImageNet" in prompt and "Remove the ImageNet claim." in prompt
    assert out.answer == "Fixed [1]."


# --- query rewrite --------------------------------------------------------------------
def test_rewrite_gate():
    assert not decide_rewrite("What ROC-AUC does the Transformer reach?", []).should_rewrite
    assert decide_rewrite("hey can you tell me the AUC", []).should_rewrite
    history = [ChatTurn(role="user", content="Tell me about the Transformer model")]
    assert decide_rewrite("what AUC does it get?", history).should_rewrite
    # pronoun without any history: nothing to resolve against
    assert not decide_rewrite("what AUC does it get?", []).should_rewrite


def test_rewrite_uses_llm_output():
    llm = FakeLLM({"rewrite": ['{"query": "Transformer ROC-AUC"}']})
    q, resp = rewrite_query(llm, "what AUC does it get?", [], model="m")
    assert q == "Transformer ROC-AUC" and resp is not None


def test_rewrite_falls_back_on_bad_output():
    llm = FakeLLM({"rewrite": ["not json"]})
    q, _ = rewrite_query(llm, "what AUC does it get?", [], model="m")
    assert q == "what AUC does it get?"


def test_rewrite_rejects_runaway_output():
    llm = FakeLLM({"rewrite": ['{"query": "' + "x" * 1000 + '"}']})
    q, _ = rewrite_query(llm, "short?", [], model="m")
    assert q == "short?"


def test_extract_citations_normalizes_model_native_markers():
    text = "AUC is 0.959【1】【2†L1-L3】 and [3†source]."
    cleaned, valid, invalid = extract_citations(text, num_sources=2)
    assert cleaned == "AUC is 0.959[1][2] and."
    assert valid == [1, 2] and invalid == [3]
