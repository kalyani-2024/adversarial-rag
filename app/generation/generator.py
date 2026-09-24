"""Grounded answer generation with numbered citations.

Chunks are shown to the model as numbered sources `[1]..[k]`. After
generation we parse the markers, drop any that point at non-existent sources
(a common hallucination), and map the rest back to chunk metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.llm import LLMClient, LLMResponse, TokenCallback
from app.generation.prompts import GENERATOR_SYSTEM, GENERATOR_USER, REGENERATE_USER
from app.schemas.query import INSUFFICIENT_EVIDENCE_ANSWER, Citation
from app.schemas.reliability import Critique
from app.schemas.retrieval import RetrievedChunk

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
# Model-native citation styles, e.g. gpt-oss emits "【4†L1-L3】"; normalized to "[4]".
_NATIVE_CITATION = re.compile(r"[【\[](\d+)†[^】\]]*[】\]]|【(\d+)】")
_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([.,;:])")


def source_label(chunk: RetrievedChunk) -> str:
    where = f"page {chunk.page}" if chunk.page is not None else f"chunk {chunk.chunk_index}"
    return f"{chunk.document_name} — {where}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] ({source_label(c)})\n{c.text}" for i, c in enumerate(chunks, start=1))


def is_abstention(answer: str) -> bool:
    """True when the answer is (essentially only) the fixed abstention sentence.

    A long partial answer that merely *mentions* missing evidence is kept.
    """
    normalized = _WS.sub(" ", answer).strip().lower()
    sentence = INSUFFICIENT_EVIDENCE_ANSWER.lower().rstrip(".")
    return sentence in normalized and len(normalized) <= len(sentence) + 60


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    invalid_citations: list[int]
    abstained: bool
    response: LLMResponse


def extract_citations(answer: str, num_sources: int) -> tuple[str, list[int], list[int]]:
    """Return `(cleaned_answer, valid_indices_in_order, invalid_indices)`.

    Out-of-range markers are removed from the text so users never see a
    citation that points nowhere.
    """
    answer = _NATIVE_CITATION.sub(lambda m: f"[{m.group(1) or m.group(2)}]", answer)
    valid: list[int] = []
    invalid: list[int] = []

    def _replace(match: re.Match) -> str:
        nums = [int(n) for n in re.split(r"\s*[,;]\s*", match.group(1))]
        keep = []
        for n in nums:
            if 1 <= n <= num_sources:
                keep.append(n)
                if n not in valid:
                    valid.append(n)
            elif n not in invalid:
                invalid.append(n)
        return "".join(f"[{n}]" for n in keep)

    cleaned = _CITATION_GROUP.sub(_replace, answer)
    if invalid:  # tidy "claim [9]." -> "claim ." -> "claim."
        cleaned = _SPACE_BEFORE_PUNCT.sub(lambda m: m.group(1), cleaned)
    return cleaned.strip(), valid, invalid


def build_citations(indices: list[int], chunks: list[RetrievedChunk]) -> list[Citation]:
    out = []
    for n in sorted(indices):
        c = chunks[n - 1]
        out.append(
            Citation(
                index=n,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                page=c.page,
                chunk_index=c.chunk_index,
                label=source_label(c),
                text=c.text,
            )
        )
    return out


def _critique_text(critique: Critique) -> str:
    sections = [
        ("Unsupported claims", critique.unsupported_claims),
        ("Contradictions", critique.contradictions),
        ("Missing evidence", critique.missing_evidence),
        ("Weak reasoning", critique.weak_reasoning),
        ("Irrelevant content", critique.irrelevant_content),
    ]
    lines = [f"{title}:\n" + "\n".join(f"- {item}" for item in items) for title, items in sections if items]
    if critique.instructions:
        lines.append(f"Reviewer instructions: {critique.instructions}")
    return "\n\n".join(lines) or "The answer was judged unreliable; make every claim traceable to a source."


def generate_answer(
    llm: LLMClient,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    temperature: float,
    critique: Critique | None = None,
    previous_answer: str | None = None,
    on_token: TokenCallback | None = None,
) -> GeneratedAnswer:
    """Initial generation, or critique-guided regeneration when `critique` is given.

    With `on_token`, raw text deltas are streamed as they arrive; the returned
    answer is still post-processed (citations validated, abstention detected).
    """
    context = format_context(chunks)
    if critique is not None and previous_answer is not None:
        user = REGENERATE_USER.format(
            context=context,
            question=question,
            previous_answer=previous_answer,
            critique=_critique_text(critique),
            insufficient=INSUFFICIENT_EVIDENCE_ANSWER,
        )
        purpose = "regenerate"
    else:
        user = GENERATOR_USER.format(context=context, question=question)
        purpose = "generate"

    messages = [{"role": "system", "content": GENERATOR_SYSTEM}, {"role": "user", "content": user}]
    if on_token is not None and hasattr(llm, "stream"):
        resp = llm.stream(messages, purpose=purpose, temperature=temperature, on_token=on_token)
    else:
        resp = llm.complete(messages, purpose=purpose, temperature=temperature)
    raw = resp.text.strip()
    if not raw or is_abstention(raw):
        return GeneratedAnswer(INSUFFICIENT_EVIDENCE_ANSWER, [], [], True, resp)

    cleaned, valid, invalid = extract_citations(raw, len(chunks))
    return GeneratedAnswer(cleaned, build_citations(valid, chunks), invalid, False, resp)
