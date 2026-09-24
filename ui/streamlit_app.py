"""RAG Reliability Lab — Streamlit front end.

Talks to the FastAPI backend over HTTP only, so it can be deployed separately
(e.g. Streamlit Community Cloud) by setting API_BASE_URL.

Safety note: model output is rendered with st.markdown/st.text (HTML disabled).
The only raw HTML we emit is built from our own span names and numbers,
escaped with html.escape.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import APIError, RagClient  # noqa: E402

st.set_page_config(page_title="RAG Reliability Lab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")


def _api_base_url() -> str:
    if os.getenv("API_BASE_URL"):
        return os.environ["API_BASE_URL"]
    try:
        return st.secrets["API_BASE_URL"]
    except Exception:
        return "http://localhost:8000"


client = RagClient(_api_base_url())

STAGE_LABELS = {
    "query_rewrite": "Query Rewrite",
    "hybrid_retrieval": "Hybrid Retrieval",
    "reranking": "Reranking",
    "evidence_gate": "Evidence Gate",
    "generation": "Generation",
    "judge": "Judge",
    "critic": "Critic",
    "regeneration": "Regeneration",
}
MIME = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }
    .lab-title { font-size: 1.9rem; font-weight: 750; letter-spacing: -0.02em; margin-bottom: 0; }
    .lab-sub { color: #6b7280; margin-top: 0.1rem; margin-bottom: 1.2rem; }
    .trace { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.86rem; }
    .trace-row { display: grid; grid-template-columns: 1.6rem 10.5rem 4.8rem 1fr; align-items: center; gap: 0.5rem; padding: 2px 0; }
    .trace-bar { height: 8px; border-radius: 4px; background: #6366f1; opacity: 0.85; }
    .trace-bar.fail { background: #ef4444; } .trace-bar.skip { background: #d1d5db; }
    .trace-bar.loop { background: #f59e0b; }
    .muted { color: #9ca3af; }
    </style>
    """,
    unsafe_allow_html=True,
)


def md_safe(text: str) -> str:
    """Stop '$' in answers from being rendered as LaTeX."""
    return text.replace("$", r"\$")


# --------------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 RAG Reliability Lab")
    try:
        health = client.health()
        ok = health["llm_configured"] and health["index_consistent"]
        st.caption(
            f"{'🟢' if ok else '🟠'} API online · {health['documents']} docs · {health['chunks']} chunks  \n"
            f"LLM `{health['llm_model']}` · judge `{health['judge_model']}`"
        )
        if not health["llm_configured"]:
            st.warning("GROQ_API_KEY is not configured on the API.")
    except APIError as e:
        health = None
        st.error(e.message)

    st.markdown("#### Documents")
    uploads = st.file_uploader("Upload", type=["pdf", "txt", "md", "markdown", "docx"], accept_multiple_files=True,
                               label_visibility="collapsed")
    if uploads and st.button("Index documents", use_container_width=True, type="primary"):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    res = client.upload(f.name, f.getvalue(), MIME.get(f.name.rsplit(".", 1)[-1].lower(), "application/octet-stream"))
                    st.success(res["message"])
                except APIError as e:
                    (st.info if e.code == "duplicate_document" else st.error)(f"{f.name}: {e.message}")

    if health is not None:
        try:
            docs = client.list_documents()["documents"]
        except APIError:
            docs = []
        if not docs:
            st.caption("No documents indexed yet.")
        for d in docs:
            c1, c2 = st.columns([5, 1])
            pages = f" · {d['num_pages']} pp" if d.get("num_pages") else ""
            c1.markdown(f"**{html.escape(d['filename'])}**  \n<span class='muted'>{d['num_chunks']} chunks{pages}</span>",
                        unsafe_allow_html=True)
            if c2.button("✕", key=f"del-{d['id']}", help="Delete document and its chunks"):
                try:
                    client.delete(d["id"])
                    st.rerun()
                except APIError as e:
                    st.error(e.message)

    st.markdown("#### Pipeline")
    mode = st.radio("Mode", ["adversarial", "baseline"], horizontal=True,
                    help="Baseline = single-pass RAG (no judge/critic). Adversarial = judge + conditional critique loop.")
    compare = st.toggle("Compare with baseline", value=False, help="Also run single-pass RAG and show both answers side by side.")
    top_k = st.slider("Top-k context chunks", 1, 10, 5)
    use_reranker = st.toggle("Cross-encoder reranker", value=True)
    use_rewrite = st.toggle("Query rewrite (when needed)", value=True)

    st.markdown("#### Reliability")
    max_retries = st.slider("Max retries", 0, 3, 2, disabled=mode == "baseline")
    faith_t = st.slider("Faithfulness threshold", 0.0, 1.0, 0.80, 0.05, disabled=mode == "baseline")
    rel_t = st.slider("Relevance threshold", 0.0, 1.0, 0.70, 0.05, disabled=mode == "baseline")
    comp_t = st.slider("Completeness threshold", 0.0, 1.0, 0.60, 0.05, disabled=mode == "baseline")
    strict_claims = st.toggle("Fail on any unsupported claim", value=True, disabled=mode == "baseline",
                              help="The judge sometimes lists unsupported claims but still scores faithfulness high.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

options = {
    "mode": mode, "top_k": top_k, "max_retries": max_retries, "use_reranker": use_reranker,
    "use_query_rewrite": use_rewrite, "faithfulness_threshold": faith_t,
    "relevance_threshold": rel_t, "completeness_threshold": comp_t,
    "fail_on_unsupported_claims": strict_claims,
}


# --------------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------------
def fmt(x, digits=2) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def verdict_badge(resp: dict) -> str:
    rel, status = resp["reliability"], resp["status"]
    if status == "no_documents":
        return ":gray-background[NO DOCUMENTS]"
    if status == "insufficient_evidence":
        return ":orange-background[ABSTAINED · insufficient evidence]"
    if resp["mode"] == "baseline":
        return ":gray-background[BASELINE · unverified]"
    if rel["judge_error"]:
        return ":orange-background[UNVERIFIED · judge unavailable]"
    final = rel["final"]
    if final and final["verdict"] == "PASS":
        extra = f" after {rel['retries']} retr{'y' if rel['retries'] == 1 else 'ies'}" if rel["retries"] else ""
        return f":green-background[PASS{extra}]"
    return f":red-background[FAIL · best of {len(rel['attempts'])} attempts]"


def render_citations(resp: dict) -> None:
    for c in resp["citations"]:
        with st.expander(f"[{c['index']}] {c['label']}"):
            st.text(c["text"])


def render_evidence(resp: dict) -> None:
    chunks = resp["retrieved_chunks"]
    if not chunks:
        st.caption("No chunks retrieved.")
        return
    cited = {c["chunk_id"] for c in resp["citations"]}
    rows = [{
        "#": c["final_rank"], "source": c["document_name"],
        "loc": f"p.{c['page']}" if c["page"] is not None else f"chunk {c['chunk_index']}",
        "cited": "✓" if c["chunk_id"] in cited else "",
        "dense": c["dense_score"], "dense rank": c["dense_rank"], "bm25": c["bm25_score"], "bm25 rank": c["bm25_rank"],
        "rrf": c["rrf_score"], "fusion rank": c["fusion_rank"], "rerank": c["rerank_score"],
    } for c in chunks]
    st.dataframe(rows, hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("dense", "bm25", "rrf", "rerank")})
    st.caption("Ranks show where each chunk sat in the dense and BM25 lists before fusion and reranking.")
    for c in chunks:
        with st.expander(f"#{c['final_rank']} · {c['document_name']} · "
                         f"{'page ' + str(c['page']) if c['page'] is not None else 'chunk ' + str(c['chunk_index'])}"):
            st.text(c["text"])


def score_metrics(judge: dict, baseline: dict | None = None) -> None:
    cols = st.columns(3)
    for col, key in zip(cols, ("faithfulness", "relevance", "completeness")):
        delta = None if baseline is None else round(judge[key] - baseline[key], 2) or None
        col.metric(key.capitalize(), fmt(judge[key]), delta=delta)


def render_reliability(resp: dict) -> None:
    rel = resp["reliability"]
    if not rel["enabled"]:
        st.caption("Reliability loop not run (baseline mode or early abstention).")
        return
    if rel["judge_error"]:
        st.warning(f"Judge unavailable: {rel['judge_error']}. Answer returned unverified.")
        return
    final, initial = rel["final"], rel["initial"]
    if rel["improved"]:
        st.success(f"The adversarial loop improved the answer: faithfulness {fmt(initial['faithfulness'])} → "
                   f"{fmt(final['faithfulness'])}, overall verdict {initial['verdict']} → {final['verdict']}.")
    st.markdown(f"**Verdict:** {verdict_badge(resp)}")
    score_metrics(final, initial if rel["improved"] else None)
    if final["failed_checks"]:
        st.markdown(f"Failed checks: `{', '.join(final['failed_checks'])}`")
    if final["unsupported_claims"]:
        st.markdown("**Unsupported claims flagged:**")
        for claim in final["unsupported_claims"]:
            st.markdown(f"- {md_safe(claim)}")
    st.caption(f"Judge reasoning: {final['reason']}")


def render_critique(resp: dict) -> None:
    attempts = resp["reliability"]["attempts"]
    for a in attempts:
        j = a["judge"]
        label = f"Attempt {a['attempt']}" + (" (initial)" if a["attempt"] == 0 else " (regenerated)")
        verdict = f"{j['verdict']} · f={fmt(j['faithfulness'])} r={fmt(j['relevance'])} c={fmt(j['completeness'])}" if j else "not judged"
        st.markdown(f"**{label}** — {verdict}")
        st.markdown(f"> {md_safe(a['answer'])}".replace("\n", "\n> "))
        crit = a["critique"]
        if crit:
            st.markdown("**Critic found:**")
            for key, title in [("unsupported_claims", "Unsupported"), ("contradictions", "Contradiction"),
                               ("missing_evidence", "Missing"), ("weak_reasoning", "Weak reasoning"),
                               ("irrelevant_content", "Irrelevant")]:
                for item in crit[key]:
                    st.markdown(f"- *{title}:* {md_safe(item)}")
            if crit["instructions"]:
                st.caption(f"Instructions to regenerator: {crit['instructions']}")
        st.divider()


def render_trace(resp: dict) -> None:
    t = resp["trace"]
    spans = t["spans"]
    longest = max((s["duration_ms"] for s in spans), default=1) or 1
    rows = []
    for s in spans:
        attrs = s["attributes"]
        failed = s["status"] == "error" or attrs.get("verdict") == "FAIL" or attrs.get("passed") is False
        icon = "⏭" if s["status"] == "skipped" else ("✗" if failed else "✓")
        cls = "skip" if s["status"] == "skipped" else ("fail" if failed else ("loop" if s["name"] in {"critic", "regeneration"} else ""))
        width = max(1.0, 100 * s["duration_ms"] / longest) if s["status"] != "skipped" else 1.0
        label = STAGE_LABELS.get(s["name"], s["name"])
        dur = "skipped" if s["status"] == "skipped" else (f"{s['duration_ms'] / 1000:.2f}s" if s["duration_ms"] >= 1000 else f"{s['duration_ms']:.0f}ms")
        rows.append(
            f"<div class='trace-row'><span>{icon}</span><span>{html.escape(label)}</span>"
            f"<span class='muted'>{html.escape(dur)}</span><div class='trace-bar {cls}' style='width:{width:.1f}%'></div></div>"
        )
    rel = resp["reliability"]
    final = rel["final"]["verdict"] if rel["final"] else ("ABSTAINED" if resp["status"] != "answered" else "UNVERIFIED")
    st.markdown(f"<div class='trace'>{''.join(rows)}</div>", unsafe_allow_html=True)
    st.markdown(f"**Final:** {final} · **Retries:** {rel['retries']} · **Total:** {t['total_ms'] / 1000:.2f}s")
    c = st.columns(4)
    c[0].metric("LLM calls", t["llm_calls"])
    c[1].metric("Prompt tokens", f"{t['prompt_tokens']:,}")
    c[2].metric("Completion tokens", f"{t['completion_tokens']:,}")
    c[3].metric("Est. cost", "n/a" if t["estimated_cost_usd"] is None else f"${t['estimated_cost_usd']:.5f}",
                help="Set PRICE_PROMPT_PER_1M / PRICE_COMPLETION_PER_1M on the API to estimate cost.")
    st.caption(f"request_id `{t['request_id']}` · retrieval {t['retrieval_ms']:.0f}ms · rerank {t['rerank_ms']:.0f}ms · "
               f"generation {t['generation_ms']:.0f}ms · evaluation {t['evaluation_ms']:.0f}ms · chunks {t['retrieved_chunks']}")
    if t.get("throttle_ms"):
        st.caption(f"⚠️ {t['throttle_ms'] / 1000:.1f}s of the total was spent waiting on LLM provider rate limits "
                   f"({t['llm_retries']} retried call{'s' if t['llm_retries'] != 1 else ''}).")
    with st.expander("Span attributes"):
        st.json(spans, expanded=False)


def render_response(resp: dict, *, compact: bool = False) -> None:
    st.markdown(verdict_badge(resp))
    if resp["query_rewritten"]:
        st.caption(f"Searched as: *{resp['retrieval_query']}*")
    st.markdown(md_safe(resp["answer"]))
    if resp["citations"]:
        st.markdown("**Sources**")
        render_citations(resp)
    if compact:
        return
    with st.expander(f"📚 Retrieved evidence ({len(resp['retrieved_chunks'])} chunks)"):
        render_evidence(resp)
    with st.expander("⚖️ Reliability evaluation"):
        render_reliability(resp)
    if resp["reliability"]["retries"] > 0:
        with st.expander(f"⚔️ Adversarial critique ({resp['reliability']['retries']} retr"
                         f"{'y' if resp['reliability']['retries'] == 1 else 'ies'})", expanded=True):
            render_critique(resp)
    with st.expander(f"⏱ Execution trace ({resp['trace']['total_ms'] / 1000:.2f}s)"):
        render_trace(resp)


def render_assistant(msg: dict) -> None:
    if "error" in msg:
        st.error(msg["error"])
        return
    if msg.get("baseline"):
        left, right = st.columns(2)
        with left:
            st.markdown("##### Baseline RAG")
            render_response(msg["baseline"], compact=True)
            b = msg["baseline"]["trace"]
            st.caption(f"{b['llm_calls']} LLM calls · {b['total_ms'] / 1000:.2f}s")
        with right:
            st.markdown(f"##### {msg['response']['mode'].capitalize()} RAG")
            render_response(msg["response"], compact=True)
            a = msg["response"]["trace"]
            st.caption(f"{a['llm_calls']} LLM calls · {a['total_ms'] / 1000:.2f}s")
        st.markdown("---")
        with st.expander("Details for the adversarial run"):
            render_response(msg["response"])
    else:
        render_response(msg["response"])


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------
st.markdown("<div class='lab-title'>RAG Reliability Lab</div>"
            "<div class='lab-sub'>Hybrid retrieval · cross-encoder reranking · conditional adversarial verification</div>",
            unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Ask a question about your documents…")
if not st.session_state.messages and not prompt:
    st.info("Upload documents in the sidebar, then ask a question. Each answer shows its sources, the retrieval scores, "
            "the judge's scores, any adversarial critique, and a per-stage execution trace.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(md_safe(msg["content"]))
        else:
            render_assistant(msg)

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
               if m.get("content")]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(md_safe(prompt))
    with st.chat_message("assistant"):
        msg: dict = {"role": "assistant"}
        try:
            with st.spinner("Retrieving, generating and verifying…"):
                msg["response"] = client.query(prompt, history, options)
                if compare and mode == "adversarial":
                    msg["baseline"] = client.query(prompt, history, {**options, "mode": "baseline"})
            msg["content"] = msg["response"]["answer"]
        except APIError as e:
            hint = " The LLM provider is rate limiting; wait a few seconds and retry." if e.status == 429 else ""
            msg["error"] = f"{e.message}{hint}"
        render_assistant(msg)
    st.session_state.messages.append(msg)
