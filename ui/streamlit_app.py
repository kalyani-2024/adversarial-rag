"""RAG Reliability Lab: chat front end.

A plain chat interface: the user sees their question, the answer, and inline
citation badges ([1], [2]) that reveal the source on hover/click. Retrieval
scores, judge verdicts, critiques and timings are logged by the backend
(see app/observability/query_log.py), not shown here.

Safety: model output is HTML-escaped before citation badges are inserted, so
the only HTML rendered is the markup we generate ourselves.
"""

from __future__ import annotations

import hmac
import html
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import APIError, RagClient  # noqa: E402

st.set_page_config(page_title="RAG Reliability Lab", page_icon="🧪", layout="centered")


def _api_base_url() -> str:
    if os.getenv("API_BASE_URL"):
        return os.environ["API_BASE_URL"]
    try:
        return st.secrets["API_BASE_URL"]
    except Exception:
        return "http://localhost:8000"


def _require_password() -> None:
    """Optional shared password (APP_PASSWORD) for public deployments.

    On a Hugging Face Space only this UI is public (the API listens on
    127.0.0.1 inside the container), so gating the UI protects the LLM quota.
    Unset APP_PASSWORD = no gate (local development).
    """
    expected = os.getenv("APP_PASSWORD", "")
    if not expected or st.session_state.get("authenticated"):
        return
    st.markdown("<div class='empty-state'><h1>RAG Reliability Lab</h1><p>This demo is password-protected.</p></div>",
                unsafe_allow_html=True)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", use_container_width=True, type="primary")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


client = RagClient(_api_base_url())

MIME = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 6rem; }
    /* The popover is positioned against the whole answer block (not the badge), so it spans the
       answer's width and can never be clipped at the sides of the chat column. */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { position: relative; }
    .cite { position: static; display: inline-block; margin: 0 1px; vertical-align: super; line-height: 1; }
    .cite > a {
        display: inline-block; min-width: 1.15em; padding: 1px 5px; border-radius: 999px;
        font-size: 0.68em; font-weight: 600; text-align: center; text-decoration: none; cursor: pointer;
        color: #4338ca; background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .cite > a:hover, .cite:focus-within > a { background: rgba(99, 102, 241, 0.22); }
    .cite-pop {
        visibility: hidden; opacity: 0; transition: opacity 0.12s ease;
        /* top:auto keeps the box at the badge's line (its static position); margin drops it just below. */
        position: absolute; left: 0; right: 0; top: auto; margin-top: 1.25em;
        max-height: min(360px, 55vh); overflow-y: auto; overscroll-behavior: contain; z-index: 1000; padding: 10px 12px; border-radius: 10px;
        background: #ffffff; color: #1f2937; border: 1px solid #e5e7eb;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.14);
        font-size: 0.8rem; font-weight: 400; line-height: 1.45; text-align: left; white-space: normal;
        vertical-align: baseline;
    }
    .cite-pop b { display: block; margin-bottom: 4px; color: #4338ca; font-size: 0.78rem; }
    .cite:hover .cite-pop, .cite:focus-within .cite-pop { visibility: visible; opacity: 1; }
    @media (prefers-color-scheme: dark) {
        .cite > a { color: #c7d2fe; }
        .cite-pop { background: #1f2330; color: #e5e7eb; border-color: #374151; }
        .cite-pop b { color: #a5b4fc; }
    }
    .cursor { color: #6366f1; animation: blink 1s steps(2, start) infinite; }
    @keyframes blink { to { visibility: hidden; } }
    .empty-state { text-align: center; color: #6b7280; margin-top: 18vh; }
    .empty-state h1 { font-size: 1.8rem; font-weight: 700; color: inherit; margin-bottom: 0.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

_MARKER = re.compile(r"\[(\d+)\]")
# Text inside our HTML is still parsed as Markdown by Streamlit; neutralize its syntax characters.
_MD_ENTITIES = str.maketrans({"*": "&#42;", "_": "&#95;", "`": "&#96;", "[": "&#91;", "]": "&#93;", "$": "&#36;", "~": "&#126;"})


def _inline_safe(text: str) -> str:
    return html.escape(text).translate(_MD_ENTITIES)


# LaTeX the model may emit: \[ ... \] (display) and \( ... \) (inline).
_LATEX = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)


def _escape_prose(text: str) -> str:
    return html.escape(text, quote=False).replace("$", r"\$")  # literal $ must not start math


def _escape_with_math(text: str) -> str:
    """Escape prose; convert LaTeX delimiters to Streamlit math ($$...$$ / $...$)."""
    out, pos = [], 0
    for m in _LATEX.finditer(text):
        out.append(_escape_prose(text[pos:m.start()]))
        display, inline = m.group(1), m.group(2)
        # Math is not HTML-escaped (KaTeX would show entities), so remove anything that could form a tag.
        body = (display or inline).strip().replace("$", "").replace("<", r" \lt ").replace(">", r" \gt ")
        out.append(f"\n$$\n{body}\n$$\n" if display is not None else f"${body}$")
        pos = m.end()
    out.append(_escape_prose(text[pos:]))
    return "".join(out)


def render_answer_html(answer: str, citations: list[dict]) -> str:
    """Escape the answer, then turn [n] markers into hover/click source badges."""
    by_index = {c["index"]: c for c in citations}
    text = _escape_with_math(answer)

    def badge(match: re.Match) -> str:
        n = int(match.group(1))
        c = by_index.get(n)
        if c is None:
            return ""  # the API already strips invalid markers; be defensive anyway
        # Full passage (paragraph breaks kept); the popover scrolls for long chunks.
        paragraphs = [" ".join(p.split()) for p in c["text"].split("\n\n") if p.strip()]
        passage = "<br><br>".join(_inline_safe(p) for p in paragraphs)
        return (
            f'<span class="cite"><a tabindex="0" role="button" aria-label="Source {n}: {_inline_safe(c["label"])}">{n}</a>'
            f'<span class="cite-pop"><b>{_inline_safe(c["label"])}</b>{passage}</span></span>'
        )

    return _MARKER.sub(badge, text)


# Model-native citation styles (gpt-oss emits "【4†L1-L3】") and markers cut off mid-stream.
_NATIVE_MARKER = re.compile(r"【(\d+)(?:†[^】]*)?】|\[(\d+)†[^\]]*\]")
_PARTIAL_TAIL = re.compile(r"(【[^】]*|\[\d*(?:†[^\]]*)?)$")


def render_draft_html(draft: str, sources: list[dict]) -> str:
    """Render a partially streamed answer: normalize citation markers, hide a half-received one, add a cursor."""
    text = _NATIVE_MARKER.sub(lambda m: f"[{m.group(1) or m.group(2)}]", draft)
    text = _PARTIAL_TAIL.sub("", text)
    return render_answer_html(text, sources) + ' <span class="cursor">▍</span>'


STAGE_TEXT = {
    "retrieving": "Searching your documents…",
    "generating": "Writing…",
    "verifying": "Checking the answer against your documents…",
    "revising": "Refining the answer…",
}


def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    citations = msg.get("citations", [])
    st.markdown(render_answer_html(msg["content"], citations), unsafe_allow_html=True)
    if citations:
        with st.expander(f"Sources ({len(citations)})"):
            for c in citations:
                st.markdown(f"**[{c['index']}] {html.escape(c['label'])}**")
                st.text(c["text"])


# --------------------------------------------------------------------------------------------
# Sidebar: documents + pipeline settings
# --------------------------------------------------------------------------------------------
STAGE_LABEL = {"queued": "queued", "parsing": "reading file", "embedding": "embedding", "indexing": "indexing"}


def _doc_row(d: dict) -> None:
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    name = html.escape(d["filename"])
    if d["status"] == "processing":
        stage = STAGE_LABEL.get(d.get("stage") or "queued", d.get("stage") or "")
        c1.markdown(f"⏳ {name}")
        c1.progress(d.get("progress") or 0.0, text=stage)
    elif d["status"] == "failed":
        c1.markdown(f"⚠️ {name}")
        c1.caption(d.get("error") or "Indexing failed")
    else:
        c1.markdown(f"📄 {name}")
    help_text = "Cancel indexing" if d["status"] == "processing" else "Remove document"
    if c2.button("", icon=":material/delete:", key=f"del-{d['id']}", help=help_text, type="tertiary"):
        try:
            client.delete(d["id"])
            st.rerun()
        except APIError as e:
            st.error(e.message)


def _fetch_docs() -> tuple[list[dict], str | None]:
    try:
        return client.list_documents()["documents"], None
    except APIError as e:
        return [], e.message


def documents_panel(initial: list[dict], error: str | None) -> None:
    """Document list; polls every 2 s only while something is being indexed."""
    busy = any(d["status"] == "processing" for d in initial)

    @st.fragment(run_every=2 if busy else None)
    def _panel() -> None:
        docs, err = _fetch_docs() if busy else (initial, error)
        if err:
            st.error(err)
        elif not docs:
            st.caption("No documents yet. Upload a PDF, Markdown, TXT or DOCX file (up to 100 MB).")
        for d in docs:
            _doc_row(d)
        if busy and not any(d["status"] == "processing" for d in docs):
            st.rerun()  # indexing finished: full rerun stops the polling

    _panel()


_require_password()

with st.sidebar:
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### Documents")
    uploads = st.file_uploader("Upload documents", type=["pdf", "txt", "md", "markdown", "docx"],
                               accept_multiple_files=True, label_visibility="collapsed")
    if uploads and st.button("Add to knowledge base", use_container_width=True, type="primary"):
        for f in uploads:
            try:
                client.upload(f.name, f.getvalue(), MIME.get(f.name.rsplit(".", 1)[-1].lower(), "application/octet-stream"))
                st.toast(f"Indexing {f.name} in the background")
            except APIError as e:
                (st.info if e.code == "duplicate_document" else st.error)(f"{f.name}: {e.message}")
    documents_panel(*_fetch_docs())

    st.markdown("#### Pipeline")
    mode = st.radio("Mode", ["adversarial", "baseline"], horizontal=True,
                    help="Baseline = single-pass RAG (no judge/critic). Adversarial = judge + conditional critique loop.")
    top_k = st.slider("Top-k context chunks", 1, 10, 5)
    use_reranker = st.toggle("Cross-encoder reranker", value=True)
    use_rewrite = st.toggle("Query rewrite (when needed)", value=True)

    st.markdown("#### Reliability")
    adversarial = mode == "adversarial"
    max_retries = st.slider("Max retries", 0, 3, 2, disabled=not adversarial)
    faith_t = st.slider("Faithfulness threshold", 0.0, 1.0, 0.80, 0.05, disabled=not adversarial)
    rel_t = st.slider("Relevance threshold", 0.0, 1.0, 0.70, 0.05, disabled=not adversarial)
    comp_t = st.slider("Completeness threshold", 0.0, 1.0, 0.60, 0.05, disabled=not adversarial)
    strict_claims = st.toggle("Fail on any unsupported claim", value=True, disabled=not adversarial,
                              help="The judge sometimes lists unsupported claims but still scores faithfulness high.")

OPTIONS = {
    "mode": mode, "top_k": top_k, "use_reranker": use_reranker, "use_query_rewrite": use_rewrite,
    "max_retries": max_retries, "faithfulness_threshold": faith_t, "relevance_threshold": rel_t,
    "completeness_threshold": comp_t, "fail_on_unsupported_claims": strict_claims,
}


# --------------------------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Ask a question about your documents")

if not st.session_state.messages and not prompt:
    st.markdown(
        "<div class='empty-state'><h1>RAG Reliability Lab</h1>"
        "<p>Ask anything about your documents. Answers cite their sources. Hover a number to see it.</p></div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(html.escape(msg["content"], quote=False).replace("$", r"\$"))
        else:
            render_assistant(msg)

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]
               if m.get("content") and not m.get("error")]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(html.escape(prompt, quote=False).replace("$", r"\$"))

    with st.chat_message("assistant"):
        msg: dict = {"role": "assistant"}
        answer_slot, status_slot = st.empty(), st.empty()
        status_slot.caption(STAGE_TEXT["retrieving"])
        draft, sources, last_paint = "", [], 0.0
        try:
            for event, data in client.query_stream(prompt, history, OPTIONS):
                if event == "status":
                    status_slot.caption(STAGE_TEXT.get(data.get("stage"), ""))
                elif event == "sources":
                    sources = data["sources"]
                elif event == "token":
                    draft += data["text"]
                    now = time.monotonic()
                    if now - last_paint > 0.04:  # repaint at most ~25x/s
                        answer_slot.markdown(render_draft_html(draft, sources), unsafe_allow_html=True)
                        last_paint = now
                elif event == "reset":  # the draft failed verification; a revised answer streams next
                    draft = ""
                    answer_slot.markdown('<span class="cursor">▍</span>', unsafe_allow_html=True)
                elif event == "final":
                    msg["content"], msg["citations"] = data["answer"], data["citations"]
                elif event == "error":
                    raise APIError(data.get("status", 500), data.get("code", "error"), data.get("message", "Error"))
            if "content" not in msg:
                raise APIError(502, "incomplete_stream", "The answer stream ended unexpectedly. Please try again.")
        except APIError as e:
            hint = " The model provider is rate limiting; try again in a few seconds." if e.status == 429 else ""
            msg["error"] = f"{e.message}{hint}"
        status_slot.empty()
        with answer_slot.container():
            render_assistant(msg)  # authoritative final answer (may differ from the streamed draft)
    st.session_state.messages.append(msg)
