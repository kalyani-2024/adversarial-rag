"""RAG Reliability Lab: chat front end.

A plain chat interface: the user sees their question, the answer, and inline
citation badges ([1], [2]) that reveal the source on hover/click. Retrieval
scores, judge verdicts, critiques and timings are logged by the backend
(see app/observability/query_log.py), not shown here.

Safety: model output is HTML-escaped before citation badges are inserted, so
the only HTML rendered is the markup we generate ourselves.
"""

from __future__ import annotations

import html
import os
import re
import sys
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


client = RagClient(_api_base_url())

MIME = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 6rem; }
    .cite { position: relative; display: inline-block; margin: 0 1px; vertical-align: super; line-height: 1; }
    .cite > a {
        display: inline-block; min-width: 1.15em; padding: 1px 5px; border-radius: 999px;
        font-size: 0.68em; font-weight: 600; text-align: center; text-decoration: none; cursor: pointer;
        color: #4338ca; background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .cite > a:hover, .cite:focus-within > a { background: rgba(99, 102, 241, 0.22); }
    .cite-pop {
        visibility: hidden; opacity: 0; transition: opacity 0.12s ease;
        position: absolute; left: 50%; top: calc(100% + 6px); transform: translateX(-50%);
        max-height: 240px; overflow-y: auto;
        width: min(340px, 80vw); z-index: 1000; padding: 10px 12px; border-radius: 10px;
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


def render_answer_html(answer: str, citations: list[dict]) -> str:
    """Escape the answer, then turn [n] markers into hover/click source badges."""
    by_index = {c["index"]: c for c in citations}
    text = html.escape(answer, quote=False).replace("$", r"\$")

    def badge(match: re.Match) -> str:
        n = int(match.group(1))
        c = by_index.get(n)
        if c is None:
            return ""  # the API already strips invalid markers; be defensive anyway
        snippet = " ".join(c["text"].split())
        snippet = snippet if len(snippet) <= 320 else snippet[:319] + "…"
        return (
            f'<span class="cite"><a tabindex="0" role="button" aria-label="Source {n}: {_inline_safe(c["label"])}">{n}</a>'
            f'<span class="cite-pop"><b>{_inline_safe(c["label"])}</b>{_inline_safe(snippet)}</span></span>'
        )

    return _MARKER.sub(badge, text)


def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    st.markdown(render_answer_html(msg["content"], msg.get("citations", [])), unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------
# Sidebar: documents only
# --------------------------------------------------------------------------------------------
with st.sidebar:
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("#### Documents")
    uploads = st.file_uploader("Upload documents", type=["pdf", "txt", "md", "markdown", "docx"],
                               accept_multiple_files=True, label_visibility="collapsed")
    if uploads and st.button("Add to knowledge base", use_container_width=True, type="primary"):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    client.upload(f.name, f.getvalue(), MIME.get(f.name.rsplit(".", 1)[-1].lower(), "application/octet-stream"))
                    st.toast(f"Added {f.name}")
                except APIError as e:
                    (st.info if e.code == "duplicate_document" else st.error)(f"{f.name}: {e.message}")

    try:
        docs = client.list_documents()["documents"]
        api_error = None
    except APIError as e:
        docs, api_error = [], e.message

    if api_error:
        st.error(api_error)
    elif not docs:
        st.caption("No documents yet. Upload a PDF, Markdown, TXT or DOCX file to start.")
    for d in docs:
        c1, c2 = st.columns([6, 1], vertical_alignment="center")
        c1.markdown(f"📄 {html.escape(d['filename'])}")
        if c2.button("", icon=":material/delete:", key=f"del-{d['id']}", help="Remove document", type="tertiary"):
            try:
                client.delete(d["id"])
                st.rerun()
            except APIError as e:
                st.error(e.message)


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
        try:
            with st.spinner("Thinking…"):
                resp = client.query(prompt, history, {})
            msg["content"], msg["citations"] = resp["answer"], resp["citations"]
        except APIError as e:
            hint = " The model provider is rate limiting; try again in a few seconds." if e.status == 429 else ""
            msg["error"] = f"{e.message}{hint}"
        render_assistant(msg)
    st.session_state.messages.append(msg)
