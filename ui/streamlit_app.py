"""RAG Reliability Lab: chat front end.

Implements the "Chat with inline citations" design (Geist / Geist Mono, neutral
palette): right-aligned user bubbles, answers with monospace citation chips,
a sources card, and a bordered composer. Retrieval scores, judge verdicts,
critiques and timings are logged by the backend (app/observability/query_log.py),
not shown here.

Safety: model output is HTML-escaped before citation chips are inserted, so the
only HTML rendered is the markup we generate ourselves.
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


def _setting(name: str, default: str = "") -> str:
    """Environment variable first, then Streamlit secrets (how Streamlit Community Cloud passes config)."""
    if os.getenv(name):
        return os.environ[name]
    try:
        return str(st.secrets[name])
    except Exception:
        return default


def _make_client():
    """RAG_BACKEND=http (default): call the FastAPI server. embedded: run the pipeline in this process."""
    if _setting("RAG_BACKEND", "http").lower() == "embedded":
        from embedded import EmbeddedClient

        return EmbeddedClient()
    return RagClient(_setting("API_BASE_URL", "http://localhost:8000"))


def _require_password() -> None:
    """Optional shared password (APP_PASSWORD) for public deployments.

    On a public deployment only this UI is exposed (in APP_ROLE=all the API listens on
    127.0.0.1; in embedded mode there is no API), so gating the UI protects the LLM quota.
    Unset APP_PASSWORD = no gate (local development).
    """
    expected = _setting("APP_PASSWORD")
    if not expected or st.session_state.get("authenticated"):
        return
    st.markdown("<div class='empty-state'><div class='empty-title'>RAG Reliability Lab</div>"
                "<div class='empty-sub'>This demo is password-protected.</div></div>", unsafe_allow_html=True)
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter", use_container_width=True, type="primary")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()


MIME = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown", "markdown": "text/markdown",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

# --------------------------------------------------------------------------------------------
# Design tokens (from the "Chat with inline citations" canvas)
#   text #0A0A0A / #18181B / #3F3F46 / #52525B / #71717A   lines #EBEBEB / #E4E4E7 / #F4F4F5
#   surfaces #FFFFFF / #FAFAFA / #F4F4F5 / #EFEFF0          sidebar 248px, top bar 52px, column 680px
# Fonts are set on text elements only (never with a universal selector), so Streamlit's
# icon font and KaTeX fonts keep working.
# --------------------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap');

    .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea, .stApp summary,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stWidgetLabel"] {
        font-family: 'Geist', system-ui, sans-serif !important;
    }
    .stApp { color: #0A0A0A; background: #FFFFFF; -webkit-font-smoothing: antialiased; }

    /* ---- layout: 680px column under a 52px top bar ---- */
    .block-container { max-width: 728px; padding: 40px 24px 24px; }
    [data-testid="stBottomBlockContainer"] { max-width: 728px; padding: 8px 24px 16px; }
    [data-testid="stHeader"] { background: transparent; height: 52px; }
    .topbar {
        position: fixed; top: 0; left: 0; right: 0; height: 52px; box-sizing: border-box; z-index: 99;
        padding: 0 16px 0 20px; display: flex; align-items: center; gap: 8px;
        background: #FFFFFF; border-bottom: 1px solid #EBEBEB; font-size: 13px;
    }
    body:has(section[data-testid="stSidebar"][aria-expanded="true"]) .topbar { left: 248px; }
    .topbar .crumb { color: #71717A; } .topbar .sep { color: #D4D4D8; }
    .topbar .title { font-weight: 500; color: #0A0A0A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"] { width: 248px !important; min-width: 248px !important;
        background: #FAFAFA; border-right: 1px solid #EBEBEB; }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0 12px; }
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { height: 10px; min-height: 10px; padding: 0; }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 0 0 16px; }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 13px; color: #3F3F46; }
    .brand { height: 32px; display: flex; align-items: center; gap: 8px; font-size: 13px; }
    .brand-logo { width: 20px; height: 20px; border-radius: 5px; background: #0A0A0A; color: #FFFFFF;
        font-size: 11px; font-weight: 600; display: flex; align-items: center; justify-content: center; }
    .brand-name { font-weight: 500; color: #0A0A0A; }
    .side-label { padding: 12px 8px 4px; font-size: 12px; color: #71717A; }
    .doc-row { min-height: 30px; display: flex; align-items: center; gap: 8px; font-size: 13px; color: #3F3F46; }
    .doc-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
    .doc-note { font-size: 12px; color: #71717A; padding-left: 2px; }
    section[data-testid="stSidebar"] .stButton button { min-height: 32px; border-radius: 6px; font-size: 13px; font-weight: 500; }
    section[data-testid="stSidebar"] .stButton button[kind="secondary"] { border: 1px solid #E4E4E7; background: #FFFFFF; color: #0A0A0A; }
    section[data-testid="stSidebar"] .stButton button[kind="tertiary"] { color: #52525B; }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #FFFFFF; border: 1px solid #E4E4E7; border-radius: 6px; padding: 8px; }

    /* ---- file-type tags (documents list and sources) ---- */
    .tag { font-family: 'Geist Mono', monospace !important; font-size: 10px; font-weight: 500; line-height: 16px;
        border-radius: 3px; padding: 0 4px; flex-shrink: 0; color: #3F3F46; border: 1px solid #E4E4E7; }
    .tag-pdf { color: #B42318; border-color: #F3D2CE; }
    .tag-doc { color: #1D4ED8; border-color: #D3DEF8; }

    /* ---- messages ---- */
    .row-user { display: flex; justify-content: flex-end; margin-top: 12px; }
    .msg-user { max-width: 480px; background: #F4F4F5; border-radius: 12px; padding: 10px 14px;
        font-size: 14px; line-height: 1.55; color: #0A0A0A; overflow-wrap: anywhere; }
    .asst-status { margin-top: 12px; height: 24px; display: flex; align-items: center; gap: 6px;
        font-size: 12.5px; color: #71717A; }
    .stMain [data-testid="stMarkdownContainer"] { position: relative; }  /* anchor for the citation popover */
    .stMain [data-testid="stMarkdownContainer"] p, .stMain [data-testid="stMarkdownContainer"] li {
        font-size: 14.5px; line-height: 1.7; color: #18181B; }
    .stMain [data-testid="stMarkdownContainer"] strong { font-weight: 600; }
    .stMain [data-testid="stMarkdownContainer"] code { font-family: 'Geist Mono', monospace !important; font-size: 12.5px;
        color: #18181B; background: #F4F4F5; border-radius: 4px; padding: 1px 5px; }
    .stMain [data-testid="stMarkdownContainer"] ul, .stMain [data-testid="stMarkdownContainer"] ol { padding-left: 20px; }
    .stMain [data-testid="stMarkdownContainer"] li { margin: 0 0 6px; }

    /* ---- citation chip + popover with the full passage ---- */
    .cite { position: static; display: inline; }
    .cite > a { display: inline-block; font-family: 'Geist Mono', monospace !important; font-size: 11px; line-height: 16px;
        padding: 0 5px; border: 1px solid #E4E4E7; background: #F4F4F5; border-radius: 4px;
        text-decoration: none !important; color: #3F3F46 !important; vertical-align: 1px; cursor: pointer;
        margin-left: 3px; }
    .cite > a:hover, .cite:focus-within > a { background: #EFEFF0; border-color: #D4D4D8; }
    .cite-pop { visibility: hidden; opacity: 0; transition: opacity 0.12s ease;
        /* top:auto keeps the box at the chip's line (its static position); margin drops it just below. */
        position: absolute; left: 0; right: 0; top: auto; margin-top: 22px; z-index: 1000;
        max-height: min(360px, 55vh); overflow-y: auto; overscroll-behavior: contain;
        padding: 10px 12px; border-radius: 8px; background: #FFFFFF; border: 1px solid #E4E4E7;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.10); font-size: 13px; line-height: 1.55; color: #3F3F46;
        font-weight: 400; text-align: left; white-space: normal; }
    .cite-pop b { display: block; margin-bottom: 4px; font-weight: 500; color: #0A0A0A; }
    .cite:hover .cite-pop, .cite:focus-within .cite-pop { visibility: visible; opacity: 1; }

    /* ---- sources card ---- */
    .src-card { border: 1px solid #EBEBEB; border-radius: 8px; overflow: hidden; font-size: 13px; margin-top: 2px; }
    .src-head { height: 32px; padding: 0 12px; display: flex; align-items: center; font-size: 12px; color: #71717A;
        background: #FAFAFA; border-bottom: 1px solid #EBEBEB; }
    .src-row { border-bottom: 1px solid #F4F4F5; } .src-row:last-child { border-bottom: 0; }
    .src-row > summary { list-style: none; padding: 10px 12px; display: flex; align-items: center; gap: 10px;
        color: #0A0A0A; cursor: pointer; }
    .src-row > summary::-webkit-details-marker { display: none; }
    .src-row > summary:hover { background: #FAFAFA; }
    .src-n { font-family: 'Geist Mono', monospace !important; font-size: 11px; color: #71717A; width: 12px; flex-shrink: 0; }
    .src-name { font-weight: 500; white-space: nowrap; flex-shrink: 0; max-width: 45%; overflow: hidden; text-overflow: ellipsis; }
    .src-loc { color: #71717A; white-space: nowrap; flex-shrink: 0; }
    .src-snip { flex-grow: 1; color: #71717A; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
    .src-full { padding: 2px 12px 12px 34px; color: #3F3F46; line-height: 1.6; }

    /* ---- composer ---- */
    [data-testid="stChatInput"] > div { border: 1px solid #E4E4E7; border-radius: 12px; background: #FFFFFF;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04); }
    [data-testid="stChatInput"] textarea { font-size: 14px; line-height: 1.5; color: #0A0A0A; }
    [data-testid="stChatInput"] textarea::placeholder { color: #71717A; }
    [data-testid="stChatInputSubmitButton"] { width: 32px; height: 32px; border-radius: 8px;
        background: #0A0A0A !important; color: #FFFFFF !important; }
    [data-testid="stChatInputSubmitButton"]:disabled { background: #E4E4E7 !important; color: #A1A1AA !important; }
    [data-testid="stBottomBlockContainer"]::after {
        content: "Answers are generated from your documents and can be wrong. Check the cited source.";
        display: block; text-align: center; font-size: 12px; color: #71717A; padding-top: 8px;
        font-family: 'Geist', system-ui, sans-serif; }

    .cursor { color: #71717A; animation: blink 1s steps(2, start) infinite; }
    @keyframes blink { to { visibility: hidden; } }
    .empty-state { text-align: center; margin-top: 18vh; }
    .empty-title { font-size: 20px; font-weight: 600; color: #0A0A0A; }
    .empty-sub { font-size: 14px; color: #71717A; margin-top: 6px; }
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


def _passage_html(text: str) -> str:
    """Full passage with paragraph breaks kept, safe to embed in our HTML."""
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n") if p.strip()]
    return "<br><br>".join(_inline_safe(p) for p in paragraphs)


def render_answer_html(answer: str, citations: list[dict], anchor: str = "") -> str:
    """Escape the answer, then turn [n] markers into citation chips with a full-passage popover.

    `anchor` (a per-message prefix) makes each chip link to its row in the sources card.
    """
    by_index = {c["index"]: c for c in citations}
    text = _escape_with_math(answer)

    def chip(match: re.Match) -> str:
        n = int(match.group(1))
        c = by_index.get(n)
        if c is None:
            return ""  # the API already strips invalid markers; be defensive anyway
        href = f' href="#{anchor}-{n}"' if anchor else ' role="button"'
        return (
            f'<span class="cite"><a tabindex="0"{href} aria-label="Source {n}: {_inline_safe(c["label"])}">{n}</a>'
            f'<span class="cite-pop"><b>{_inline_safe(c["label"])}</b>{_passage_html(c["text"])}</span></span>'
        )

    return _MARKER.sub(chip, text)


# Model-native citation styles (gpt-oss emits "【4†L1-L3】") and markers cut off mid-stream.
_NATIVE_MARKER = re.compile(r"【(\d+)(?:†[^】]*)?】|\[(\d+)†[^\]]*\]")
_PARTIAL_TAIL = re.compile(r"(【[^】]*|\[\d*(?:†[^\]]*)?)$")


def render_draft_html(draft: str, sources: list[dict]) -> str:
    """Render a partially streamed answer: normalize citation markers, hide a half-received one, add a cursor."""
    text = _NATIVE_MARKER.sub(lambda m: f"[{m.group(1) or m.group(2)}]", draft)
    text = _PARTIAL_TAIL.sub("", text)
    return render_answer_html(text, sources) + ' <span class="cursor">▍</span>'


def _file_tag(filename: str) -> str:
    """File-type tag as in the design: PDF (red), DOC (blue), MD / TXT (neutral)."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    label, cls = {"pdf": ("PDF", "tag-pdf"), "docx": ("DOC", "tag-doc"), "md": ("MD", ""),
                  "markdown": ("MD", ""), "txt": ("TXT", "")}.get(ext, (ext.upper()[:4] or "FILE", ""))
    return f'<span class="tag {cls}">{label}</span>'


def render_sources_html(citations: list[dict], anchor: str) -> str:
    """The sources card: one row per cited passage; a row expands to the full passage."""
    rows = []
    for c in citations:
        location = f"p. {c['page']}" if c.get("page") is not None else f"chunk {c['chunk_index']}"
        snippet = " ".join(c["text"].split())
        snippet = snippet if len(snippet) <= 110 else snippet[:109] + "…"
        rows.append(
            f'<details class="src-row" id="{anchor}-{c["index"]}"><summary>'
            f'<span class="src-n">{c["index"]}</span>{_file_tag(c["document_name"])}'
            f'<span class="src-name">{_inline_safe(c["document_name"])}</span>'
            f'<span class="src-loc">{location}</span>'
            f'<span class="src-snip">— {_inline_safe(snippet)}</span>'
            f'</summary><div class="src-full">{_passage_html(c["text"])}</div></details>'
        )
    count = f"{len(citations)} source{'s' if len(citations) != 1 else ''}"
    return f'<div class="src-card"><div class="src-head">{count}</div>{"".join(rows)}</div>'


_CHEVRON = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"></path></svg>')


def status_html(text: str) -> str:
    return f'<div class="asst-status">{_CHEVRON}{_inline_safe(text)}</div>'


def search_summary(response: dict) -> str:
    """"Searched 2 documents · 5 passages · 1.4s", from the retrieval results and the trace."""
    chunks = response.get("retrieved_chunks", [])
    docs = len({c["document_name"] for c in chunks})
    seconds = response.get("trace", {}).get("total_ms", 0) / 1000
    if not chunks:
        return f"No matching passages · {seconds:.1f}s"
    return (f"Searched {docs} document{'s' if docs != 1 else ''} · "
            f"{len(chunks)} passage{'s' if len(chunks) != 1 else ''} · {seconds:.1f}s")


STAGE_TEXT = {
    "retrieving": "Searching your documents…",
    "generating": "Writing…",
    "verifying": "Checking the answer against your documents…",
    "revising": "Refining the answer…",
}


def render_user(text: str) -> None:
    body = _inline_safe(text).replace("\n", "<br>")
    st.markdown(f'<div class="row-user"><div class="msg-user">{body}</div></div>', unsafe_allow_html=True)


def render_assistant(msg: dict, anchor: str) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    citations = msg.get("citations", [])
    st.markdown(render_answer_html(msg["content"], citations, anchor), unsafe_allow_html=True)
    if citations:
        st.markdown(render_sources_html(citations, anchor), unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------
# Sidebar: documents + pipeline settings
# --------------------------------------------------------------------------------------------
STAGE_LABEL = {"queued": "queued", "parsing": "reading file", "embedding": "embedding", "indexing": "indexing"}


def _doc_row(d: dict) -> None:
    c1, c2 = st.columns([6, 1], vertical_alignment="center")
    c1.markdown(f'<div class="doc-row">{_file_tag(d["filename"])}'
                f'<span class="doc-name">{_inline_safe(d["filename"])}</span></div>', unsafe_allow_html=True)
    if d["status"] == "processing":
        stage = STAGE_LABEL.get(d.get("stage") or "queued", d.get("stage") or "")
        c1.progress(d.get("progress") or 0.0, text=stage)
    elif d["status"] == "failed":
        c1.markdown(f'<div class="doc-note">{_inline_safe(d.get("error") or "Indexing failed")}</div>',
                    unsafe_allow_html=True)
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
            st.markdown('<div class="doc-note">No documents yet. Upload a PDF, Markdown, TXT or DOCX file '
                        "(up to 100 MB).</div>", unsafe_allow_html=True)
        for d in docs:
            _doc_row(d)
        if busy and not any(d["status"] == "processing" for d in docs):
            st.rerun()  # indexing finished: full rerun stops the polling

    _panel()


def _label(text: str) -> None:
    st.markdown(f'<div class="side-label">{text}</div>', unsafe_allow_html=True)


_require_password()
client = _make_client()

with st.sidebar:
    brand, new_chat = st.columns([6, 1], vertical_alignment="center")
    brand.markdown('<div class="brand"><span class="brand-logo">R</span>'
                   '<span class="brand-name">RAG Reliability Lab</span></div>', unsafe_allow_html=True)
    if new_chat.button("", icon=":material/edit_square:", key="new-chat", help="New chat", type="tertiary"):
        st.session_state.messages = []
        st.rerun()

    _label("Documents")
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

    _label("Pipeline")
    mode = st.radio("Mode", ["adversarial", "baseline"], horizontal=True,
                    help="Baseline = single-pass RAG (no judge/critic). Adversarial = judge + conditional critique loop.")
    top_k = st.slider("Top-k context chunks", 1, 10, 5)
    use_reranker = st.toggle("Cross-encoder reranker", value=True)
    use_rewrite = st.toggle("Query rewrite (when needed)", value=True)

    _label("Reliability")
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
messages = st.session_state.messages

prompt = st.chat_input("Ask a follow-up…" if messages else "Ask a question about your documents…")

# Top bar: "Documents / <chat title>" (the title is the first question of the conversation).
first_question = next((m["content"] for m in messages if m["role"] == "user"), prompt or "New chat")
title = first_question if len(first_question) <= 64 else first_question[:63] + "…"
st.markdown(f'<div class="topbar"><span class="crumb">Documents</span><span class="sep">/</span>'
            f'<span class="title">{_inline_safe(title)}</span></div>', unsafe_allow_html=True)

if not messages and not prompt:
    st.markdown(
        "<div class='empty-state'><div class='empty-title'>Ask about your documents</div>"
        "<div class='empty-sub'>Answers cite their sources. Hover a number to read the passage.</div></div>",
        unsafe_allow_html=True,
    )

for i, msg in enumerate(messages):
    if msg["role"] == "user":
        render_user(msg["content"])
    else:
        if msg.get("summary"):
            st.markdown(status_html(msg["summary"]), unsafe_allow_html=True)
        render_assistant(msg, anchor=f"s{i}")

if prompt:
    history = [{"role": m["role"], "content": m["content"]} for m in messages[-6:]
               if m.get("content") and not m.get("error")]
    messages.append({"role": "user", "content": prompt})
    render_user(prompt)

    msg: dict = {"role": "assistant"}
    status_slot, answer_slot = st.empty(), st.empty()
    status_slot.markdown(status_html(STAGE_TEXT["retrieving"]), unsafe_allow_html=True)
    draft, sources, last_paint = "", [], 0.0
    try:
        for event, data in client.query_stream(prompt, history, OPTIONS):
            if event == "status":
                status_slot.markdown(status_html(STAGE_TEXT.get(data.get("stage"), "")), unsafe_allow_html=True)
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
                msg["summary"] = search_summary(data)
            elif event == "error":
                raise APIError(data.get("status", 500), data.get("code", "error"), data.get("message", "Error"))
        if "content" not in msg:
            raise APIError(502, "incomplete_stream", "The answer stream ended unexpectedly. Please try again.")
    except APIError as e:
        hint = " The model provider is rate limiting; try again in a few seconds." if e.status == 429 else ""
        msg["error"] = f"{e.message}{hint}"

    if msg.get("summary"):
        status_slot.markdown(status_html(msg["summary"]), unsafe_allow_html=True)
    else:
        status_slot.empty()
    with answer_slot.container():
        render_assistant(msg, anchor=f"s{len(messages)}")  # authoritative final answer (may differ from the draft)
    messages.append(msg)
    st.rerun()  # re-render from history: the composer switches to "Ask a follow-up…"
