"""
LegalMind Chat UI -- Streamlit frontend.

Run with:  streamlit run app.py
"""

import time
import httpx
import streamlit as st

API_BASE = "http://localhost:8000/api/v1"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalMind",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── RTL + styling ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Make the whole app support RTL for Arabic */
    .stChatMessage p, .stMarkdown p, .stTextInput input, .stTextArea textarea {
        direction: rtl;
        text-align: right;
        font-family: 'Segoe UI', 'Arial', sans-serif;
        font-size: 16px;
        line-height: 1.8;
    }
    /* Keep citations LTR */
    .citation-box {
        direction: ltr;
        text-align: left;
        background: #1e2a3a;
        border-left: 3px solid #4a9eff;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 13px;
        color: #cdd9e5;
    }
    .citation-box .excerpt {
        direction: rtl;
        text-align: right;
        color: #adbac7;
        margin-top: 6px;
        font-style: italic;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-cached  { background: #1f4a1f; color: #4caf50; }
    .badge-fresh   { background: #1a2f4a; color: #4a9eff; }
    .badge-latency { background: #2a2a1a; color: #ffc107; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ LegalMind")
    st.caption("Legal Knowledge Assistant")
    st.divider()

    st.subheader("🔍 Filters")
    doc_type = st.selectbox(
        "Document type",
        ["Any", "contract", "case_file", "pleading", "brief", "correspondence"],
    )
    client_id = st.text_input("Client ID (optional)")

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Retrieve top-K chunks", 5, 30, 20)
    top_n = st.slider("Re-rank to top-N", 1, 10, 5)

    st.divider()
    st.subheader("📂 Ingest Document")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded and st.button("Ingest", type="primary"):
        with st.spinner("Ingesting..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=300,
                )
                data = resp.json()
                if resp.status_code == 202:
                    st.success(f"✅ {data['chunks_stored']} chunks stored")
                else:
                    st.error(str(data))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Health check
    try:
        h = httpx.get(f"{API_BASE}/health", timeout=3)
        health = h.json().get("checks", {})
        qdrant_ok = health.get("qdrant") == "ok"
        redis_ok  = health.get("redis")  == "ok"
        st.markdown(
            f"**Qdrant** {'🟢' if qdrant_ok else '🔴'}  &nbsp; "
            f"**Redis** {'🟢' if redis_ok else '🔴'}",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown("**API** 🔴 offline")

# ── Citation renderer ─────────────────────────────────────────────────────────
def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"📎 المصادر ({len(citations)})", expanded=False):
        for i, c in enumerate(citations, 1):
            score = c.get("relevance_score", 0)
            excerpt = c.get("excerpt", "")
            filename = c.get("filename", "")
            doc_id   = c.get("document_id", "")[:8]
            chunk_id = c.get("chunk_id", "")[:8]
            st.markdown(
                f"""<div class="citation-box">
                    <strong>#{i}</strong> &nbsp;
                    📄 <code>{filename}</code> &nbsp;
                    🔑 <code>{doc_id}…:{chunk_id}…</code> &nbsp;
                    Score: <strong>{score:.3f}</strong>
                    <div class="excerpt">{excerpt[:300]}…</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("LegalMind — محرك البحث القانوني")
st.caption("اسأل عن أي بند أو مادة في الوثائق القانونية المُدرجة")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg.get("citations"):
            _render_citations(msg["citations"])


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("اكتب سؤالك القانوني هنا... / Type your legal question here...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build request payload
    payload: dict = {
        "query": query,
        "top_k": top_k,
        "top_n": top_n,
    }
    if doc_type != "Any":
        payload["filter_document_type"] = doc_type
    if client_id.strip():
        payload["filter_client_id"] = client_id.strip()

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("جارٍ البحث... / Searching..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/query",
                    json=payload,
                    timeout=120,
                )
                data = resp.json()

                answer    = data.get("answer", "")
                citations = data.get("citations", [])
                cached    = data.get("cached", False)
                latency   = data.get("latency_ms", 0)

                # Badges
                cache_badge = (
                    '<span class="badge badge-cached">⚡ Cached</span>'
                    if cached else
                    '<span class="badge badge-fresh">🔍 Fresh</span>'
                )
                latency_badge = (
                    f'<span class="badge badge-latency">⏱ {latency:.0f}ms</span>'
                )
                st.markdown(
                    f"{cache_badge} &nbsp; {latency_badge}",
                    unsafe_allow_html=True,
                )

                st.markdown(answer)
                _render_citations(citations)

                # Store in history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                })

            except httpx.TimeoutException:
                st.error("⏱ Request timed out. The model may still be loading -- try again.")
            except Exception as e:
                st.error(f"Error: {e}")
