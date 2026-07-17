"""
app.py — Streamlit chat UI.

Run:  streamlit run app.py
(Make sure you ran ingest.py first and set GROQ_API_KEY.)
"""

import streamlit as st
from rag import MedicalRAG

st.set_page_config(page_title="Medical Q&A Bot")
st.title("🩺 Medical Q&A Bot")
st.caption("RAG over your medical documents · powered by Groq LLaMA 3.3")

st.warning(
    "Educational demo only. This is not medical advice — "
    "always consult a qualified doctor.",
)


@st.cache_resource
def load_rag():
    return MedicalRAG()


def dedupe_sources(sources: list[dict]) -> list[dict]:
    """Keep one entry per file — the highest-scoring chunk from each."""
    best = {}
    for s in sources:
        name = s["source"]
        if name not in best or s["score"] > best[name]["score"]:
            best[name] = s
    # highest score first
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def render_sources(sources: list[dict]) -> None:
    unique = dedupe_sources(sources)
    with st.expander(f"Sources ({len(unique)})"):
        for s in unique:
            st.markdown(f"**{s['source']}** (score {s['score']:.2f})")
            st.caption(s["text"][:300] + "...")


rag = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# Input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = rag.answer(prompt)
        st.markdown(result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
