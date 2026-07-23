"""
app.py — Medical AI Assistant (Streamlit)

A premium chat UI over the RAG pipeline in rag.py.

Run:  streamlit run app.py
Requires: python ingest.py has been run, and GROQ_API_KEY is set.
"""

import os
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------- config
st.set_page_config(
    page_title="Medical AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bridge Streamlit secrets -> environment BEFORE importing rag.py,
# since rag.py reads os.environ["GROQ_API_KEY"] at construction time.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ.setdefault("GROQ_API_KEY", st.secrets["GROQ_API_KEY"])
except Exception:
    pass

from rag import EMBED_MODEL, LLM_MODEL, TOP_K, MedicalRAG  # noqa: E402

GITHUB_URL = "https://github.com/minthukyaw488-commits/medical-rag-qa"

# Maps a source filename to a human-readable category badge.
CATEGORY_MAP = {
    "anemia": "Hematology",
    "blood_cells": "Hematology",
    "pneumonia": "Respiratory",
    "skin_lesion": "Dermatology",
    "medical_imaging_ai": "Imaging & AI",
}

SUGGESTIONS = [
    ("What are the five types of white blood cells?", "Hematology"),
    ("How does pneumonia appear on a chest X-ray?", "Respiratory"),
    ("What is the ABCDE rule for melanoma?", "Dermatology"),
    ("What is the difference between classification and segmentation?", "Imaging & AI"),
]


# ---------------------------------------------------------------- styling
def inject_css(dark: bool) -> None:
    if dark:
        tokens = """
            --bg: #0B1220;
            --surface: rgba(23, 32, 51, 0.72);
            --surface-solid: #172033;
            --border: rgba(148, 163, 184, 0.18);
            --text: #E8EEF9;
            --muted: #94A3B8;
            --hero-glow: radial-gradient(1100px 460px at 50% -10%,
                          rgba(37,99,235,0.30), transparent 65%);
            --shadow: 0 1px 2px rgba(0,0,0,.36), 0 12px 32px rgba(0,0,0,.34);
            --shadow-sm: 0 1px 2px rgba(0,0,0,.32);
        """
    else:
        tokens = """
            --bg: #F8FAFC;
            --surface: rgba(255, 255, 255, 0.78);
            --surface-solid: #FFFFFF;
            --border: rgba(15, 23, 42, 0.08);
            --text: #0F172A;
            --muted: #64748B;
            --hero-glow: radial-gradient(1100px 460px at 50% -10%,
                          rgba(14,165,233,0.16), transparent 65%);
            --shadow: 0 1px 2px rgba(15,23,42,.04), 0 12px 32px rgba(15,23,42,.07);
            --shadow-sm: 0 1px 2px rgba(15,23,42,.05);
        """

    st.markdown(
        f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {{
    {tokens}
    --primary: #2563EB;
    --secondary: #0EA5E9;
    --radius: 18px;
}}

/* hide default Streamlit chrome */
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
.stDeployButton {{ display: none; }}

html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.stApp {{
    background: var(--bg);
    background-image: var(--hero-glow);
    background-repeat: no-repeat;
    color: var(--text);
}}
.block-container {{ padding-top: 1.4rem; max-width: 980px; }}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.fade {{ animation: fadeUp .45s cubic-bezier(.22,.61,.36,1) both; }}

@media (prefers-reduced-motion: reduce) {{
    .fade {{ animation: none; }}
    * {{ transition: none !important; }}
}}

/* ---------- top bar ---------- */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: .7rem 1.15rem; margin-bottom: 1.6rem;
    background: var(--surface);
    -webkit-backdrop-filter: blur(14px); backdrop-filter: blur(14px);
    border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
}}
.brand {{ display: flex; align-items: center; gap: .6rem;
          font-weight: 700; font-size: .98rem; letter-spacing: -.01em; }}
.brand .dot {{
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: #fff; font-size: 15px;
}}
.navlinks {{ display: flex; align-items: center; gap: 1.25rem; }}
.navlinks a {{
    color: var(--muted); text-decoration: none;
    font-size: .86rem; font-weight: 500; transition: color .18s ease;
}}
.navlinks a:hover {{ color: var(--primary); }}

/* ---------- hero ---------- */
.hero {{ text-align: center; padding: 2.2rem 1rem 1.4rem; }}
.hero .mark {{
    width: 58px; height: 58px; margin: 0 auto 1.1rem; border-radius: 17px;
    display: grid; place-items: center; font-size: 27px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    box-shadow: 0 10px 26px rgba(37,99,235,.28);
}}
.hero h1 {{
    font-size: 2.5rem; font-weight: 800; letter-spacing: -.035em;
    margin: 0 0 .55rem; color: var(--text); line-height: 1.1;
}}
.hero p {{
    color: var(--muted); font-size: 1.02rem; margin: 0 auto; max-width: 520px;
    line-height: 1.55;
}}

/* ---------- disclaimer ---------- */
.notice {{
    display: flex; gap: .85rem; align-items: flex-start;
    background: var(--surface-solid);
    border: 1px solid var(--border); border-left: 3px solid var(--primary);
    border-radius: 14px; padding: .9rem 1.1rem;
    box-shadow: var(--shadow-sm); margin: 1.1rem 0 1.5rem;
}}
.notice .ico {{ font-size: 1.05rem; line-height: 1.4; }}
.notice p {{ margin: 0; font-size: .875rem; color: var(--muted); line-height: 1.55; }}

/* ---------- section label ---------- */
.eyebrow {{
    font-size: .72rem; font-weight: 600; letter-spacing: .09em;
    text-transform: uppercase; color: var(--muted); margin: .4rem 0 .7rem;
}}

/* ---------- suggestion buttons ---------- */
div[data-testid="column"] .stButton > button {{
    width: 100%; text-align: left; white-space: normal; height: auto;
    min-height: 78px; padding: .95rem 1.05rem;
    background: var(--surface-solid); color: var(--text);
    border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
    font-size: .9rem; font-weight: 500; line-height: 1.45;
    transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}}
div[data-testid="column"] .stButton > button:hover {{
    transform: translateY(-2px);
    border-color: var(--primary);
    box-shadow: var(--shadow);
    color: var(--text);
}}
div[data-testid="column"] .stButton > button:focus {{
    color: var(--text); border-color: var(--primary);
}}

/* ---------- chat ---------- */
[data-testid="stChatMessage"] {{
    background: transparent; padding: .15rem 0; border: none;
}}
[data-testid="stChatMessageContent"] {{ color: var(--text); }}

.bubble-user {{
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: #fff; padding: .8rem 1.1rem; border-radius: 18px 18px 5px 18px;
    display: inline-block; max-width: 100%; font-size: .94rem; line-height: 1.5;
    box-shadow: 0 6px 18px rgba(37,99,235,.22);
}}
.answer-card {{
    background: var(--surface-solid); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.15rem 1.3rem;
    box-shadow: var(--shadow); color: var(--text);
}}

/* ---------- meta chips ---------- */
.chips {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem; }}
.chip {{
    font-size: .72rem; font-weight: 600; padding: .3rem .65rem;
    border-radius: 999px; border: 1px solid var(--border);
    color: var(--muted); background: var(--surface);
}}
.chip.high {{ color: #059669; border-color: rgba(5,150,105,.3); }}
.chip.med  {{ color: #D97706; border-color: rgba(217,119,6,.3); }}
.chip.low  {{ color: #DC2626; border-color: rgba(220,38,38,.3); }}
.chip.cat  {{ color: var(--primary); border-color: rgba(37,99,235,.3); }}

/* ---------- source cards ---------- */
.srccard {{
    background: var(--surface-solid); border: 1px solid var(--border);
    border-radius: 14px; padding: .85rem 1rem; margin-bottom: .6rem;
    box-shadow: var(--shadow-sm);
    transition: transform .16s ease, box-shadow .16s ease;
}}
.srccard:hover {{ transform: translateY(-2px); box-shadow: var(--shadow); }}
.srchead {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .4rem; }}
.srcicon {{
    width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center;
    background: rgba(37,99,235,.1); color: var(--primary); font-size: 13px; flex: 0 0 auto;
}}
.srcname {{ font-weight: 600; font-size: .85rem; color: var(--text); }}
.srcscore {{
    margin-left: auto; font-size: .72rem; font-weight: 600;
    color: var(--primary); background: rgba(37,99,235,.09);
    padding: .18rem .5rem; border-radius: 999px;
}}
.srcprev {{ font-size: .8rem; color: var(--muted); line-height: 1.5; margin: 0; }}

/* ---------- input ---------- */
[data-testid="stChatInput"] {{
    border-radius: 16px; border: 1px solid var(--border);
    background: var(--surface-solid); box-shadow: var(--shadow);
    transition: border-color .18s ease, box-shadow .18s ease;
}}
[data-testid="stChatInput"]:focus-within {{
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(37,99,235,.14), var(--shadow);
}}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
    background: var(--surface-solid); border-right: 1px solid var(--border);
}}
[data-testid="stSidebar"] * {{ color: var(--text); }}
.kb {{
    display: flex; justify-content: space-between; align-items: center;
    padding: .5rem 0; border-bottom: 1px solid var(--border); font-size: .82rem;
}}
.kb:last-child {{ border-bottom: none; }}
.kb .k {{ color: var(--muted); }}
.kb .v {{ font-weight: 600; text-align: right; }}
.live {{
    display: inline-flex; align-items: center; gap: .4rem;
    font-size: .78rem; font-weight: 600; color: #059669;
}}
.live .pulse {{
    width: 7px; height: 7px; border-radius: 50%; background: #10B981;
    box-shadow: 0 0 0 0 rgba(16,185,129,.6); animation: pulse 2s infinite;
}}
@keyframes pulse {{
    70%  {{ box-shadow: 0 0 0 7px rgba(16,185,129,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(16,185,129,0); }}
}}

/* ---------- footer ---------- */
.foot {{
    margin-top: 2.6rem; padding-top: 1.3rem; border-top: 1px solid var(--border);
    text-align: center; color: var(--muted); font-size: .8rem; line-height: 1.7;
}}
.foot a {{ color: var(--primary); text-decoration: none; font-weight: 500; }}
.foot .stack {{
    display: flex; gap: .45rem; justify-content: center;
    flex-wrap: wrap; margin-bottom: .6rem;
}}
.foot .stack span {{
    border: 1px solid var(--border); border-radius: 999px;
    padding: .22rem .6rem; font-size: .73rem;
}}

@media (max-width: 640px) {{
    .hero h1 {{ font-size: 1.85rem; }}
    .topbar {{ flex-direction: column; gap: .7rem; }}
}}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner=False)
def load_rag() -> MedicalRAG:
    return MedicalRAG()


def dedupe_sources(sources: list[dict]) -> list[dict]:
    """One entry per file — keep the highest-scoring chunk from each."""
    best: dict[str, dict] = {}
    for s in sources:
        name = s["source"]
        if name not in best or s["score"] > best[name]["score"]:
            best[name] = s
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def confidence(sources: list[dict]) -> tuple[str, str]:
    """Bucket the top similarity score into a readable confidence level."""
    if not sources:
        return "Low", "low"
    top = max(s["score"] for s in sources)
    if top >= 0.50:
        return "High confidence", "high"
    if top >= 0.35:
        return "Medium confidence", "med"
    return "Low confidence", "low"


def category_of(sources: list[dict]) -> str | None:
    if not sources:
        return None
    stem = Path(sources[0]["source"]).stem.lower()
    return CATEGORY_MAP.get(stem)


def reading_time(text: str) -> str:
    words = len(text.split())
    return f"{max(1, round(words / 200))} min read"


def pretty_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").title()


def render_sources(sources: list[dict]) -> None:
    unique = dedupe_sources(sources)
    st.markdown(
        f"<div class='eyebrow'>Sources · {len(unique)} document"
        f"{'s' if len(unique) != 1 else ''}</div>",
        unsafe_allow_html=True,
    )
    for s in unique:
        preview = s["text"][:190].strip()
        st.markdown(
            f"""
<div class="srccard fade">
  <div class="srchead">
    <div class="srcicon">📄</div>
    <div class="srcname">{pretty_name(s['source'])}</div>
    <div class="srcscore">{s['score']:.2f}</div>
  </div>
  <p class="srcprev">{preview}…</p>
</div>""",
            unsafe_allow_html=True,
        )


def render_meta(answer: str, sources: list[dict]) -> None:
    label, cls = confidence(sources)
    cat = category_of(sources)
    chips = [f"<span class='chip {cls}'>{label}</span>"]
    if cat:
        chips.append(f"<span class='chip cat'>{cat}</span>")
    chips.append(f"<span class='chip'>{reading_time(answer)}</span>")
    st.markdown(f"<div class='chips'>{''.join(chips)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("dark", False)
st.session_state.setdefault("pending", None)
st.session_state.setdefault("times", [])

inject_css(st.session_state.dark)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Knowledge base")

    rag = load_rag()
    chunks = len(rag.metadata)
    docs = sorted({m["source"] for m in rag.metadata})

    st.markdown(
        "<div class='live'><span class='pulse'></span>Index ready</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="kb"><span class="k">Documents</span><span class="v">{len(docs)}</span></div>
<div class="kb"><span class="k">Chunks indexed</span><span class="v">{chunks}</span></div>
<div class="kb"><span class="k">Retrieved per query</span><span class="v">top {TOP_K}</span></div>
<div class="kb"><span class="k">Embeddings</span><span class="v">{EMBED_MODEL.split('/')[-1]}</span></div>
<div class="kb"><span class="k">Language model</span><span class="v">{LLM_MODEL}</span></div>
""",
        unsafe_allow_html=True,
    )

    with st.expander("Indexed documents"):
        for d in docs:
            st.markdown(f"- {pretty_name(d)}")

    st.markdown("### Session")
    times = st.session_state.times
    qcount = len([m for m in st.session_state.messages if m["role"] == "user"])
    avg = f"{sum(times)/len(times):.1f}s" if times else "—"
    st.markdown(
        f"""
<div class="kb"><span class="k">Questions asked</span><span class="v">{qcount}</span></div>
<div class="kb"><span class="k">Avg response time</span><span class="v">{avg}</span></div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.times = []
        st.rerun()

    if st.button(
        "Switch to light mode" if st.session_state.dark else "Switch to dark mode",
        use_container_width=True,
    ):
        st.session_state.dark = not st.session_state.dark
        st.rerun()

    st.caption(
        "To add documents: drop .txt or .pdf files into `data/`, "
        "run `python ingest.py`, then restart the app."
    )

# ---------------------------------------------------------------- header
st.markdown(
    f"""
<div class="topbar fade">
  <div class="brand"><div class="dot">✚</div>Medical AI Assistant</div>
  <div class="navlinks">
    <a href="{GITHUB_URL}" target="_blank" rel="noopener">GitHub</a>
    <a href="{GITHUB_URL}#readme" target="_blank" rel="noopener">About</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- hero
if not st.session_state.messages:
    st.markdown(
        """
<div class="hero fade">
  <div class="mark">🩺</div>
  <h1>Medical AI Assistant</h1>
  <p>Ask evidence-based medical questions. Every answer is drawn from the
     indexed documents and shows the sources it came from.</p>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="notice fade">
  <div class="ico"></div>
  <p>This assistant provides educational information and does not replace
     professional medical advice. It cannot diagnose conditions or recommend
     treatment. Consult a qualified doctor for medical concerns.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- suggestions
if not st.session_state.messages:
    st.markdown("<div class='eyebrow'>Try asking</div>", unsafe_allow_html=True)
    for row in (SUGGESTIONS[:2], SUGGESTIONS[2:]):
        cols = st.columns(2, gap="small")
        for col, (question, tag) in zip(cols, row):
            with col:
                if st.button(f"{question}\n\n{tag}", key=f"sug_{question}"):
                    st.session_state.pending = question
                    st.rerun()

# ---------------------------------------------------------------- history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(
                f"<div class='bubble-user'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
    else:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)
            render_meta(msg["content"], msg.get("sources", []))

            a, b, c, _ = st.columns([1, 1, 1, 3])
            with a:
                st.download_button(
                    "Download",
                    msg["content"],
                    file_name="medical-answer.md",
                    key=f"dl_{i}",
                    use_container_width=True,
                )
            with b:
                if st.button("Regenerate", key=f"rg_{i}", use_container_width=True):
                    # re-ask the question that produced this answer
                    for prev in reversed(st.session_state.messages[:i]):
                        if prev["role"] == "user":
                            st.session_state.pending = prev["content"]
                            st.session_state.messages = st.session_state.messages[:i]
                            st.rerun()
            with c:
                with st.popover("Copy", use_container_width=True):
                    st.code(msg["content"], language=None)

            if msg.get("sources"):
                render_sources(msg["sources"])

# ---------------------------------------------------------------- input
typed = st.chat_input("Ask about diseases, treatments, anatomy…")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(
            f"<div class='bubble-user'>{question}</div>", unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Searching the knowledge base…"):
            started = time.time()
            try:
                result = rag.answer(question)
            except Exception as exc:  # noqa: BLE001
                st.error(
                    "Couldn't reach the language model. Check that GROQ_API_KEY "
                    f"is set, then try again. ({type(exc).__name__})"
                )
                st.stop()
            elapsed = time.time() - started

        st.session_state.times.append(elapsed)
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown(result["answer"])
        st.markdown("</div>", unsafe_allow_html=True)
        render_meta(result["answer"], result["sources"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
    st.rerun()

# ---------------------------------------------------------------- footer
st.markdown(
    f"""
<div class="foot">
  <div class="stack">
    <span>Groq</span><span>Llama 3.3 70B</span><span>S-PubMedBERT</span>
    <span>sentence-transformers</span><span>NumPy</span><span>Streamlit</span>
  </div>
  <a href="{GITHUB_URL}" target="_blank" rel="noopener">View source on GitHub</a><br>
  Built by NOVEM (Min Thu Kyaw) · Medical AI, Konyang University
</div>
""",
    unsafe_allow_html=True,
)