"""
app.py — With conversation memory + source chunk viewer
"""
import os
import time
import warnings
import shutil
from pathlib import Path
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6e1; }

[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #c9c7c0 !important; }

.app-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    background: linear-gradient(135deg, #e8e6e1 0%, #7c6fff 60%, #ff6fb8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
    line-height: 1;
}
.app-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #7c6fff;
    background: #1a1730;
    border: 1px solid #2d2660;
    padding: 2px 8px;
    border-radius: 4px;
}
.app-sub {
    font-size: 0.85rem;
    color: #555;
    margin-bottom: 24px;
    font-family: 'DM Mono', monospace;
}
.answer-block {
    background: #0f0f1a;
    border-left: 2px solid #7c6fff;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 4px 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #e8e6e1;
}
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #12122a;
    color: #7c6fff;
    border: 1px solid #2d2660;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}
.guardrail-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a0f0f;
    color: #ff6b6b;
    border: 1px solid #3d1515;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
}
.memory-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #0f1a12;
    color: #4caf82;
    border: 1px solid #1a3d28;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    margin-left: 6px;
}
.chunk-box {
    background: #0c0c18;
    border: 1px solid #1a1a30;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    color: #888;
    line-height: 1.6;
}
.chunk-meta {
    font-size: 0.68rem;
    color: #7c6fff;
    margin-bottom: 6px;
    font-family: 'DM Mono', monospace;
}
[data-testid="stChatInput"] {
    background: #0f0f1a !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7c6fff !important;
    box-shadow: 0 0 0 3px rgba(124,111,255,0.12) !important;
}
.stButton > button {
    background: #14132a !important;
    color: #9e9db5 !important;
    border: 1px solid #2a2a40 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}
.stButton > button:hover {
    border-color: #7c6fff !important;
    color: #e8e6e1 !important;
    background: #1a1730 !important;
}
[data-testid="stFileUploader"] {
    background: #0f0f1a !important;
    border: 1px dashed #2a2a40 !important;
    border-radius: 10px !important;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 2px; }
section-divider { border: none; border-top: 1px solid #1a1a2e; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(exist_ok=True)
CHROMA_DIR = "chroma_db"


def get_paper_list():
    return sorted(PAPERS_DIR.glob("*.pdf"))


def get_paper_size(path):
    size = path.stat().st_size
    return f"{size/1024:.0f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"


def ingest_single_pdf(pdf_path):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from rag_chain import get_embeddings

    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(pages)
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="research_papers",
    )
    vectorstore.add_documents(chunks)
    return len(chunks)


def rebuild_vectorstore():
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
    import rag_chain
    rag_chain._vectorstore = None
    from ingest import ingest_papers
    ingest_papers()


# ── API key check ─────────────────────────────────────────────────────────
if not os.getenv("GROQ_API_KEY"):
    st.error("❌ **GROQ_API_KEY** not found. Add it to your `.env` file.")
    st.stop()

if not os.path.exists(CHROMA_DIR):
    with st.spinner("Setting up knowledge base..."):
        from ingest import ingest_papers
        ingest_papers()

# ── Load agent (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_agent(k: int):
    from rag_chain import build_rag_chain
    from guardrails_wrapper import GuardrailedAgent
    return GuardrailedAgent(build_rag_chain(top_k=k))


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px'>
        <div style='font-family: Syne, sans-serif; font-weight: 800; font-size: 1.3rem;
                    background: linear-gradient(135deg, #e8e6e1, #7c6fff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🧠 ResearchMind
        </div>
        <div style='font-family: DM Mono, monospace; font-size: 0.65rem; color: #444; margin-top: 2px;'>
            RAG · GROQ · LANGCHAIN · MEMORY
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#666; font-family:DM Mono,monospace; margin-bottom:8px;'>KNOWLEDGE BASE</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            dest = PAPERS_DIR / uf.name
            if not dest.exists():
                with st.spinner(f"Ingesting {uf.name}..."):
                    dest.write_bytes(uf.read())
                    try:
                        n_chunks = ingest_single_pdf(dest)
                        import rag_chain
                        rag_chain._vectorstore = None
                        st.success(f"✓ {uf.name} — {n_chunks} chunks")
                    except Exception as e:
                        st.error(f"Failed: {e}")
                        dest.unlink(missing_ok=True)
            else:
                st.info(f"Already in KB: {uf.name}")

    papers = get_paper_list()
    if papers:
        st.markdown(f"<div style='font-size:0.7rem;color:#555;font-family:DM Mono,monospace;margin:10px 0 6px;'>{len(papers)} PAPER(S)</div>", unsafe_allow_html=True)
        for p in papers:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"<div style='font-size:0.75rem;font-family:DM Mono,monospace;color:#9e9db5;"
                    f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;' title='{p.name}'>"
                    f"📄 {p.name[:26]}{'…' if len(p.name)>26 else ''}</div>"
                    f"<div style='font-size:0.65rem;color:#444;'>{get_paper_size(p)}</div>",
                    unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{p.name}"):
                    p.unlink(missing_ok=True)
                    with st.spinner("Rebuilding KB..."):
                        rebuild_vectorstore()
                    st.rerun()
    else:
        st.markdown("<div style='font-size:0.75rem;color:#444;font-family:DM Mono,monospace;padding:8px 0;'>No papers. Upload above.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem; color:#666; font-family:DM Mono,monospace; margin-bottom:8px;'>SETTINGS</div>", unsafe_allow_html=True)

    top_k = st.slider("Chunks retrieved", 2, 8, 4)
    show_sources = st.toggle("Show sources", value=True)
    show_chunks = st.toggle("Show retrieved chunks", value=False)
    memory_window = st.slider("Memory window (exchanges)", 1, 10, 5,
                               help="How many past Q&A pairs the model remembers")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("📊 Eval"):
            with st.spinner("Running RAGAS..."):
                try:
                    from evaluate import run_evaluation
                    scores = run_evaluation()
                    st.session_state.eval_scores = scores
                    st.success("Done!")
                except Exception as e:
                    st.error(f"{e}")

    if "eval_scores" in st.session_state:
        for k, v in st.session_state.eval_scores.items():
            st.markdown(
                f"<div style='background:#0f0f1a;border:1px solid #1e1e35;border-radius:8px;"
                f"padding:6px 10px;font-family:DM Mono,monospace;font-size:0.72rem;"
                f"color:#7c6fff;margin:3px 0;'>{v:.2f}"
                f"<span style='display:block;color:#555;font-size:0.65rem;'>{k.replace('_',' ')}</span></div>",
                unsafe_allow_html=True)

# ── Load agent ────────────────────────────────────────────────────────────
agent = load_agent(top_k)

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ── Main header ───────────────────────────────────────────────────────────
papers = get_paper_list()
n_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
memory_active = n_msgs > 0

col_t, col_tag = st.columns([6, 1])
with col_t:
    st.markdown("<div class='app-title'>ResearchMind</div>", unsafe_allow_html=True)
with col_tag:
    st.markdown("<div class='app-tag' style='margin-top:12px;'>v2.1</div>", unsafe_allow_html=True)

st.markdown("<div class='app-sub'>Ask anything — answers grounded in your uploaded research papers</div>", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Papers", len(papers))
c2.metric("Questions", n_msgs)
c3.metric("Top-K", top_k)
c4.metric("Memory", f"{min(n_msgs, memory_window)}/{memory_window}")
c5.metric("Model", "LLaMA 3.1")

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f"<div class='answer-block'>{msg['content']}</div>",
                unsafe_allow_html=True)

            # Memory indicator — show on all except first answer
            if msg.get("used_memory"):
                st.markdown(
                    f'<span class="memory-badge">🧠 Used {msg["used_memory"]} prior exchange(s)</span>',
                    unsafe_allow_html=True)

            # Sources
            if msg.get("sources") and show_sources:
                st.markdown(
                    " ".join(
                        f'<span class="source-chip">📄 {s.split("/")[-1].split(chr(92))[-1]}</span>'
                        for s in msg["sources"]
                    ), unsafe_allow_html=True)

            # Retrieved chunks expander
            if msg.get("retrieved_chunks") and show_chunks:
                with st.expander(f"🔍 View {len(msg['retrieved_chunks'])} retrieved chunks"):
                    for i, chunk in enumerate(msg["retrieved_chunks"]):
                        src = chunk['source'].split('/')[-1].split('\\')[-1]
                        st.markdown(
                            f"<div class='chunk-meta'>Chunk {i+1} · {src} · page {chunk['page']}</div>"
                            f"<div class='chunk-box'>{chunk['content'][:500]}{'…' if len(chunk['content'])>500 else ''}</div>",
                            unsafe_allow_html=True)

            if msg.get("guardrail_triggered"):
                st.markdown('<span class="guardrail-badge">🛡️ Off-topic blocked</span>',
                            unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# ── Suggestion cards ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("<div style='font-size:0.75rem;color:#555;font-family:DM Mono,monospace;margin-bottom:10px;'>SUGGESTED QUESTIONS</div>", unsafe_allow_html=True)
    suggestions = [
        ("🔄", "What is the Transformer architecture?"),
        ("🎯", "How does BERT handle bidirectional context?"),
        ("📚", "What problem does RAG solve in LLMs?"),
        ("⚡", "Explain the attention mechanism"),
        ("🧮", "What are the key contributions of this paper?"),
        ("🔬", "What datasets were used for evaluation?"),
    ]
    cols = st.columns(3)
    for i, (icon, q) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"{icon} {q}", key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

# ── Process question ──────────────────────────────────────────────────────
user_input = st.chat_input("Ask about your research papers...")

question = user_input or st.session_state.get("pending_question")
if st.session_state.pending_question:
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Build history window — exclude the question we just appended
    all_msgs = st.session_state.messages[:-1]
    # Keep last memory_window exchanges = memory_window * 2 messages
    history_window = all_msgs[-(memory_window * 2):]
    n_exchanges_used = len([m for m in history_window if m["role"] == "user"])

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            result = agent.ask(question, chat_history=history_window)

        placeholder = st.empty()
        displayed = ""
        for char in result["answer"]:
            displayed += char
            placeholder.markdown(
                f"<div class='answer-block'>{displayed}▌</div>",
                unsafe_allow_html=True)
            time.sleep(0.006)
        placeholder.markdown(
            f"<div class='answer-block'>{displayed}</div>",
            unsafe_allow_html=True)

        # Memory badge — only show if we actually had history
        if n_exchanges_used > 0 and not result.get("guardrail_triggered"):
            st.markdown(
                f'<span class="memory-badge">🧠 Remembered {n_exchanges_used} prior exchange(s)</span>',
                unsafe_allow_html=True)

        if result.get("sources") and show_sources and not result.get("guardrail_triggered"):
            st.markdown(
                " ".join(
                    f'<span class="source-chip">📄 {s.split("/")[-1].split(chr(92))[-1]}</span>'
                    for s in result["sources"]
                ), unsafe_allow_html=True)

        if result.get("retrieved_chunks") and show_chunks and not result.get("guardrail_triggered"):
            with st.expander(f"🔍 View {len(result['retrieved_chunks'])} retrieved chunks"):
                for i, chunk in enumerate(result["retrieved_chunks"]):
                    src = chunk['source'].split('/')[-1].split('\\')[-1]
                    st.markdown(
                        f"<div class='chunk-meta'>Chunk {i+1} · {src} · page {chunk['page']}</div>"
                        f"<div class='chunk-box'>{chunk['content'][:500]}{'…' if len(chunk['content'])>500 else ''}</div>",
                        unsafe_allow_html=True)

        if result.get("guardrail_triggered"):
            st.markdown('<span class="guardrail-badge">🛡️ Off-topic blocked</span>',
                        unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "retrieved_chunks": result.get("retrieved_chunks", []),
        "guardrail_triggered": result.get("guardrail_triggered", False),
        "used_memory": n_exchanges_used if n_exchanges_used > 0 else None,
    })