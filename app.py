"""
app.py — Fixed suggestion buttons + fast startup
"""
import os
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Student Learning Agent", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.source-chip {
    display: inline-block; background: #1e3a5f; color: #7eb8f7;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem; margin: 2px;
}
.guardrail-badge {
    background: #5c1a1a; color: #ff7f7f;
    border-radius: 8px; padding: 4px 12px; font-size: 0.8rem;
}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🎓 Learning Agent")
    st.markdown("**Groq LLaMA3.1 + RAG + LangChain**")
    st.divider()
    top_k = st.slider("Retrieved chunks (top-k)", 2, 8, 4)
    show_sources = st.toggle("Show source documents", value=True)
    st.divider()
    st.markdown("""
**Stack:**
- 🔍 ChromaDB vector search
- 🤖 Groq LLaMA-3.1-8b (free)
- 📦 HuggingFace embeddings (local)
- 🛡️ Topic guardrails
- 📊 RAGAS evaluation
""")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("📊 Run RAGAS Eval"):
        with st.spinner("Running evaluation (~2 min)..."):
            try:
                from evaluate import run_evaluation
                scores = run_evaluation()
                st.success("Done!")
                for k, v in scores.items():
                    st.metric(k.replace("_", " ").title(), f"{v:.3f}")
            except Exception as e:
                st.error(f"Error: {e}")

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
    st.code("GROQ_API_KEY=gsk_your_key_here")
    st.stop()

if not os.path.exists("chroma_db"):
    st.warning("⚠️ No vector store found. Running ingestion...")
    with st.spinner("Ingesting papers..."):
        from ingest import ingest_papers
        ingest_papers()
    st.success("✅ Done!")

@st.cache_resource(show_spinner="⏳ Loading model (only happens once)...")
def load_agent(k: int):
    from rag_chain import build_rag_chain
    from guardrails_wrapper import GuardrailedAgent
    return GuardrailedAgent(build_rag_chain(top_k=k))

agent = load_agent(top_k)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Key fix: track a pending question from button clicks ─────────────────
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

st.title("🎓 Student Learning Agent")
st.caption("Ask questions about the research papers in the knowledge base.")
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            st.markdown(
                "**Sources:** " + " ".join(
                    f'<span class="source-chip">📄 {s.split("/")[-1]}</span>'
                    for s in msg["sources"]
                ), unsafe_allow_html=True)
        if msg.get("guardrail_triggered"):
            st.markdown('<span class="guardrail-badge">🛡️ Guardrail triggered</span>',
                        unsafe_allow_html=True)

# ── Suggestion buttons (only show when chat is empty) ─────────────────────
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    suggestions = [
        "What is the Transformer architecture?",
        "How does BERT handle bidirectional context?",
        "What problem does RAG solve?",
        "Explain the attention mechanism",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}"):
            # Store as pending — will be processed below just like chat input
            st.session_state.pending_question = s
            st.rerun()

# ── Resolve input: either from chat box or suggestion button ──────────────
user_input = st.chat_input("Ask about the research papers...")

question_to_process = None
if user_input:
    question_to_process = user_input
elif st.session_state.pending_question:
    question_to_process = st.session_state.pending_question
    st.session_state.pending_question = None  # clear it

# ── Process the question ──────────────────────────────────────────────────
if question_to_process:
    st.session_state.messages.append({"role": "user", "content": question_to_process})

    with st.chat_message("user"):
        st.markdown(question_to_process)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.ask(question_to_process)

        placeholder = st.empty()
        displayed = ""
        for char in result["answer"]:
            displayed += char
            placeholder.markdown(displayed + "▌")
            time.sleep(0.006)
        placeholder.markdown(displayed)

        if result.get("sources") and show_sources:
            st.markdown(
                "**Sources:** " + " ".join(
                    f'<span class="source-chip">📄 {s.split("/")[-1]}</span>'
                    for s in result["sources"]
                ), unsafe_allow_html=True)
        if result.get("guardrail_triggered"):
            st.markdown(
                '<span class="guardrail-badge">🛡️ Guardrail triggered — off-topic blocked</span>',
                unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "guardrail_triggered": result.get("guardrail_triggered", False),
    })