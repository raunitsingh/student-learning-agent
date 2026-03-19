"""
rag_chain.py — With conversation memory (last 5 exchanges)
"""
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Prompt now includes chat history ─────────────────────────────────────
STUDENT_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are an expert academic tutor helping students understand research papers.
Use ONLY the context below to answer the question. If the answer is not in the context,
say "I don't have enough information in the provided papers to answer this."

Do NOT make up information. Be clear, structured, and educational.
If relevant, break your answer into steps or bullet points.

IMPORTANT: Use the chat history to understand follow-up questions and references
like "it", "that", "the model", "the paper" — resolve them from prior context.

Context from research papers:
{context}

Chat History:
{chat_history}

Student Question: {question}

Detailed Answer:""",
)

# ── Module-level cache ────────────────────────────────────────────────────
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("📦 Loading embedding model (one-time)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
            collection_name="research_papers",
        )
    return _vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history: list[dict]) -> str:
    """
    Takes last N messages from st.session_state.messages format:
    [{"role": "user"|"assistant", "content": str}, ...]
    Returns a plain-text block for the prompt.
    """
    if not history:
        return "No previous conversation."

    lines = []
    for msg in history:
        role = "Student" if msg["role"] == "user" else "Tutor"
        # Truncate very long assistant answers to keep prompt lean
        content = msg["content"]
        if msg["role"] == "assistant" and len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rag_chain(top_k: int = 4):
    retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": 10},
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=1024,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    return {"retriever": retriever, "llm": llm}


def get_answer(chain, question: str, chat_history: list[dict] = None) -> dict:
    """
    Args:
        chain: built by build_rag_chain()
        question: current user question
        chat_history: last N messages from session_state (excluding current question)
                      format: [{"role": "user"|"assistant", "content": str}]
    """
    if chat_history is None:
        chat_history = []

    # Keep only last 5 exchanges (10 messages) to avoid prompt bloat
    recent_history = chat_history[-10:]

    docs = chain["retriever"].invoke(question)
    context = format_docs(docs)
    history_text = format_chat_history(recent_history)

    prompt_text = STUDENT_PROMPT.format(
        context=context,
        chat_history=history_text,
        question=question,
    )

    response = chain["llm"].invoke(prompt_text)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_chunks": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
            }
            for doc in docs
        ],
    }