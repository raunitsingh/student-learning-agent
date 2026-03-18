"""
rag_chain.py — Fast version: embeddings cached at module level
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

STUDENT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert academic tutor helping students understand research papers.
Use ONLY the context below to answer the question. If the answer is not in the context,
say "I don't have enough information in the provided papers to answer this."

Do NOT make up information. Be clear, structured, and educational.
If relevant, break your answer into steps or bullet points.

Context from research papers:
{context}

Student Question: {question}

Detailed Answer:""",
)

# ── Cached at module level — loaded once, reused forever ─────────────────
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


def get_answer(chain, question: str) -> dict:
    docs = chain["retriever"].invoke(question)
    context = format_docs(docs)
    prompt_text = STUDENT_PROMPT.format(context=context, question=question)
    response = chain["llm"].invoke(prompt_text)
    sources = list({doc.metadata.get("source", "unknown") for doc in docs})
    return {"answer": response.content, "sources": sources}