"""
rag_chain.py — With paper scoping, confidence scoring, memory
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

CONFIDENCE_PROMPT = """You are evaluating whether an AI answer is grounded in the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Rate how well the answer is supported by the context on a scale of 1 to 5:
1 = Not supported at all / likely hallucinated
2 = Weakly supported
3 = Partially supported
4 = Mostly supported
5 = Fully supported by context

Reply with ONLY a single integer (1, 2, 3, 4, or 5). Nothing else."""

_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model (one-time)...")
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


def format_chat_history(history: list) -> str:
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "Student" if msg["role"] == "user" else "Tutor"
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
    return {"retriever": retriever, "llm": llm, "top_k": top_k}


def get_confidence_score(llm, context: str, question: str, answer: str) -> int:
    """Calls LLM to rate how well the answer is grounded in context. Returns 1-5."""
    try:
        prompt = CONFIDENCE_PROMPT.format(
            context=context[:2000],  # truncate to keep it cheap
            question=question,
            answer=answer[:800],
        )
        response = llm.invoke(prompt)
        score = int(response.content.strip()[0])
        return max(1, min(5, score))  # clamp to 1-5
    except Exception:
        return 0  # 0 = could not compute


def get_answer(
    chain,
    question: str,
    chat_history: list = None,
    scoped_papers: list = None,   # list of filenames to restrict retrieval to
    compute_confidence: bool = False,
) -> dict:
    if chat_history is None:
        chat_history = []

    recent_history = chat_history[-10:]
    vectorstore = get_vectorstore()

    # ── Feature 3: Paper scoping via metadata filter ──────────────────────
    if scoped_papers and len(scoped_papers) > 0:
        # ChromaDB where filter — match any of the selected filenames
        # Sources are stored as full paths, so we do a broad fetch then filter
        all_docs = vectorstore.similarity_search(question, k=chain["top_k"] * 4)
        docs = [
            d for d in all_docs
            if any(
                sp in d.metadata.get("source", "")
                for sp in scoped_papers
            )
        ]
        # Fallback: if filter killed everything, use unfiltered
        if not docs:
            docs = all_docs[:chain["top_k"]]
        else:
            docs = docs[:chain["top_k"]]
    else:
        docs = chain["retriever"].invoke(question)

    context = format_docs(docs)
    history_text = format_chat_history(recent_history)

    prompt_text = STUDENT_PROMPT.format(
        context=context,
        chat_history=history_text,
        question=question,
    )

    llm = chain["llm"]
    response = llm.invoke(prompt_text)
    answer = response.content

    # ── Feature 4: Confidence score ───────────────────────────────────────
    confidence = None
    if compute_confidence:
        confidence = get_confidence_score(llm, context, question, answer)

    sources = list({doc.metadata.get("source", "unknown") for doc in docs})

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "retrieved_chunks": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
            }
            for doc in docs
        ],
    }