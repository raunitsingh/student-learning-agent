"""
ingest.py - Run once to embed PDFs into ChromaDB
    python ingest.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PAPERS_DIR = Path("papers")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    print("📦 Loading HuggingFace embedding model (first run downloads ~90MB)...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_papers():
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("⚠️  No PDFs found in /papers. Using sample placeholder document...")
        from langchain.schema import Document
        chunks = [
            Document(
                page_content=(
                    "Transformer architecture introduced in 'Attention is All You Need' "
                    "uses self-attention mechanisms to process sequences in parallel. "
                    "BERT uses bidirectional transformers for pre-training deep language representations. "
                    "GPT models use autoregressive language modeling for generation tasks. "
                    "RAG (Retrieval Augmented Generation) combines retrieval with generation "
                    "to produce factual, grounded answers from external knowledge sources. "
                    "Large Language Models are trained on massive text corpora and can "
                    "perform NLP tasks including QA, summarization, and translation."
                ),
                metadata={"source": "sample_placeholder", "page": 0},
            )
        ]
    else:
        print(f"📄 Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
        all_docs = []
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            all_docs.extend(pages)
            print(f"   Loaded: {pdf_path.name} ({len(pages)} pages)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "],
        )
        chunks = splitter.split_documents(all_docs)
        print(f"✂️  Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    print("🔢 Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="research_papers",
    )

    print(f"✅ Done! {vectorstore._collection.count()} chunks stored in '{CHROMA_DIR}/'")
    return vectorstore


if __name__ == "__main__":
    ingest_papers()