"""
evaluate.py — RAGAS evaluation using Groq + HuggingFace embeddings
Run: python evaluate.py
"""
import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from rag_chain import build_rag_chain, get_answer, get_embeddings

load_dotenv()

EVAL_DATASET = [
    {
        "question": "What is the main idea behind the Transformer architecture?",
        "ground_truth": (
            "The Transformer architecture relies entirely on self-attention mechanisms "
            "to process sequences in parallel, eliminating recurrence and convolutions."
        ),
    },
    {
        "question": "How does BERT use bidirectional context?",
        "ground_truth": (
            "BERT uses a masked language model objective to pre-train deep bidirectional "
            "representations by conditioning on both left and right context simultaneously."
        ),
    },
    {
        "question": "What problem does RAG solve in language models?",
        "ground_truth": (
            "RAG addresses hallucination and knowledge staleness in LLMs by retrieving "
            "relevant documents at inference time and conditioning generation on them."
        ),
    },
    {
        "question": "What is the attention mechanism in transformers?",
        "ground_truth": (
            "Attention computes a weighted sum of value vectors, where weights are "
            "determined by the compatibility between query and key vectors."
        ),
    },
]


def run_evaluation():
    print("Building RAG chain...")
    chain = build_rag_chain(top_k=4)

    from langchain_chroma import Chroma
    emb = get_embeddings()

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"Running {len(EVAL_DATASET)} evaluation queries...\n")
    for i, item in enumerate(EVAL_DATASET):
        q, gt = item["question"], item["ground_truth"]
        print(f"  [{i+1}/{len(EVAL_DATASET)}] {q[:65]}...")

        result = get_answer(chain, q)
        answers.append(result["answer"])

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=emb,
            collection_name="research_papers",
        )
        docs = vectorstore.similarity_search(q, k=4)
        contexts.append([doc.page_content for doc in docs])

        questions.append(q)
        ground_truths.append(gt)

    eval_data = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Wire Groq + HuggingFace into RAGAS
    groq_llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    hf_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    ragas_llm = LangchainLLMWrapper(groq_llm)
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    print("\nComputing RAGAS metrics...")
    results = evaluate(eval_data, metrics=metrics)

    df = results.to_pandas()
    summary = {
        "faithfulness":      df["faithfulness"].mean(),
        "answer_relevancy":  df["answer_relevancy"].mean(),
        "context_recall":    df["context_recall"].mean(),
        "context_precision": df["context_precision"].mean(),
    }

    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)
    for metric, score in summary.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.3f}  {bar}")
    print("=" * 55)
    print(f"  Overall: {sum(summary.values())/len(summary):.3f}")

    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved to evaluation_results.csv")
    return summary


if __name__ == "__main__":
    run_evaluation()
