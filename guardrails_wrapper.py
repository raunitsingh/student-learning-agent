"""
guardrails_wrapper.py — passes scoped_papers + confidence flag through
"""
import os
from dotenv import load_dotenv
load_dotenv()

try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

OFF_TOPIC_KEYWORDS = [
    "joke", "recipe", "stock price", "relationship advice", "poem",
    "who won", "play a game", "weather", "movie review", "write my essay",
    "do my homework", "girlfriend", "boyfriend", "cooking",
]

ACADEMIC_KEYWORDS = [
    "paper", "research", "model", "algorithm", "neural", "learning",
    "transformer", "attention", "bert", "gpt", "dataset", "accuracy",
    "training", "embedding", "nlp", "classification", "regression",
    "gradient", "loss", "metric", "rag", "retrieval", "vector",
    "explain", "what is", "how does", "define", "summarize", "llm",
    "architecture", "method", "result", "experiment", "baseline",
    "performance", "benchmark", "fine-tun", "pre-train", "inference",
    "layer", "weight", "parameter", "encode", "decode", "token",
    "it", "that", "this", "they", "the paper", "the model", "what about",
    "and", "also", "tell me more", "elaborate", "explain more", "why",
    "how", "what", "difference", "compare", "versus", "vs",
]


def is_off_topic(question: str) -> bool:
    q = question.lower().strip()
    if len(q.split()) <= 4:
        return False
    if any(kw in q for kw in OFF_TOPIC_KEYWORDS):
        return True
    if len(q.split()) > 6 and not any(kw in q for kw in ACADEMIC_KEYWORDS):
        return True
    return False


class GuardrailedAgent:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain

        if NEMO_AVAILABLE:
            try:
                from nemoguardrails import RailsConfig, LLMRails
                config = RailsConfig.from_path("guardrails/")
                self.rails = LLMRails(config)
                print("NeMo Guardrails loaded.")
            except Exception as e:
                print(f"NeMo config error: {e}. Using keyword fallback.")
        else:
            print("Using keyword topic filter.")

    def ask(
        self,
        question: str,
        chat_history: list = None,
        scoped_papers: list = None,
        compute_confidence: bool = False,
    ) -> dict:
        if chat_history is None:
            chat_history = []

        if is_off_topic(question):
            return {
                "answer": (
                    "I'm designed to help with research papers and academic topics. "
                    "Please ask me something related to the papers you're studying!"
                ),
                "sources": [],
                "retrieved_chunks": [],
                "confidence": None,
                "guardrail_triggered": True,
            }

        from rag_chain import get_answer
        result = get_answer(
            self.rag_chain,
            question,
            chat_history=chat_history,
            scoped_papers=scoped_papers,
            compute_confidence=compute_confidence,
        )
        result["guardrail_triggered"] = False
        return result