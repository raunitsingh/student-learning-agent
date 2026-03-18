"""
guardrails_wrapper.py — topic guardrails, Groq-compatible
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
    "joke", "recipe", "stock price", "relationship", "poem",
    "who won", "game", "weather", "movie", "music", "song",
    "write my essay", "do my homework", "girlfriend", "boyfriend",
]

ACADEMIC_KEYWORDS = [
    "paper", "research", "model", "algorithm", "neural", "learning",
    "transformer", "attention", "bert", "gpt", "dataset", "accuracy",
    "training", "embedding", "nlp", "classification", "regression",
    "gradient", "loss", "metric", "rag", "retrieval", "vector",
    "explain", "what is", "how does", "define", "summarize", "llm",
]


def is_off_topic(question: str) -> bool:
    q = question.lower()
    if any(kw in q for kw in OFF_TOPIC_KEYWORDS):
        return True
    if len(q.split()) > 5 and not any(kw in q for kw in ACADEMIC_KEYWORDS):
        return True
    return False


class GuardrailedAgent:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.rails = None

        if NEMO_AVAILABLE:
            try:
                config = RailsConfig.from_path("guardrails/")
                self.rails = LLMRails(config)
                print("NeMo Guardrails loaded.")
            except Exception as e:
                print(f"NeMo config error: {e}. Using keyword fallback.")
        else:
            print("nemoguardrails not installed — using keyword guardrails.")

    def ask(self, question: str) -> dict:
        if is_off_topic(question):
            return {
                "answer": (
                    "I'm designed to help with research papers and academic topics. "
                    "Please ask me something related to the papers you're studying!"
                ),
                "sources": [],
                "guardrail_triggered": True,
            }

        from rag_chain import get_answer
        result = get_answer(self.rag_chain, question)
        result["guardrail_triggered"] = False
        return result
