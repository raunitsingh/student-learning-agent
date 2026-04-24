# 🎓 LLM-Powered Student Learning Agent

AI chatbot that answers questions from research papers using RAG, LangChain, NeMo Guardrails, and RAGAS evaluation.

## Stack
| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Framework | LangChain |
| Vector DB | ChromaDB |
| Guardrails | NVIDIA NeMo Guardrails |
| Evaluation | RAGAS |
| UI | Streamlit |

---

## Setup

### 1. Clone & install
```bash
git clone <your-repo>
cd student-learning-agent
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Add your research papers
Drop any `.pdf` research papers into the `papers/` folder.
> If no PDFs are added, a sample placeholder document is used automatically.

### 4. Ingest papers into ChromaDB
```bash
python ingest.py
```

### 5. Run the chatbot
```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Run RAGAS Evaluation
```bash
python evaluate.py
```
Outputs metrics to console and saves `evaluation_results.csv`.

Metrics evaluated:
- **Faithfulness** — Is the answer grounded in retrieved context?
- **Answer Relevancy** — Does it actually answer the question?
- **Context Recall** — Does retrieved context cover the ground truth?
- **Context Precision** — Is retrieved context relevant, not noisy?

---

## Project Structure
```
student-learning-agent/
├── app.py                  # Streamlit UI
├── rag_chain.py            # RAG pipeline (LangChain + ChromaDB)
├── ingest.py               # PDF ingestion & embedding
├── guardrails_wrapper.py   # NeMo Guardrails integration
├── evaluate.py             # RAGAS evaluation
├── guardrails/
│   ├── config.yml          # NeMo config
│   └── main.co             # Colang topic-compliance rules
├── papers/                 # Drop your PDFs here
├── requirements.txt
└── .env.example
```

---

## How It Works

```
User Question
     │
     ▼
[NeMo Guardrails]  ──off-topic──►  "I can only help with research topics"
     │ on-topic
     ▼
[ChromaDB Retriever]  ──► top-k relevant chunks from PDFs
     │
     ▼
[LangChain RetrievalQA]  ──► GPT-4o-mini generates answer
     │
     ▼
Structured Answer + Source citations
```

---

## Bullet Points
- Engineered RAG pipeline with LangChain + ChromaDB; retrieves top-k semantically relevant chunks from research PDFs using MMR search
- Integrated NVIDIA NeMo Guardrails with Colang topic-compliance rules to restrict responses to academic domains
- Implemented RAGAS evaluation suite measuring Faithfulness, Answer Relevancy, Context Recall, and Context Precision
- Built Streamlit UI with streaming responses, source attribution, and guardrail status indicators
