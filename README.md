# LegalMind: AI-Powered Legal Knowledge Assistant

LegalMind is a production-grade RAG (Retrieval-Augmented Generation) system designed for law firms to query thousands of case files and contracts with high factual accuracy, mandatory source citations, and multilingual support (Arabic & English).

## 🚀 Key Features

- **Hybrid Retrieval**: Combines Dense Vector search with Multilingual BM42 Keyword search.
- **Two-Stage Re-ranking**: Uses Cohere and local BGE-v2 Cross-Encoders to surface the most relevant legal context.
- **Semantic Caching**: Redis-backed cache for instant responses to common legal queries.
- **Autonomous Evaluation Agents**:
  - **Adversarial Lawyer**: Generates hard, multi-hop synthetic test cases.
  - **Compliance Auditor**: Hallucination detector using LLM-as-a-judge (Faithfulness).
  - **Shepardizer**: Validates citations for existence, relevance, and attribution.
- **Quality Gates**: CI/CD (GitHub Actions) enforced thresholds (Faithfulness >= 0.9).

---

## 🛠️ Tech Stack

- **Core**: Python 3.11, FastAPI, Streamlit
- **RAG Infrastructure**: Qdrant (Vector DB), Redis (Cache), LlamaIndex
- **Models**: OpenAI (GPT-4o, Embeddings), Cohere (Rerank), FastEmbed (Sparse BM42)
- **Evaluation**: DeepEval, Pytest

---

## 💻 Local Setup

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) (for ultra-fast Python package management)
- Docker & Docker Compose (for Qdrant and Redis)

### 2. Infrastructure
Start the required services:
```bash
docker-compose up -d
```

### 3. Installation
Install dependencies and set up the environment:
```bash
uv pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and COHERE_API_KEY
```

---

## 🏃 Running the Application

### 1. Backend (FastAPI)
```bash
uv run uvicorn main:app --reload
```

### 2. Frontend (Streamlit)
```bash
uv run streamlit run app.py
```

### 3. Ingestion
To ingest the sample legal documents:
```bash
# Ingest Arabic Labor System
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@sample_docs/labor_system.pdf"
```

---

## 🧪 Evaluation & Quality Assurance

### Automated System Benchmarking
Run the live RAG system benchmark against the "Golden Dataset" (15+ hard questions):
```bash
uv run deepeval test run tests/eval/test_rag_benchmark.py
```

### Human-in-the-loop Evaluation CLI
Run the evaluation CLI with a detailed report and LLM-judge scores:
```bash
uv run python golden_dataset/eval_cli.py --mode llm
```

### CI/CD Guardrails
Every Pull Request triggers the `.github/workflows/eval.yml` workflow, which:
1. Spins up Qdrant & Redis services.
2. Boots the RAG backend.
3. Runs the full benchmark.
4. **Fails the build** if Faithfulness falls below **0.9** (Compliance Auditor) or Context Precision falls below **0.8** (Shepardizer).

---

## 📁 Project Structure

- `src/agents/`: Logic for the three specialized evaluation agents.
- `src/retrieval/`: Hybrid search and re-ranking implementations.
- `src/cache/`: Redis semantic cache logic.
- `golden_dataset/`: Ground-truth Q&A pairs and the evaluation CLI.
- `tests/eval/`: Core quality metrics and system benchmark tests.
- `.github/workflows/`: Automated CI/CD pipelines.
