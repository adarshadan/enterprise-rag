# Enterprise RAG API

A high-performance, containerized Enterprise Retrieval-Augmented Generation (RAG) API built with **FastAPI**, **LangChain**, and **Docker**. This system uses a hybrid retrieval pipeline combining semantic vector search via **ChromaDB** and keyword search via **BM25**, backed by **Groq Cloud LLMs** for ultra-fast generation times. 

To achieve maximum reliability and performance in containerized or network-restricted environments, the embedding model weights are fully baked into the Docker image, enabling **100% offline embedding execution**.

---

## 🚀 Features

- **Hybrid Retrieval:** Blends semantic vector embeddings with keyword-based retrieval (BM25) for high accuracy.
- **Section Reconstruction:** Intellectually reconstructs document sections from fragmented PDF chunks to preserve contextual coherence.
- **Offline Embeddings:** Docker image downloads and packages `all-mpnet-base-v2` at build time, eliminating runtime DNS/network dependencies.
- **FastAPI Core:** Provides an asynchronous, high-throughput REST API with comprehensive runtime performance telemetry (retrieval and generation metrics).
- **Blazing Fast LLM Inference:** Powered by Groq Cloud's LPU infrastructure via `langchain-groq`.

---

## 🛠️ Tech Stack

- **API Framework:** FastAPI & Uvicorn
- **Orchestration:** LangChain
- **LLM Provider:** Groq (via `ChatGroq`)
- **Embedding Model:** `sentence-transformers/all-mpnet-base-v2` (via `HuggingFaceEmbeddings`)
- **Vector Database:** ChromaDB
- **Keyword Index:** BM25

---

## 📋 Prerequisites

Before running the application, make sure you have installed:
- [Docker](https://www.docker.com/) & Docker Compose
- A free **Groq API Key** (Get one from the [Groq Console](https://console.groq.com/))

---

## 🔧 Configuration (.env)

Create a `.env` file in the root directory of the project to set up your environment variables:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
HF_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

📦 Getting Started (Docker Deployment)
Spin up the entire application stack using Docker Compose. The build process will automatically bake the embedding model weights directly into the container storage.

Bash
# Build and start the container
docker-compose up --build
The server will automatically start on http://localhost:8000. You can view the interactive API documentation (Swagger UI) at http://localhost:8000/docs.

🔌 API Endpoints
1. Health Check
Verify the service is up and healthy.

Method: GET

URL: /health

Response:

JSON
{
  "status": "healthy"
}
2. Load Documents
Ingest a PDF document via JSON body.

Method: POST
URL: /load
Headers: Content-Type: application/json

Request Body:
{
  "file_path": "/app/data/indian-penal-code.pdf",
  "collection": "test"
}

Response:

JSON
{
  "status": "loaded",
  "chunks": 353
}
3. Query
Submit a question to get a contextual answer, extracted metadata sources, and system execution performance metrics.

Method: POST

URL: /query

Headers: Content-Type: application/json

Request Body:

JSON
{
  "query": "What is the punishment for theft according to Section 379?",
  "collection": "default",
  "top_k": 5
}
Response:

JSON
{
  "answer": "According to Section 379, the punishment for theft is imprisonment of either description for a term which may extend to three years, or with fine, or with both.",
  "sources": ["/app/data/indian-penal-code.pdf"],
  "confidence": 0.9,
  "retrieval_time_ms": 14.5,
  "generation_time_ms": 230.1,
  "cache_hit": false
}
📁 Project Directory Structure
Plaintext
enterprise-rag/

├── app/
│   ├── main.py              # FastAPI endpoints
│   └── rag_engine.py        # Core RAG logic
│   └── locustfile.py        # Load testingc
├── tests/
│   ├── init.py
│   └── test_health.py       # Unit tests
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD
├── data/                    # PDFs and vector stores
├── Dockerfile               # Docker image with embedded models
├── docker-compose.yml       # Local development stack
├── evals.py                 # Evaluation metrics
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .env                     # Configuration (not in git)