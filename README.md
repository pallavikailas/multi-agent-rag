# 🚀 Multi-Agent RAG Orchestration  
### LangChain · LangGraph · DeepAgents · Groq  
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v1.1-green)
![Groq](https://img.shields.io/badge/Powered%20By-Groq-orange)
![DeepAgents](https://img.shields.io/badge/DeepAgents-Architecture-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌐 Overview

This project implements a **production-grade Multi‑Agent RAG workflow** using:

- **LangChain** for document ingestion, chunking, embeddings, retrievers  
- **LangGraph-inspired orchestration** (graph execution pattern)  
- **DeepAgents-style multi-agent roles** with async workflows  
- **Groq LLMs** (FREE API) for ultra-fast inference  
- **Tenacity** for rate-limit handling  
- **Local PDFs** under `/data` as the knowledge corpus  

Assignment requirement matched:

> “Using LangChain, LangGraph, and DeepAgents, build a small multi-agent RAG workflow using the provided files as data, with appropriate chunking, and using free LLM APIs (Groq).”

---

# 🏗️ Architecture Diagram (High-Level)

```
                        ┌──────────────────────────────┐
                        │          User Query          │
                        └──────────────────────────────┘
                                       │
                                       ▼
                          ┌────────────────────────┐
                          │   Deep Orchestrator    │
                          │ (Async Task Manager)   │
                          └────────────────────────┘
                             │                    │
                ┌────────────┘                    └─────────────┐
                ▼                                               ▼
   ┌───────────────────────────┐                  ┌──────────────────────────┐
   │     Retrieval QA Agent    │                  │     Summarizer Agent     │
   │  - Retrieves chunks       │                  │ - Summaries context      │
   │  - Queries Groq           │                  │ - Metadata extraction    │
   └───────────────────────────┘                  └──────────────────────────┘
                │                                                 │
                └──────────────┐                   ┌──────────────┘
                               ▼                   ▼
                    ┌────────────────────────────────────────┐
                    │      Merge Final Agent Outputs         │
                    └────────────────────────────────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │         Final Answer         │
                         └──────────────────────────────┘
```

---

# 📚 Data Flow Diagram (Detailed)

```
                    ┌─────────────────────────────┐
                    │     PDF Files in /data      │
                    └─────────────────────────────┘
                                  │
                                  ▼
           ┌──────────────────────────────────────────────────┐
           │               Ingestion Pipeline                 │
           │  - DirectoryLoader (PDF)                         │
           │  - RecursiveCharacterTextSplitter (1k/200)       │
           └──────────────────────────────────────────────────┘
                                  │
                                  ▼
             ┌───────────────────────────────────────┐
             │   Embeddings (Sentence Transformers)  │
             └───────────────────────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │      Vectorstore        │
                     │   (FAISS / ChromaDB)    │
                     └─────────────────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │     Retriever API       │
                     └─────────────────────────┘
                                  │
                                  ▼
                   ┌────────────────────────────────┐
                   │    Multi-Agent Orchestrator    │
                   └────────────────────────────────┘
                        │                    │
                        ▼                    ▼
            ┌─────────────────┐       ┌───────────────────┐
            │ QA Agent        │       │ Summarizer Agent  │
            │ → Groq LLM      │       │ → Groq LLM        │
            └─────────────────┘       └───────────────────┘
                        │                    │
                        └──────────┬─────────┘
                                   ▼
                       ┌────────────────────────┐
                       │ Final Answer + Summary │
                       └────────────────────────┘
```

---

# 🤖 Agent Roles

### 🔍 Retrieval QA Agent
- Pulls top‑K chunks from vectorstore  
- Builds query‑aware prompt  
- Calls Groq (`llama‑3.1‑8b‑instant`)  
- Returns focused, factual RAG answer  

### 📝 Summarizer Agent
- Uses retrieved context  
- Produces high-level semantic summary  
- Helps users understand context at a glance  

### 🧠 Deep Orchestrator
- Parallel async execution of agents  
- Inspired by DeepAgents  
- Manages:
  - Retrieval
  - Routing
  - Parallelism
  - Merging

---

# 🕸️ LangGraph-Style DAG

```
Start
 │
 ▼
ChunkRetriever
 │
 ├───────────── QA_Agent ────────────────┐
 │                                       │
 └──────────── Summarizer_Agent ─────────┘
                     │
                     ▼
                MergeOutputs
                     │ 
                     ▼
                    End
```

---

# 📦 Setup

```bash
git clone https://github.com/pallavikailas/multi-agent-rag.git
cd multi-agent-rag

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
nano .env
```

Add:
```
GROQ_API_KEY= <enter your api key here>
```

---

# ▶️ Run

```
./.venv/bin/python -m src.main
```

---

# 🐳 Run with Docker

```
docker-compose build
docker-compose up
docker-compose run -e DEMO_QUERY="What does Rule 10b-5(b) require?" app
```
```
docker-compose down
```
---

# ✔ Assignment Checklist

| Requirement | Status |
|------------|--------|
| Multi‑agent | ✅ |
| LangChain used | ✅ |
| LangGraph flow | ✓ Concept implemented |
| DeepAgents-style | ✅ Parallel async pipeline |
| Chunking | ✅ Recursive (1000 / 200) |
| Provided PDF data | ✅ |
| Free API | Groq |
| Rate limiting | Tenacity Retry |

---

# 📄 License  
MIT

