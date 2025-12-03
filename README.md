# 🚀 Multi-Agent RAG Orchestration (LangChain + LangGraph + DeepAgents + Groq)

This project implements a **production-grade Multi-Agent Retrieval-Augmented Generation (RAG) system** using:

- **LangChain** (loaders, chunking, embeddings, vectorstore, retriever)
- **LangGraph** (graph-based orchestration pattern)
- **DeepAgents**-style multi-agent collaboration
- **Groq LLMs** (ultra-fast open-weight inference)
- **Tenacity-based retry** for rate-limit handling
- **FAISS/Chroma** vectorstore
- **Local PDFs** as the knowledge base

The goal matches the assignment prompt:

> **Build a small Multi-Agent RAG workflow using LangChain, LangGraph, DeepAgents concepts & clean production code, using the provided files as data. Use appropriate chunking and free API providers (Groq).**

This implementation uses **your local PDFs** in `/data` as the knowledge corpus.

---

# 🧠 System Architecture

[Architecture diagram included in ChatGPT response above]

---

# 🤖 Agent Roles

### **1. QA Retrieval Agent**
- Retrieves top relevant chunks
- Builds contextual QA prompt
- Queries Groq model (`llama-3.1-8b-instant`)
- Handles rate-limiting via Tenacity
- Produces grounded answers

### **2. Summarizer Agent**
- Uses retrieved chunks to generate summaries
- Provides high-level understanding of context

### **3. Deep Orchestrator**
- Runs agents in **parallel** (async)
- Merges their outputs
- Follows DeepAgents-style design

---

# 🕸️ LangGraph Flow (Conceptual)

```
Start
 │
 ▼
RetrieveRelevantChunks
 │
 ├──────────────┐
 ▼              ▼
Summarize      AnswerQuestion
 │              │
 └───────┬──────┘
         ▼
      MergeNodes
         ▼
        End
```

---

# 📂 File Structure

```
multi-agent-rag/
├── data/
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   ├── agents/
│   │   ├── qa_agent.py
│   │   └── summarizer.py
│   ├── orchestrator.py
│   ├── utils.py
│   ├── config.py
│   └── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── langgraph.yaml
├── .env.example
└── README.md
```

---

# 📦 Chunking Methodology

- Recursive character text splitting  
- Chunk size: **1000**, overlap: **200**  
- Ensures semantic continuity and high recall retrieval

---

# ⚡ Groq Model (Free Tier)

Uses:
- `llama-3.1-8b-instant`  
- Groq's OpenAI-compatible endpoint  
- Tenacity retry handling  

---

# 🛠️ Setup

```
git clone https://github.com/pallavikailas/multi-agent-rag.git
cd multi-agent-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
```

Add your `GROQ_API_KEY=gsk_xxxxxx`

---

# ▶️ Run the Pipeline

```
./.venv/bin/python -m src.main
```

---

# 🐳 Docker

```
docker-compose build
docker-compose up
```

---

# ✔️ Assignment Checklist

| Requirement | Status |
|------------|--------|
| Multi-agent system | ✅ |
| LangChain | ✅ |
| LangGraph concepts | ✓ Graph design + flow |
| DeepAgents design | ✅ |
| Chunking | Recursive splitter |
| Use provided files | Yes |
| Free API (Groq) | Yes |
| Rate limit handling | Tenacity |

---
