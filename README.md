# Multi-Agent RAG — Groq-only

A focused scaffold for a Retrieval-Augmented Generation pipeline that uses only Groq for model inference, LangChain for document handling and vectorstore integrations, an illustrative LangGraph spec, and a DeepAgents-inspired orchestrator.

Requirements: create a Groq account and API key.

Run:

1. Copy `.env.example` to `.env` and set GROQ_API_KEY
2. `pip install -r requirements.txt`
3. `python src/main.py`

Optional: build the image with Docker.

