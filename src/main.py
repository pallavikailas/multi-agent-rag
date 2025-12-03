import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .orchestrator import DeepOrchestrator
from .config import settings

def build_system():
    docs = load_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    vectordb = build_vectorstore(chunks)
    return vectordb

async def demo_query(query: str):
    vectordb = build_system()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    orchestrator = DeepOrchestrator(retriever)
    out = await orchestrator.handle_query(query)
    print("--- Summary ---")
    print(out['summary'])
    print("--- Answer ---")
    print(out['answer'])

if __name__ == '__main__':
    import os
    # 1. If DEMO_QUERY env var is set → use it
    env_query = os.environ.get("DEMO_QUERY")

    # 2. Otherwise, ask the user
    if env_query:
        query = env_query
    else:
        query = input("\n🔍 Enter your query: ")

        # fallback if the user pressed Enter without typing anything
        if not query.strip():
            query = "What is the main point of the documents?"

    print(f"\n👉 Running multi-agent RAG on query:\n{query}\n")
    
    asyncio.run(demo_query(query))
