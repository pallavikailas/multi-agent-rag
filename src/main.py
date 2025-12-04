import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .graph.rag_graph import compiled_rag_graph
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
    state = {
        "query": query,
        "context": [d.page_content for d in retriever._get_relevant_documents(query, run_manager=None)],
        "qa_output": None,
        "summary": None
    }

    out = compiled_rag_graph.invoke(state)

    print("--- Summary ---")
    print(out['summary'])
    print("--- Answer ---")
    print(out['answer'])

if __name__ == "__main__":
    import os
    import sys
    import asyncio

    # If running inside Docker (no interactive terminal)
    if not sys.stdin.isatty():
        query = os.environ.get("DEMO_QUERY", "What is the main point of the documents?")
        print(f"\n[Docker Mode] Using query: {query}\n")
    else:
        env_query = os.environ.get("DEMO_QUERY")
        if env_query:
            query = env_query
        else:
            query = input("\n🔍 Enter your query: ").strip()
            if not query:
                query = "What is the main point of the documents?"

    asyncio.run(demo_query(query))
