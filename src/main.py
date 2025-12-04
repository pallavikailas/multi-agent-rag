import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

from src.graph.rag_graph import build_graph


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

    graph = build_graph()

    state = {
        "query": query,
        "retriever": retriever
    }

    out = graph.invoke(state)

    print("--- Summary ---")
    print(out.get("summary"))
    print("--- Answer ---")
    print(out.get("answer"))


if __name__ == "__main__":
    import os
    import sys
    import asyncio

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
