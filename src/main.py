import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

from langchain_groq import ChatGroq
from deepagents import DeepAgent 


# ---------------------------
# Build RAG system
# ---------------------------
def build_system():
    docs = load_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vectordb = build_vectorstore(chunks)
    return vectordb


# ---------------------------
# Use DeepAgent as a SIMPLE LLM layer
# ---------------------------
async def demo_query(query: str):
    vectordb = build_system()

    # Retrieval
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs[:3])

    # Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_api_key,
    )

    # DeepAgent (as plain LLM caller)
    agent = DeepAgent(model=llm)

    # Create final prompt manually
    prompt = (
        "You are a helpful RAG assistant.\n"
        "Use ONLY the context below to answer.\n"
        "If the answer is not in the context, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    # Call DeepAgent without tools
    result = await agent.ainvoke(prompt)

    print("\n--- ANSWER ---\n")
    print(result["output"])


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    import sys
    import os

    if not sys.stdin.isatty():
        query = os.environ.get("DEMO_QUERY", "What is the main point of the documents?")
    else:
        query = input("\n🔍 Enter your query: ").strip() or "What is the main point of the documents?"

    asyncio.run(demo_query(query))
