import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

from deepagents import create_deep_agent
from langchain_groq import ChatGroq

from src.agents.qa_agent import QAAgent
from src.agents.summarizer import SummarizerAgent


def build_system():
    docs = load_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    vectordb = build_vectorstore(chunks)
    return vectordb


async def demo_query(query: str):
    # 1. Build vectorstore and retriever
    vectordb = build_system()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # 2. Instantiate your existing agents
    qa_agent = QAAgent(retriever)
    summ_agent = SummarizerAgent()

    # 3. Wrap your agents as DeepAgent tools
    async def qa_tool(query: str, docs: list):
        """Answer a question using retrieved documents."""
        return await qa_agent.run(query)

    async def summarizer_tool(docs: list):
        """Summarize retrieved documents."""
        return await summ_agent.run(docs)

    # 4. Groq LLM (this is CRITICAL — prevents DeepAgents from defaulting to Anthropic)
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0,
    )

    # 5. Create DeepAgent
    deep_agent = create_deep_agent(
        tools=[qa_tool, summarizer_tool],
        system_prompt=(
            "You are a retrieval-augmented deep agent. "
            "Given a query and retrieved documents, use qa_tool(query, docs) "
            "to answer the question. Then optionally call summarizer_tool(docs) "
            "to produce a concise summary."
        ),
        model=llm,
    )

    # 6. Retrieve documents
    docs = await retriever.ainvoke(query)

    # 7. Run DeepAgent with the context + query
    result = await deep_agent.ainvoke({
        "query": query,
        "docs": docs
    })

    print("\n==================== FINAL OUTPUT ====================")
    print(result["messages"][-1]["content"])
    print("======================================================\n")


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
