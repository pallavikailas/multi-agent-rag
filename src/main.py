import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

from deepagents import create_deep_agent
from src.agents.qa_agent import QAAAgent
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
    qa_agent = QAAAgent(retriever)
    summ_agent = SummarizerAgent()

    # 3. Wrap your agents as deep agent tools
    async def qa_tool(query: str, docs: list):
        """Use your QA agent with context docs."""
        return await qa_agent.run(query)

    async def summarizer_tool(docs: list):
        """Use summarizer on retrieved docs."""
        return await summ_agent.run(docs)

    # 4. Create a DeepAgent with both tools
    deep_agent = create_deep_agent(
        tools=[qa_tool, summarizer_tool],
        system_prompt=(
            "You are a retrieval-augmented deep agent. "
            "First use qa_tool(query, docs) to answer. "
            "Then use summarizer_tool(docs) to generate a summary."
        )
    )

    # 5. Retrieve docs using LangChain
    docs = await retriever.ainvoke(query)

    # 6. Pass docs + query into deep agent
    result = await deep_agent.ainvoke({
        "query": query,
        "docs": docs
    })

    print("\n--- FINAL DEEPAGENT OUTPUT ---")
    print(result["messages"][-1]["content"])


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
