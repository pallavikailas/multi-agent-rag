import asyncio
import os
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

# DeepAgents modern API (langchain-ai/deepagents)
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
    # 1) vectorstore + retriever
    vectordb = build_system()
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # start small to keep token budget

    # 2) wrap existing agents as async tools
    qa_agent = QAAgent(retriever)
    summ_agent = SummarizerAgent()

    # deepagents accepts normal callables or async; wrap to match expected signatures
    async def qa_tool(payload: dict):
        # payload shape can be flexible; we'll accept {"query": str, "docs": list}
        q = payload.get("query") or payload.get("question")
        docs = payload.get("docs")
        if docs is None:
            # fallback to retrieval if docs not supplied
            docs = await retriever.ainvoke(q)
        # your QAAAgent.run returns a string
        return await qa_agent.run(q)

    async def summarizer_tool(payload: dict):
        docs = payload.get("docs")
        if docs is None:
            return ""
        return await summ_agent.run(docs)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0,
        api_key=getattr(settings, "groq_api_key", os.environ.get("GROQ_API_KEY")),
    )

    # 4) Create the deep agent, restricting built-ins so it only uses our tools
    agent = create_deep_agent(
        tools=[qa_tool, summarizer_tool],
        model=llm,
        system_prompt=(
            "You are a retrieval-augmented research agent. Use only the provided tools:\n"
            "- qa_tool(payload)  -> returns an answer string given a payload {'query':..., 'docs':[...]}\n"
            "- summarizer_tool(payload) -> returns a short summary string for the given docs\n"
            "Never call or rely on any built-in deepagents tools such as write_todos, filesystem, or subagents."
        ),
    )

    # 5) Prepare inputs and invoke the deep agent
    docs = await retriever.ainvoke(query)
    # deepagents accepts a dict-like request; use messages or a custom input depending on the package docs
    # modern deepagents examples use {"messages": [{"role":"user","content": "..."}]} or a custom payload,
    # but create_deep_agent also accepts plain payloads for compiled graphs.
    payload = {
        "query": query,
        "docs": docs
    }

    # `ainvoke` streams; use it to get final result. We can use aasync iterator or .invoke/.ainvoke.
    result = await agent.ainvoke(payload)

    # result structure varies; most examples show messages/choices. Try to print the final content robustly:
    if isinstance(result, dict) and "messages" in result:
        final = result["messages"][-1]["content"]
    elif isinstance(result, dict) and "output" in result:
        final = result["output"]
    else:
        final = str(result)

    print("\n--- FINAL OUTPUT ---\n")
    print(final)
    print("\n--------------------\n")


if __name__ == "__main__":
    # choose query from env or prompt
    if not os.isatty(0):
        q = os.environ.get("DEMO_QUERY", "What is the main point of the documents?")
    else:
        q = os.environ.get("DEMO_QUERY") or input("\n🔍 Enter your query: ").strip() or "What is the main point of the documents?"
    asyncio.run(demo_query(q))
