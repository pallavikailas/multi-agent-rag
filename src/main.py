import asyncio
from .ingest import load_documents, chunk_documents
from .retriever import build_vectorstore
from .config import settings

from langchain_groq import ChatGroq
from deepagents import Orchestrator 


# ---------------------------
# Disable DeepAgents built-ins
# ---------------------------
def create_rag_agent(model, tools):
    """
    Wrapper to ensure DeepAgents v0.2.x ONLY uses our tools
    and disables all builtin tools that cause tool_use_failed errors.
    """
    orch = Orchestrator(
        model=model,
        extra_tools=tools,      # <--- forces ONLY these tools
        enable_fs=False,        # disable filesystem
        enable_code_exec=False, # disable terminal + code execution
    )
    return orch


# ---------------------------------
# RAG Tools (QA + Summarization)
# ---------------------------------
from langchain.tools import tool


@tool
def qa_tool(query: str, docs: list):
    """Answer a question using retrieved docs."""
    text = "\n\n".join(d.page_content for d in docs[:3])
    return f"Context:\n{text}\n\nQuestion: {query}"


@tool
def summarizer_tool(docs: list):
    """Summarize retrieved docs."""
    text = "\n\n".join(d.page_content for d in docs[:3])
    return f"Summarize this:\n{text}"


# -------------------
# System construction
# -------------------
def build_system():
    docs = load_documents(settings.data_dir)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    vectordb = build_vectorstore(chunks)
    return vectordb


# -------------------
# Demo query pipeline
# -------------------
async def demo_query(query: str):
    vectordb = build_system()

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})  # reduced to prevent token overflow

    # Use supported Groq model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=settings.groq_api_key,
    )

    # Create DeepAgent orchestrator with ONLY our RAG tools
    deep_agent = create_rag_agent(
        model=llm,
        tools=[qa_tool, summarizer_tool],
    )

    # Run the DeepAgent orchestration
    result = await deep_agent.ainvoke({
        "query": query,
        "retriever": retriever
    })

    print("\n--- RESULT ---")
    print(result)
