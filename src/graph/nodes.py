import asyncio
from src.agents.qa_agent import QAAAgent
from src.agents.summarizer import SummarizerAgent


async def retrieve_node(state):
    retriever = state["retriever"]
    query = state["query"]

    docs = await retriever.ainvoke(query)
    return {"docs": docs}


async def qa_node(state):
    retriever = state["retriever"]
    query = state["query"]

    agent = QAAAgent(retriever=retriever)
    answer = await agent.run(query)
    return {"answer": answer}


async def summarize_node(state):
    docs = state["docs"]

    agent = SummarizerAgent()
    summary = await agent.run(docs)
    return {"summary": summary}
