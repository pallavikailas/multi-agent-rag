from src.deep_agents.manager import build_deep_agent
from src.agents.qa_agent import QAAAgent
from src.agents.summarizer import SummarizerAgent


async def retrieve_node(state):
    retriever = state["retriever"]
    docs = await retriever.ainvoke(state["query"])
    return {"docs": docs}


async def deepagent_node(state):
    query = state["query"]
    docs = state["docs"]

    qa_agent = QAAAgent(state["retriever"])
    summarizer = SummarizerAgent()

    agent = build_deep_agent(
        qa_func=lambda q, d: qa_agent.run(q),
        summary_func=lambda d: summarizer.run(d)
    )

    result = await agent.arun(
        f"Given the query '{query}', use tools (qa_tool or summary_tool) to produce the final answer.",
        docs=docs,
        query=query
    )

    return {"final": result}
