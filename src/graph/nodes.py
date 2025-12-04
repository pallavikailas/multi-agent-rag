from src.deepagents.manager import QAADeepAgent, SummDeepAgent
from src.agents.qa_agent import QAAAgent
from src.agents.summarizer import SummarizerAgent


async def retrieve_node(state):
    retriever = state["retriever"]
    docs = await retriever.ainvoke(state["query"])
    return {"docs": docs}


async def qa_node(state):
    qa_agent = QAAAgent(state["retriever"])
    deep_agent = QAADeepAgent(lambda q, d: qa_agent.run(q))
    return await deep_agent.run(state)


async def summarize_node(state):
    summ_agent = SummarizerAgent()
    deep_agent = SummDeepAgent(summ_agent.run)
    return await deep_agent.run(state)
