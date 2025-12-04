from src.agents.qa_agent import QAAgent
from src.agents.summarizer import SummarizerAgent
from src.retriever import get_retriever

qa_agent = QAAgent()
summ_agent = SummarizerAgent()
retriever = get_retriever()

def retrieve_node(state):
    query = state["query"]
    docs = retriever(query)
    return {"docs": docs}

def qa_node(state):
    answer = qa_agent.run(state["query"], state["docs"])
    return {"answer": answer}

def summarize_node(state):
    summary = summ_agent.run(state["answer"])
    return {"summary": summary}
