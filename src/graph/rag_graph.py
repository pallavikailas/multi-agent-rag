from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

from src.agents.qa_agent import QARetrievalAgent
from src.agents.summarizer import SummarizerAgent
from src.retriever import build_vectorstore


class RAGState(TypedDict):
    query: str
    context: Optional[List[str]]
    qa_output: Optional[str]
    summary: Optional[str]


# 1. Node functions --------------------------------------------------

def retrieve_node(state: RAGState):
    retriever = build_vectorstore()
    ctx = retriever.get_relevant_documents(state["query"])
    return {"context": [d.page_content for d in ctx]}


def qa_node(state: RAGState):
    agent = QARetrievalAgent()
    answer = agent.answer(question=state["query"], context=state["context"])
    return {"qa_output": answer}


def summarizer_node(state: RAGState):
    agent = SummarizerAgent()
    summary = agent.summarize(context=state["context"])
    return {"summary": summary}


# 2. Build the LangGraph DAG -----------------------------------------

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qa", qa_node)
    graph.add_node("summarize", summarizer_node)

    graph.set_entry_point("retrieve")

    # after retrieve → run QA + summary in parallel (real LangGraph fan-out)
    graph.add_edge("retrieve", "qa")
    graph.add_edge("retrieve", "summarize")

    # merge both results → END
    graph.add_edge("qa", END)
    graph.add_edge("summarize", END)

    return graph.compile()


compiled_rag_graph = build_rag_graph()
