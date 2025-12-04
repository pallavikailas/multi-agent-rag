import asyncio
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional

from src.agents.qa_agent import QARetrievalAgent
from src.agents.summarizer import SummarizerAgent
from src.ingest import load_documents, chunk_documents
from src.config import settings
from src.retriever import build_vectorstore
    

class RAGState(TypedDict):
    query: str
    retriever: any
    context: Optional[List[str]]
    qa_output: Optional[str]
    summary: Optional[str]


# 1. Node functions --------------------------------------------------

def retrieve_node(state: RAGState):
    retriever = state["retriever"]
    docs = retriever._get_relevant_documents(state["query"], run_manager=None)
    return {"context": [d.page_content for d in docs]}


def qa_node(state: RAGState):
    retriever = state["retriever"]
    agent = QARetrievalAgent(retriever)

    answer = asyncio.run(agent.run(state["query"]))

    return {"qa_output": answer}


def summarizer_node(state: RAGState):
    agent = SummarizerAgent()
    
    # Convert strings back into fake docs
    class FakeDoc:
        def __init__(self, text):
            self.page_content = text

    documents = [FakeDoc(t) for t in state["context"]]
    summary = asyncio.run(agent.run(documents))

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
