from langgraph.graph import StateGraph
from .nodes import retrieve_node, qa_node, summarize_node
from src.deepagents.manager import RAGOrchestrator


def build_graph():
    graph = StateGraph()
    orchestrator = RAGOrchestrator()

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qa_agent", qa_node)
    graph.add_node("summarizer_agent", summarize_node)

    graph.set_entry_point("retrieve")

    def route(state):
        return orchestrator.route(state)

    graph.add_conditional_edges(
        "retrieve",
        route,
        {"qa_agent": "qa_agent"}
    )

    graph.add_conditional_edges(
        "qa_agent",
        route,
        {"summarizer_agent": "summarizer_agent"}
    )

    graph.set_finish_point("summarizer_agent")

    return graph.compile(async_mode=True)
