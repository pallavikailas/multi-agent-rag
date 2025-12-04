from langgraph.graph import StateGraph
from .nodes import retrieve_node, qa_node, summarize_node
from src.deepagents.manager import RAGSupervisor

def build_graph():
    graph = StateGraph()

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("qa", qa_node)
    graph.add_node("summarize", summarize_node)

    sup = RAGSupervisor()

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        sup.route,
        {"qa": "qa"}
    )

    graph.add_conditional_edges(
        "qa",
        sup.route,
        {"summarize": "summarize"}
    )

    graph.add_node("summarize", summarize_node)

    graph.set_finish_point("summarize")

    return graph.compile()
