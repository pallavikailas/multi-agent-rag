from langgraph.graph import StateGraph
from .nodes import retrieve_node, deepagent_node


def build_graph():
    graph = StateGraph()

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("deep_agent", deepagent_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "deep_agent")
    graph.set_finish_point("deep_agent")

    return graph.compile(async_mode=True)
