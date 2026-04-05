"""
graph.py - LangGraph StateGraph definition for the DRHP Capital Structure Agent.

Flow:
  load_documents → classify_node → link_node → extract_node
                → validate_node → table_builder_node → render_node
"""

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    classify_node,
    link_node,
    extract_node,
    validate_node,
    table_builder_node,
    render_node,
)


def load_documents_node(state: AgentState) -> dict:
    """
    Pass-through node: raw_documents already populated by main.py before graph.invoke().
    Exists as an explicit node so the graph is self-documenting and testable in isolation.
    """
    print(f"[load_documents_node] {len(state['raw_documents'])} documents in state.")
    return {}


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("load_documents", load_documents_node)
    graph.add_node("classify", classify_node)
    graph.add_node("link", link_node)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("table_builder", table_builder_node)
    graph.add_node("render", render_node)

    # Linear edges
    graph.set_entry_point("load_documents")
    graph.add_edge("load_documents", "classify")
    graph.add_edge("classify", "link")
    graph.add_edge("link", "extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "table_builder")
    graph.add_edge("table_builder", "render")
    graph.add_edge("render", END)

    return graph.compile()
