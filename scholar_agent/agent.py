from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from scholar_agent.utils.nodes import (
    draft_deep_analysis_note_node,
    generate_summary_node,
    human_note_review_node,
    human_summary_review_node,
    initialize_memory_node,
    load_review_queue_node,
    prepare_deep_analysis_context_node,
    record_summary_decision_node,
    revise_deep_analysis_note_node,
    save_final_note_node,
    scan_library_node,
    select_unsummarized_paper_node,
)
from scholar_agent.utils.state import ResearchAgentState


def route_after_select_unsummarized(state: ResearchAgentState) -> str:
    if state.get("status") == "unsummarized_paper_selected":
        return "generate_summary"
    return "load_review_queue"


def route_after_review_queue(state: ResearchAgentState) -> str:
    if state.get("review_queue"):
        return "human_summary_review"
    return "end"


def route_after_summary_decision(state: ResearchAgentState) -> str:
    if state.get("user_decision") == "deep_analysis":
        return "prepare_deep_analysis_context"
    return "load_review_queue"


def route_after_note_review(state: ResearchAgentState) -> str:
    if state.get("note_review_action") == "confirm":
        return "save_final_note"
    return "revise_deep_analysis_note"


def build_graph(*, checkpointer: Any | None = None):
    builder = StateGraph(ResearchAgentState)

    builder.add_node("initialize_memory", initialize_memory_node)
    builder.add_node("scan_library", scan_library_node)
    builder.add_node("select_unsummarized_paper", select_unsummarized_paper_node)
    builder.add_node("generate_summary", generate_summary_node)
    builder.add_node("load_review_queue", load_review_queue_node)
    builder.add_node("human_summary_review", human_summary_review_node)
    builder.add_node("record_summary_decision", record_summary_decision_node)
    builder.add_node("prepare_deep_analysis_context", prepare_deep_analysis_context_node)
    builder.add_node("draft_deep_analysis_note", draft_deep_analysis_note_node)
    builder.add_node("human_note_review", human_note_review_node)
    builder.add_node("revise_deep_analysis_note", revise_deep_analysis_note_node)
    builder.add_node("save_final_note", save_final_note_node)

    builder.add_edge(START, "initialize_memory")
    builder.add_edge("initialize_memory", "scan_library")
    builder.add_edge("scan_library", "select_unsummarized_paper")
    builder.add_conditional_edges(
        "select_unsummarized_paper",
        route_after_select_unsummarized,
        {
            "generate_summary": "generate_summary",
            "load_review_queue": "load_review_queue",
        },
    )
    builder.add_edge("generate_summary", "select_unsummarized_paper")
    builder.add_conditional_edges(
        "load_review_queue",
        route_after_review_queue,
        {
            "human_summary_review": "human_summary_review",
            "end": END,
        },
    )
    builder.add_edge("human_summary_review", "record_summary_decision")
    builder.add_conditional_edges(
        "record_summary_decision",
        route_after_summary_decision,
        {
            "prepare_deep_analysis_context": "prepare_deep_analysis_context",
            "load_review_queue": "load_review_queue",
        },
    )
    builder.add_edge("prepare_deep_analysis_context", "draft_deep_analysis_note")
    builder.add_edge("draft_deep_analysis_note", "human_note_review")
    builder.add_conditional_edges(
        "human_note_review",
        route_after_note_review,
        {
            "revise_deep_analysis_note": "revise_deep_analysis_note",
            "save_final_note": "save_final_note",
        },
    )
    builder.add_edge("revise_deep_analysis_note", "human_note_review")
    builder.add_edge("save_final_note", "load_review_queue")

    if checkpointer is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
local_graph = build_graph(checkpointer=InMemorySaver())
