"""Shared turn execution for CLI and Streamlit (graph + Bonus B recommender)."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from src.query_recommender import (
    RecommenderState,
    format_suggestion_message,
    is_confirmation,
    is_recommendation_request,
    suggest_query,
)
from src.user_profile import update_profile


@dataclass
class ReasoningStep:
    kind: str
    text: str


@dataclass
class TurnResult:
    final_answer: str
    query_type: str | None = None
    reasoning: list[ReasoningStep] = field(default_factory=list)
    recommender: RecommenderState = field(default_factory=RecommenderState)
    skip_profile_update: bool = False


def _truncate(text: str, max_len: int = 600) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... [{len(text) - max_len} more chars]"


def _collect_reasoning(node_name: str, update: dict) -> list[ReasoningStep]:
    steps: list[ReasoningStep] = []
    if node_name == "router":
        qtype = update.get("query_type")
        reason = update.get("route_reason", "")
        steps.append(ReasoningStep("router", f"type={qtype} | {reason}"))
    elif node_name == "decline":
        steps.append(ReasoningStep("router", "out-of-scope → declining"))
    elif node_name == "limit":
        steps.append(ReasoningStep("agent", "max iterations reached"))
    elif node_name == "agent":
        for msg in update.get("messages", []):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "?")
                        args = tc.get("args", {})
                        steps.append(ReasoningStep("tool_call", f"{name}({args})"))
                elif msg.content:
                    steps.append(ReasoningStep("final", str(msg.content)))
    elif node_name == "profile":
        for msg in update.get("messages", []):
            if isinstance(msg, AIMessage) and msg.content:
                steps.append(ReasoningStep("profile", str(msg.content)))
    elif node_name == "tools":
        for msg in update.get("messages", []):
            if isinstance(msg, ToolMessage):
                steps.append(ReasoningStep("observation", _truncate(str(msg.content))))
    return steps


def run_turn(
    graph: CompiledStateGraph,
    config: dict,
    user_input: str,
    recommender: RecommenderState | None = None,
    history_messages: list | None = None,
) -> TurnResult:
    """Run one user turn (recommender flow or full graph)."""
    rec = recommender or RecommenderState()
    history_messages = history_messages or []

    # Bonus B: execute pending suggestion after confirmation
    if rec.awaiting_confirmation and rec.pending_query:
        if is_confirmation(user_input):
            user_input = rec.pending_query
            rec = RecommenderState()
        else:
            revised = suggest_query(
                history_messages,
                config["configurable"]["user_id"],
                refinement=user_input,
                previous_suggestion=rec.pending_query,
            )
            rec.pending_query = revised
            return TurnResult(
                final_answer=format_suggestion_message(revised),
                query_type="recommendation",
                reasoning=[ReasoningStep("recommendation", f"Revised suggestion: {revised}")],
                recommender=rec,
                skip_profile_update=True,
            )

    # Bonus B: new recommendation request — do not run the graph yet
    if is_recommendation_request(user_input):
        suggested = suggest_query(
            history_messages,
            config["configurable"]["user_id"],
        )
        rec.pending_query = suggested
        rec.awaiting_confirmation = True
        return TurnResult(
            final_answer=format_suggestion_message(suggested),
            query_type="recommendation",
            reasoning=[
                ReasoningStep(
                    "recommendation",
                    f"Suggested (not executed): {suggested}",
                )
            ],
            recommender=rec,
            skip_profile_update=True,
        )

    turn_input = {
        "messages": [HumanMessage(content=user_input)],
        "iteration_count": 0,
    }

    reasoning: list[ReasoningStep] = []
    last_query_type: str | None = None
    final_answer = ""

    for event in graph.stream(turn_input, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "router":
                last_query_type = update.get("query_type")
            steps = _collect_reasoning(node_name, update)
            reasoning.extend(steps)
            for step in steps:
                if step.kind == "final" or step.kind == "profile":
                    final_answer = step.text

    if not final_answer:
        snapshot = graph.get_state(config)
        if snapshot.values and snapshot.values.get("messages"):
            last = snapshot.values["messages"][-1]
            if isinstance(last, AIMessage) and last.content:
                final_answer = str(last.content)

    skip_profile = last_query_type in ("out_of_scope", "profile")
    if not skip_profile:
        snapshot = graph.get_state(config)
        if snapshot.values:
            update_profile(config["configurable"]["user_id"], snapshot.values["messages"])

    return TurnResult(
        final_answer=final_answer,
        query_type=last_query_type,
        reasoning=reasoning,
        recommender=rec,
        skip_profile_update=skip_profile,
    )
