"""LangGraph definition: router → decline | ReAct agent loop."""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import MAX_ITERATIONS
from src.dataset_context import get_working_row_ids, reset_working, sync_row_ids_from_state
from src.router import get_llm, is_profile_question, router_node
from src.state import AgentState, QueryType
from src.tools.dataset_tools import get_all_tools
from src.user_profile import EMPTY_PROFILE, load_profile

STRUCTURED_SYSTEM = """You are a data analyst for the Bitext customer support dataset.

Rules:
- Never guess numbers. Always use tools and report tool results.
- For "how many" questions: call filter_by_intent (or filter_by_category) FIRST, then count_rows.
- Call only ONE tool at a time, wait for the result, then decide the next step.
- When done, give a short direct answer to the user (include the number).
- Follow-up questions ("show me 3 more", "what about refunds?") refer to earlier turns in this session.

Intent names are lowercase with underscores (e.g. get_refund, track_refund, complaint).
Categories are uppercase (e.g. REFUND, ACCOUNT, SHIPPING).
Use list_intents if unsure which intent matches the user's wording.
If the user asks what categories exist, call list_categories.
If the user asks what intents exist, call list_intents.
If the user asks what you remember about them, answer from the user profile below (not from chat logs)."""

UNSTRUCTURED_SYSTEM = """You are a data analyst for the Bitext customer support dataset.

The user wants a summary or qualitative analysis. Use tools to fetch representative data:
filter to the relevant category or intent, then sample_examples (use n=10–15 for summaries).
Base your answer only on tool results — do not invent examples.
Follow-up questions refer to earlier turns in this session.
If the user asks what you remember about them, answer from the user profile below."""

DECLINE_MESSAGE = (
    "I can only answer questions about the Bitext customer support dataset "
    "(categories, intents, counts, examples, and summaries of that data). "
    "I can't help with that question."
)

FALLBACK_MESSAGE = (
    "I couldn't finish analyzing your question within the step limit. "
    "Try a narrower question (e.g. specify a category or intent)."
)


def decline_node(state: AgentState) -> dict:
    """Politely decline out-of-scope queries."""
    return {
        "messages": [AIMessage(content=DECLINE_MESSAGE)],
    }


def profile_node(state: AgentState, config: RunnableConfig) -> dict:
    """Answer from the saved user profile."""
    user_id = config.get("configurable", {}).get("user_id", "default")
    profile = load_profile(user_id)
    if not profile or profile == EMPTY_PROFILE:
        content = (
            "I don't have any saved information about you yet. "
            "Tell me your name or what you're interested in, and I'll remember it for next time."
        )
    else:
        content = f"Here's what I remember about you:\n\n{profile}"
    return {"messages": [AIMessage(content=content)]}


def _system_prompt_for(query_type: QueryType | None, user_id: str) -> str:
    base = UNSTRUCTURED_SYSTEM if query_type == "unstructured" else STRUCTURED_SYSTEM
    profile = load_profile(user_id)
    if profile and profile != EMPTY_PROFILE:
        base += f"\n\nUser profile:\n{profile}"
    return base


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Call the LLM with tools bound."""
    user_id = config.get("configurable", {}).get("user_id", "default")
    llm = get_llm().bind_tools(get_all_tools(), parallel_tool_calls=False)
    query_type = state.get("query_type")
    prompt = _system_prompt_for(query_type, user_id)
    reason = state.get("route_reason")
    if reason:
        prompt += f"\n\nClassifier note: {reason}"
    system = SystemMessage(content=prompt)

    convo = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    convo = [system] + convo

    response = llm.invoke(convo)
    iteration = state.get("iteration_count", 0) + 1
    return {
        "messages": [response],
        "iteration_count": iteration,
    }


def route_after_router(state: AgentState) -> str:
    if state.get("query_type") == "out_of_scope":
        return "decline"
    if state.get("query_type") == "profile":
        return "profile"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and is_profile_question(msg.content):
            return "profile"
    return "agent"


def route_after_agent(state: AgentState) -> str:
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "limit"
    messages = state["messages"]
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def limit_node(state: AgentState) -> dict:
    """Return graceful fallback when max iterations exceeded."""
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def tools_node(state: AgentState) -> dict:
    """Run tools and sync working row ids back into graph state."""
    sync_row_ids_from_state(state.get("working_row_ids"))
    tool_node = ToolNode(get_all_tools())
    result = tool_node.invoke(state)
    return {**result, "working_row_ids": get_working_row_ids()}


def _latest_user_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


_FOLLOW_UP_PATTERNS = (
    r"\bshow me \d+ more\b",
    r"\b\d+ more\b",
    r"\bmore examples\b",
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\band the total\b",
    r"\btotal count of the last\b",
    r"\bthe last two\b",
    r"\bsame (?:category|intent|filter)\b",
    r"\bthose (?:rows|examples)\b",
)


def _is_follow_up(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(p, normalized) for p in _FOLLOW_UP_PATTERNS)


def prepare_turn(state: AgentState) -> dict:
    """Restore filter context for follow-ups; clear stale filters on new questions."""
    user_text = _latest_user_message(state)
    if _is_follow_up(user_text):
        sync_row_ids_from_state(state.get("working_row_ids"))
    else:
        reset_working()
        return {"iteration_count": 0, "working_row_ids": None}
    return {"iteration_count": 0}


def build_graph(checkpointer=None):
    """Compile the LangGraph agent, optionally with a checkpointer for session memory."""
    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_turn)
    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("profile", profile_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("limit", limit_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"decline": "decline", "profile": "profile", "agent": "agent"},
    )
    graph.add_edge("decline", END)
    graph.add_edge("profile", END)
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "limit": "limit", END: END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("limit", END)

    return graph.compile(checkpointer=checkpointer)
