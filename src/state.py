"""LangGraph agent state definition."""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

QueryType = Literal["structured", "unstructured", "out_of_scope", "profile"]


class AgentState(TypedDict):
    """State carried through the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    query_type: QueryType | None
    route_reason: str | None
    working_row_ids: list[int] | None
    iteration_count: int
