"""Query router: classifies user questions before tool use."""

from __future__ import annotations

import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME, NEBIUS_BASE_URL
from src.state import AgentState, QueryType

ROUTER_SYSTEM = """You classify user questions for a data analyst agent over the Bitext customer support dataset.

The dataset contains tagged customer support conversations with fields: instruction (customer message), response (agent reply), category, and intent.

The agent also keeps a short saved USER PROFILE (name, interests) — NOT the full chat log.

Reply on the first line with exactly one label:
- structured
- unstructured
- out_of_scope
- profile

On the second line, give a brief reason (one sentence).

structured = concrete data questions (counts, lists, examples, distributions).
unstructured = open-ended summaries or how agents typically respond.
profile = the user asks what you remember/know about THEM (saved profile), e.g. "what do you remember about me?"
out_of_scope = NOT about the dataset OR the saved user profile (general knowledge, poems, unrelated products, etc.).

Only classify — do not answer the question."""

_PROFILE_PATTERNS = (
    r"\bwhat do you remember about me\b",
    r"\bwhat do you know about me\b",
    r"\bwhat have you learned about me\b",
    r"\bwhat did you learn about me\b",
    r"\bwhat's my profile\b",
    r"\bwhat is my profile\b",
    r"\bdo you remember me\b",
    r"\bwhat do you recall about me\b",
)


def is_profile_question(text: str) -> bool:
    """Check if the user is asking about their saved profile."""
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in _PROFILE_PATTERNS)


def get_llm(max_tokens: int = 1024) -> ChatOpenAI:
    """Create the Nebius-backed chat model."""
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=NEBIUS_BASE_URL,
        api_key=os.environ["NEBIUS_API_KEY"],
        temperature=0,
        max_tokens=max_tokens,
    )


def _parse_route(text: str) -> tuple[QueryType, str]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0].lower() if lines else ""
    reason = lines[1] if len(lines) > 1 else text.strip()

    if "out_of_scope" in first or "out-of-scope" in first or "out of scope" in first:
        return "out_of_scope", reason
    if "profile" in first:
        return "profile", reason
    if "unstructured" in first:
        return "unstructured", reason
    if "structured" in first:
        return "structured", reason

    lower = text.lower()
    if re.search(r"out[- ]of[- ]scope", lower):
        return "out_of_scope", reason
    if re.search(r"\bprofile\b", lower) and "user" in lower:
        return "profile", reason
    if "unstructured" in lower:
        return "unstructured", reason
    return "structured", reason


def router_node(state: AgentState) -> dict:
    """Classify the latest user message."""
    messages = state["messages"]
    user_text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_text = msg.content
            break

    if is_profile_question(user_text):
        return {
            "query_type": "profile",
            "route_reason": "User asked about their saved profile.",
        }

    response = get_llm(max_tokens=150).invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=user_text),
        ]
    )
    query_type, reason = _parse_route(response.content or "")
    return {
        "query_type": query_type,
        "route_reason": reason,
    }
