"""Suggest follow-up queries without running them until the user confirms."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.router import get_llm
from src.user_profile import EMPTY_PROFILE, load_profile

_RECOMMEND_PATTERNS = (
    r"\bwhat should i query next\b",
    r"\bwhat should i ask next\b",
    r"\bwhat can i query next\b",
    r"\bwhat could i ask next\b",
    r"\bsuggest (?:a |another )?(?:follow[- ]?up )?quer",
    r"\bwhat (?:should|can) i (?:query|ask|explore) next\b",
)

_CONFIRM_PATTERNS = (
    r"^(yes|yep|yeah|sure|ok|okay|go ahead|do it|please do|run it|execute)([.!]|$)",
    r"^yes[, ]+do it",
)


@dataclass
class RecommenderState:
    """Tracks a suggested query awaiting user confirmation."""

    pending_query: str | None = None
    awaiting_confirmation: bool = False


def is_recommendation_request(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in _RECOMMEND_PATTERNS)


def is_confirmation(text: str) -> bool:
    normalized = " ".join(text.lower().strip().rstrip(".!").split())
    return any(re.search(pattern, normalized) for pattern in _CONFIRM_PATTERNS)


def _recent_context(messages: list[BaseMessage], limit: int = 8) -> str:
    lines: list[str] = []
    for msg in messages[-limit:]:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            lines.append(f"Assistant: {msg.content[:400]}")
    return "\n".join(lines) if lines else "(no prior turns)"


def suggest_query(
    messages: list[BaseMessage],
    user_id: str,
    refinement: str | None = None,
    previous_suggestion: str | None = None,
) -> str:
    """Return a single suggested query string to run on the dataset."""
    profile = load_profile(user_id)
    profile_block = profile if profile != EMPTY_PROFILE else "(empty)"

    if refinement and previous_suggestion:
        task = (
            f"The user was offered this query: {previous_suggestion}\n"
            f"They replied: {refinement}\n"
            "Propose ONE revised query (one short sentence) about the Bitext dataset."
        )
    else:
        task = (
            "Propose ONE useful next query (one short sentence) about the Bitext "
            "customer support dataset based on the conversation and profile."
        )

    prompt = f"""You help suggest the next data analysis question for the Bitext customer support dataset.

User profile:
{profile_block}

Recent conversation:
{_recent_context(messages)}

{task}

Reply with ONLY the suggested query text — no quotes, no explanation."""

    llm = get_llm()
    suggestion = (llm.invoke([SystemMessage(content=prompt)]).content or "").strip()
    return suggestion.strip('"').strip("'")


def format_suggestion_message(suggested_query: str) -> str:
    return (
        f"You might want to try:\n\n**{suggested_query}**\n\n"
        "Should I go ahead and run that? (Reply **yes** to execute, or tell me how to change it.)"
    )
