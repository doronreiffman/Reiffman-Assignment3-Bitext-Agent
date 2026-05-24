"""Per-user distilled profile, stored separately from conversation checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config import PROFILES_DIR
from src.router import get_llm

EMPTY_PROFILE = "No profile information yet."


def _profile_path(user_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
    return PROFILES_DIR / f"{safe}.json"


def load_profile(user_id: str) -> str:
    """Load distilled profile text for a user."""
    path = _profile_path(user_id)
    if not path.exists():
        return EMPTY_PROFILE
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("profile", EMPTY_PROFILE) or EMPTY_PROFILE


def save_profile(user_id: str, profile: str) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(user_id)
    path.write_text(
        json.dumps({"profile": profile.strip()}, indent=2),
        encoding="utf-8",
    )


def _latest_exchange(messages: list[BaseMessage]) -> tuple[str, str]:
    user_text = ""
    assistant_text = ""
    for msg in reversed(messages):
        if not assistant_text and isinstance(msg, AIMessage) and msg.content:
            assistant_text = str(msg.content)
        if isinstance(msg, HumanMessage):
            user_text = str(msg.content)
            break
    return user_text, assistant_text


def update_profile(user_id: str, messages: list[BaseMessage]) -> str:
    """Merge facts from the latest turn into the user's profile."""
    user_text, assistant_text = _latest_exchange(messages)
    if not user_text:
        return load_profile(user_id)

    current = load_profile(user_id)
    llm = get_llm()
    prompt = f"""You maintain a short user profile for a data-analyst chatbot (not a transcript).

Current profile:
{current}

Latest user message:
{user_text}

Latest assistant reply:
{assistant_text}

Update the profile with any new durable facts (name, preferences, topics they care about).
Do not copy whole messages. Keep it under 8 bullet points. If nothing new, return the current profile unchanged."""

    updated = llm.invoke([SystemMessage(content=prompt)]).content or current
    save_profile(user_id, str(updated))
    return str(updated)
