"""Detect follow-up questions that should keep the previous filter."""

from __future__ import annotations

import re

_FOLLOW_UP_PATTERNS = (
    r"\bshow me \d+ more\b",
    r"\b\d+ more\b",
    r"\bmore examples\b",
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\band the total\b",
    r"\btotal count of the last\b",
    r"\bthe last two\b",
    r"\bin the current\b",
    r"\bcurrent (?:view|filter)\b",
    r"\bsame (?:category|intent|filter)\b",
    r"\bthose (?:rows|examples)\b",
    r"\bthat (?:category|intent)\b",
)


def is_follow_up_question(text: str) -> bool:
    """True when the user likely refers to the previous turn's filter or context."""
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in _FOLLOW_UP_PATTERNS)
