#!/usr/bin/env python3
"""FastMCP server exposing Bitext dataset analyst tools (Task 3)."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_ROOT, "..", "..", ".env"))

from fastmcp import FastMCP

from src.data_loader import load_bitext_dataframe
from src.dataset_context import reset_working, set_full_dataset

mcp = FastMCP(
    "bitext-customer-support-analyst",
    instructions=(
        "Tools for analyzing the Bitext customer support dataset. "
        "For counts or examples within a category or intent, call filter_by_category "
        "or filter_by_intent first, then count_rows or sample_examples."
    ),
)


def _ensure_dataset_loaded() -> None:
    try:
        from src.dataset_context import full_df
        full_df()
    except RuntimeError:
        set_full_dataset(load_bitext_dataframe())


@mcp.tool
def list_categories() -> str:
    """List all high-level category names (e.g. ACCOUNT, REFUND, SHIPPING)."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import list_categories as tool

    return tool.invoke({})


@mcp.tool
def list_intents(category: str | None = None) -> str:
    """List intent names, optionally limited to one category."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import list_intents as tool

    return tool.invoke({"category": category})


@mcp.tool
def filter_by_category(category: str) -> str:
    """Restrict the working view to rows in the given category."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import filter_by_category as tool

    return tool.invoke({"category": category})


@mcp.tool
def filter_by_intent(intent: str) -> str:
    """Restrict the working view to rows with the given intent (e.g. get_refund)."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import filter_by_intent as tool

    return tool.invoke({"intent": intent})


@mcp.tool
def count_rows() -> str:
    """Count rows in the current filtered view (after a filter tool)."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import count_rows as tool

    return tool.invoke({})


@mcp.tool
def sample_examples(n: int = 3) -> str:
    """Return random instruction/response examples from the current filtered view."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import sample_examples as tool

    return tool.invoke({"n": n})


@mcp.tool
def intent_distribution() -> str:
    """Return intent counts for the current filtered view (often after filter_by_category)."""
    _ensure_dataset_loaded()
    from src.tools.dataset_tools import intent_distribution as tool

    return tool.invoke({})


if __name__ == "__main__":
    print("Loading Bitext dataset...", file=sys.stderr)
    reset_working()
    set_full_dataset(load_bitext_dataframe())
    print("Dataset ready. Starting MCP server (stdio).", file=sys.stderr)
    mcp.run()
