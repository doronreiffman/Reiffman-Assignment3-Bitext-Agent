"""Dataset tools for the ReAct agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src.dataset_context import current_df, full_df, reset_working, set_working_from_df
from src.tools.schemas import (
    FilterByCategoryInput,
    FilterByIntentInput,
    ListIntentsInput,
    SampleExamplesInput,
)


def _normalize_category(category: str) -> str:
    return category.strip().upper()


def _normalize_intent(intent: str) -> str:
    return intent.strip().lower()


@tool(args_schema=ListIntentsInput)
def list_intents(category: str | None = None) -> str:
    """List all intent names in the dataset, optionally filtered to one category."""
    base = full_df()
    if category:
        cat = _normalize_category(category)
        base = base[base["category"] == cat]
        if base.empty:
            return f"No rows found for category '{cat}'."
    intents = sorted(base["intent"].unique().tolist())
    return json.dumps(intents)


@tool
def list_categories() -> str:
    """List all category names in the dataset (e.g. ACCOUNT, REFUND, SHIPPING)."""
    categories = sorted(full_df()["category"].unique().tolist())
    return json.dumps(categories)


@tool(args_schema=FilterByCategoryInput)
def filter_by_category(category: str) -> str:
    """Filter the working view to rows matching a category (e.g. SHIPPING, ACCOUNT)."""
    cat = _normalize_category(category)
    filtered = current_df()
    filtered = filtered[filtered["category"] == cat]
    set_working_from_df(filtered)
    if filtered.empty:
        return f"No rows found for category '{cat}'."
    return f"Filtered to category '{cat}': {len(filtered)} rows."


@tool(args_schema=FilterByIntentInput)
def filter_by_intent(intent: str) -> str:
    """Filter the working view to rows with a specific intent (e.g. get_refund, complaint)."""
    intent_norm = _normalize_intent(intent)
    filtered = current_df()
    filtered = filtered[filtered["intent"] == intent_norm]
    set_working_from_df(filtered)
    if filtered.empty:
        return f"No rows found for intent '{intent_norm}'."
    return f"Filtered to intent '{intent_norm}': {len(filtered)} rows."


@tool
def reset_filters() -> str:
    """Clear all filters and reset the working view to the full dataset."""
    reset_working()
    return f"Filters reset. Full dataset has {len(full_df())} rows."


@tool
def count_rows() -> str:
    """Return the number of rows in the current filtered view."""
    return str(len(current_df()))


@tool(args_schema=SampleExamplesInput)
def sample_examples(n: int = 3) -> str:
    """Return n random instruction/response examples from the current filtered view."""
    df = current_df()
    if df.empty:
        return "No rows in the current view. Apply a filter or reset_filters."
    sample = df.sample(n=min(n, len(df)))
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "category": row["category"],
                "intent": row["intent"],
                "instruction": row["instruction"],
                "response": row["response"],
            }
        )
    return json.dumps(rows, indent=2)


@tool
def intent_distribution() -> str:
    """Return counts of each intent in the current filtered view."""
    df = current_df()
    if df.empty:
        return "No rows in the current view."
    counts = df["intent"].value_counts().to_dict()
    return json.dumps(counts, indent=2)


def get_all_tools() -> list:
    """Return all dataset tools for the agent."""
    return [
        list_categories,
        list_intents,
        filter_by_category,
        filter_by_intent,
        reset_filters,
        count_rows,
        sample_examples,
        intent_distribution,
    ]
