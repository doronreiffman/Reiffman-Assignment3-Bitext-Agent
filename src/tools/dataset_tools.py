"""Dataset tools for the ReAct agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool

from src.dataset_context import current_df, reset_working, set_working_from_df
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
    """List intent names in the dataset, optionally filtered by category.

    Use when the user asks what intents exist, or before filter_by_intent when
    you need the exact intent string (e.g. get_refund vs track_refund).
    Do not use for counting rows — use count_rows after filtering.
    """
    from src.dataset_context import _FULL_DF

    base = _FULL_DF
    if category:
        cat = _normalize_category(category)
        base = base[base["category"] == cat]
        if base.empty:
            return f"No rows found for category '{cat}'."
    intents = sorted(base["intent"].unique().tolist())
    return json.dumps(intents)


@tool
def list_categories() -> str:
    """List all high-level category names in the dataset (e.g. ACCOUNT, REFUND, SHIPPING).

    Use when the user asks what categories exist. Do not use for intent lists or counts.
    """
    from src.dataset_context import _FULL_DF

    categories = sorted(_FULL_DF["category"].unique().tolist())
    return json.dumps(categories)


@tool(args_schema=FilterByCategoryInput)
def filter_by_category(category: str) -> str:
    """Restrict the working view to one category. Required before sampling/counting within a category.

    Call before count_rows, sample_examples, or intent_distribution when the question
    mentions a category (e.g. SHIPPING, ACCOUNT). Use list_categories if unsure of exact names.
    """
    cat = _normalize_category(category)
    filtered = current_df()
    filtered = filtered[filtered["category"] == cat]
    set_working_from_df(filtered)
    if filtered.empty:
        return f"No rows found for category '{cat}'."
    return f"Filtered to category '{cat}': {len(filtered)} rows."


@tool(args_schema=FilterByIntentInput)
def filter_by_intent(intent: str) -> str:
    """Restrict the working view to one intent (e.g. get_refund, complaint, delivery_options).

    Use for questions about a specific intent. Chain with count_rows for 'how many' questions.
    Call list_intents if unsure of the exact intent string.
    Filters the current view (or full dataset if none). If you get 0 rows unexpectedly, call reset_filters and try again.
    """
    intent_norm = _normalize_intent(intent)
    filtered = current_df()
    filtered = filtered[filtered["intent"] == intent_norm]
    set_working_from_df(filtered)
    if filtered.empty:
        return f"No rows found for intent '{intent_norm}'."
    return f"Filtered to intent '{intent_norm}': {len(filtered)} rows."


@tool
def reset_filters() -> str:
    """Clear all filters and reset the working view to the full dataset.

    Use when a previous filter was wrong or you need to start a new analysis from scratch.
    """
    from src.dataset_context import _FULL_DF

    reset_working()
    return f"Filters reset. Full dataset has {len(_FULL_DF)} rows."


@tool
def count_rows() -> str:
    """Return the number of rows in the current filtered view.

    Use after filter_by_category or filter_by_intent when the user asks 'how many'.
    If no filter was applied yet, counts the entire dataset.
    """
    return str(len(current_df()))


@tool(args_schema=SampleExamplesInput)
def sample_examples(n: int = 3) -> str:
    """Return random customer instruction/response pairs from the current filtered view.

    Use when the user wants examples or when you need text to summarize (unstructured questions).
    Apply filters first so examples match the requested category or intent.
    """
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
    """Return counts of each intent in the current filtered view.

    Use after filter_by_category when the user asks for intent distribution within a category.
    """
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
