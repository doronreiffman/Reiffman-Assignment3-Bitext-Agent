#!/usr/bin/env python3
"""Regression test for cross-turn filter handling (Task 2a).

Bug: "How many complaints?" then "What about refunds?" returned 0. The previous
complaint filter was carried into the next turn and intersected with the refund
filter (a row has exactly one intent) -> empty set.

Root fix: filter state is turn-scoped. prepare_turn resets the working view at
the start of EVERY turn regardless of phrasing, so a new filter always applies to
the full dataset and can never intersect a stale filter. (Conversational context
still lives in the checkpointer's chat history; the agent re-applies filters from
it for follow-ups.)

This guards against the original brittleness: the fix must NOT depend on matching
particular follow-up phrases. No LLM/network needed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.data_loader import load_bitext_dataframe
from src.dataset_context import (
    full_df,
    get_working_row_ids,
    reset_working,
    set_full_dataset,
)
from src.graph import prepare_turn
from src.tools.dataset_tools import count_rows, filter_by_intent

# Phrasings that all mean "now switch to refunds" — including ones that previously
# matched a filter-reuse regex ("more", "those") and ones that never matched any.
TOPIC_SWITCH_PHRASINGS = [
    "What about refunds?",
    "And refunds?",
    "show me 3 more refund examples",
    "those examples but for refunds",
    "now refunds",
]


def test_filter_resets_every_turn_regardless_of_phrasing() -> None:
    set_full_dataset(load_bitext_dataframe())
    expected_refunds = int((full_df()["intent"] == "get_refund").sum())
    assert expected_refunds > 0, "fixture should contain get_refund rows"

    for follow_up in TOPIC_SWITCH_PHRASINGS:
        reset_working()
        # Turn 1 leaves a complaint filter active.
        filter_by_intent.invoke({"intent": "complaint"})
        assert int(count_rows.invoke({})) > 0, "turn 1 should find complaint rows"

        # Turn 2: ANY phrasing must start from a clean filter (no phrase heuristic).
        state = {
            "messages": [HumanMessage(follow_up)],
            "working_row_ids": get_working_row_ids(),
        }
        prepare_turn(state)
        assert get_working_row_ids() is None, (
            f"{follow_up!r}: prepare_turn must reset the working view to the full dataset"
        )

        # The new filter therefore applies to the full dataset, not the complaint subset.
        filter_by_intent.invoke({"intent": "get_refund"})
        got = int(count_rows.invoke({}))
        assert got == expected_refunds, (
            f"{follow_up!r}: expected {expected_refunds} refund rows from the full "
            f"dataset, got {got} (stale filter not cleared?)"
        )

    print(
        f"OK every follow-up phrasing resets to the full dataset "
        f"(get_refund={expected_refunds} for all {len(TOPIC_SWITCH_PHRASINGS)} phrasings)"
    )


if __name__ == "__main__":
    test_filter_resets_every_turn_regardless_of_phrasing()
    print("\nFollow-up filter test passed.")
