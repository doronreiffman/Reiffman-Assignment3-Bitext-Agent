#!/usr/bin/env python3
"""Regression test for topic-switch follow-ups (Task 2a).

Bug: "How many complaints are in the dataset?" then "What about refunds?"
returned 0. "what about ..." was treated as a *filter-reuse* follow-up, so the
complaint filter was kept and the refund filter intersected it to the empty set
(a row has exactly one intent).

A topic-switch ("what about X") must reset the working view to the full dataset;
a genuine reuse follow-up ("show me 3 more") must keep it. Conversational context
is preserved by the checkpointer either way, so resetting the row-id filter is safe.

No LLM/network needed — this exercises the deterministic prepare/filter layers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage

from src.data_loader import load_bitext_dataframe
from src.dataset_context import get_working_row_ids, reset_working, set_full_dataset
from src.graph import prepare_turn
from src.tools.dataset_tools import count_rows, filter_by_intent


def test_topic_switch_resets_filter() -> None:
    set_full_dataset(load_bitext_dataframe())
    reset_working()

    # Turn 1: count complaints -> leaves a complaint filter active.
    filter_by_intent.invoke({"intent": "complaint"})
    complaints = int(count_rows.invoke({}))
    assert complaints > 0, "expected some complaint rows in turn 1"

    # Turn 2: "What about refunds?" is a TOPIC SWITCH, not filter reuse.
    state = {
        "messages": [HumanMessage("What about refunds?")],
        "working_row_ids": get_working_row_ids(),
    }
    prepare_turn(state)
    assert get_working_row_ids() is None, (
        "topic-switch follow-up must reset the working view to the full dataset; "
        "the previous complaint filter was kept instead"
    )

    # prepare_turn reset the working view, so the refund filter sees the full dataset.
    filter_by_intent.invoke({"intent": "get_refund"})
    refund_rows = int(count_rows.invoke({}))
    assert refund_rows > 0, f"expected refund rows after reset, got {refund_rows}"
    print(f"OK topic-switch reset: complaints={complaints}, get_refund={refund_rows}")


def test_more_reuses_filter() -> None:
    set_full_dataset(load_bitext_dataframe())
    reset_working()

    filter_by_intent.invoke({"intent": "complaint"})
    before = get_working_row_ids()

    state = {"messages": [HumanMessage("show me 3 more")], "working_row_ids": before}
    prepare_turn(state)
    assert get_working_row_ids() == before, (
        "genuine reuse follow-up ('show me 3 more') must preserve the working view"
    )
    print("OK filter reuse preserved for 'show me 3 more'")


if __name__ == "__main__":
    test_topic_switch_resets_filter()
    test_more_reuses_filter()
    print("\nAll follow-up routing tests passed.")
