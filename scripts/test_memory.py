#!/usr/bin/env python3
"""Quick tests for Task 2 session memory and user profile."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, ".env"))
load_dotenv(os.path.join(_root, "..", "..", ".env"))

from langchain_core.messages import HumanMessage

from src.checkpointer import get_checkpointer
from src.data_loader import load_bitext_dataframe
from src.dataset_context import set_full_dataset
from src.graph import build_graph
from src.user_profile import load_profile, save_profile


def test_session_memory() -> None:
    if not os.environ.get("NEBIUS_API_KEY"):
        print("Skip API tests: no NEBIUS_API_KEY")
        return

    set_full_dataset(load_bitext_dataframe())
    session = "test_memory_session"
    config = {"configurable": {"thread_id": session, "user_id": "test_user"}}

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        r1 = graph.invoke(
            {"messages": [HumanMessage("How many rows have intent get_refund?")], "iteration_count": 0},
            config=config,
        )
        assert r1["messages"][-1].content
        ids_after = r1.get("working_row_ids")
        print("Turn 1 done, working_row_ids:", len(ids_after) if ids_after else "full")

        r2 = graph.invoke(
            {"messages": [HumanMessage("How many rows are in the current filtered view?")], "iteration_count": 0},
            config=config,
        )
        print("Turn 2 answer:", r2["messages"][-1].content[:200])
        print("Session memory test OK")


def test_profile_file() -> None:
    save_profile("pytest_user", "- Name: Test User\n- Interests: refunds")
    assert "Test User" in load_profile("pytest_user")
    print("Profile file test OK")


if __name__ == "__main__":
    test_profile_file()
    test_session_memory()
    print("\nAll Task 2 tests passed.")
