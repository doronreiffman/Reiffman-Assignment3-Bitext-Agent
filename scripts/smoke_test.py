#!/usr/bin/env python3
"""Smoke tests for Task 1 (tools offline + agent with API)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, ".env"))
load_dotenv(os.path.join(_root, "..", "..", ".env"))  # Nebius Assignments/.env

from langchain_core.messages import HumanMessage

from src.data_loader import load_bitext_dataframe
from src.checkpointer import get_checkpointer
from src.graph import build_graph
from src.dataset_context import set_full_dataset


def test_tools_offline() -> None:
    print("=== Offline tool tests ===")
    df = load_bitext_dataframe()
    set_full_dataset(df)

    from src.dataset_context import set_working_from_df
    from src.tools.dataset_tools import count_rows, list_categories, list_intents

    cats = list_categories.invoke({})
    assert "REFUND" in cats, cats
    print("list_categories: OK")

    intents = list_intents.invoke({"category": "ACCOUNT"})
    assert "create_account" in intents, intents
    print("list_intents(ACCOUNT): OK")

    set_working_from_df(df[df["intent"] == "get_refund"])
    n = count_rows.invoke({})
    assert int(n) > 0, n
    print(f"count_rows on get_refund filter: {n} rows OK")


def test_agent_queries() -> None:
    if not os.environ.get("NEBIUS_API_KEY"):
        print("Skipping API tests (no NEBIUS_API_KEY)")
        return

    print("\n=== Agent API tests ===")
    df = load_bitext_dataframe()
    set_full_dataset(df)
    config = {"configurable": {"thread_id": "smoke_test", "user_id": "smoke_test"}}

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        cases = [
            ("What categories exist in the dataset?", "structured"),
            ("Who is the president of France?", "out_of_scope"),
        ]

        for query, expected_route in cases:
            print(f"\nQuery: {query}")
            result = graph.invoke(
                {"messages": [HumanMessage(content=query)], "iteration_count": 0},
                config=config,
            )
            qtype = result.get("query_type")
            print(f"  route={qtype} (expected {expected_route})")
            assert qtype == expected_route, f"got {qtype}"
            last = result["messages"][-1]
            print(f"  answer preview: {str(last.content)[:120]}...")

        print("\nMulti-step query: How many refund requests did we get?")
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(content="How many refund requests did we get?")
                ],
                "iteration_count": 0,
            },
            config=config,
        )
        print(f"  route={result.get('query_type')}")
        print(f"  answer: {result['messages'][-1].content[:300]}")


if __name__ == "__main__":
    test_tools_offline()
    test_agent_queries()
    print("\nAll smoke tests passed.")
