"""Interactive CLI with visible reasoning steps and session memory."""

from __future__ import annotations

import argparse

from src.agent_runner import ReasoningStep, run_turn
from src.checkpointer import get_checkpointer
from src.data_loader import load_bitext_dataframe
from src.dataset_context import set_full_dataset
from src.graph import build_graph
from src.query_recommender import RecommenderState
from src.user_profile import load_profile


def _print_reasoning(steps: list[ReasoningStep]) -> None:
    for step in steps:
        if step.kind == "router":
            print(f"\n[router] {step.text}")
        elif step.kind == "tool_call":
            print(f"  [tool call] {step.text}")
        elif step.kind == "observation":
            print(f"  [observation] {step.text}")
        elif step.kind == "recommendation":
            print(f"\n[recommendation] {step.text}")
        elif step.kind == "profile":
            print(f"\n[profile] answer:\n{step.text}")
        elif step.kind == "final":
            print("\n[agent] final answer:")
            print(step.text)
        elif step.kind == "agent":
            print(f"\n[agent] {step.text}")


def run_interactive(session_id: str, user_id: str) -> None:
    """Run the REPL with persistent session and user profile."""
    print("Loading dataset (first run may download from Hugging Face)...")
    df = load_bitext_dataframe()
    set_full_dataset(df)
    print(f"Dataset ready: {len(df)} rows.")
    print(f"Session: {session_id} | User profile: {user_id}")
    profile = load_profile(user_id)
    if profile and profile != "No profile information yet.":
        print(f"Profile loaded ({len(profile)} chars).\n")
    else:
        print("No saved profile yet.\n")

    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        }
    }

    recommender = RecommenderState()

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        print("Bitext Data Analyst Agent")
        print("Ask about the dataset. Bonus B: try 'What should I query next?'")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye.")
                break

            print("\n--- reasoning ---")
            snapshot = graph.get_state(config)
            history = list(snapshot.values.get("messages", [])) if snapshot.values else []

            result = run_turn(
                graph,
                config,
                user_input,
                recommender=recommender,
                history_messages=history,
            )
            recommender = result.recommender
            _print_reasoning(result.reasoning)
            if result.final_answer and not any(s.kind == "final" for s in result.reasoning):
                print("\n[agent] final answer:")
                print(result.final_answer)
            print("--- end ---\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitext customer support data analyst agent")
    parser.add_argument(
        "--session",
        default="default",
        help="Conversation thread ID (restored across restarts). Same as thread_id in LangGraph.",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User ID for long-term profile file (defaults to --session).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_id = args.user or args.session
    run_interactive(session_id=args.session, user_id=user_id)
