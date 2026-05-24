#!/usr/bin/env python3
"""Bonus A: Streamlit chat UI for the Bitext data analyst agent."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv(os.path.join(_ROOT, "..", "..", ".env"))

import streamlit as st

from src.agent_runner import ReasoningStep, run_turn
from src.config import CHECKPOINT_PATH, DATA_DIR
from src.data_loader import load_bitext_dataframe
from src.dataset_context import set_full_dataset
from src.graph import build_graph
from src.query_recommender import RecommenderState


@st.cache_resource
def load_dataset():
    df = load_bitext_dataframe()
    set_full_dataset(df)
    return len(df)


@st.cache_resource
def get_graph():
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(checkpointer=checkpointer)


def _render_reasoning(steps: list[ReasoningStep]) -> None:
    if not steps:
        return
    with st.expander("Reasoning steps", expanded=True):
        for step in steps:
            if step.kind == "router":
                st.markdown(f"**Router** — {step.text}")
            elif step.kind == "tool_call":
                st.code(step.text, language=None)
            elif step.kind == "observation":
                st.text(step.text)
            elif step.kind == "recommendation":
                st.info(step.text)
            elif step.kind == "profile":
                st.markdown(step.text)
            else:
                st.markdown(step.text)


def main() -> None:
    st.set_page_config(page_title="Bitext Analyst", page_icon="💬", layout="wide")
    st.title("Bitext Customer Support Analyst")
    st.caption("Assignment 3 — Streamlit UI (Bonus A) with query recommender (Bonus B)")

    if not os.environ.get("NEBIUS_API_KEY"):
        st.error("Set NEBIUS_API_KEY in `.env` before running the app.")
        st.stop()

    row_count = load_dataset()
    graph = get_graph()

    with st.sidebar:
        st.header("Session")
        session_id = st.text_input("Session ID", value=st.session_state.get("session_id", "default"))
        user_id = st.text_input("User ID (profile)", value=st.session_state.get("user_id", session_id))
        st.session_state["session_id"] = session_id
        st.session_state["user_id"] = user_id
        st.caption(f"Dataset: {row_count:,} rows · checkpoints persist per Session ID")
        if st.button("Clear chat display"):
            st.session_state["chat_messages"] = []
            st.session_state["recommender"] = RecommenderState()
            st.rerun()

        st.divider()
        st.markdown("**Try Bonus B:** ask *What should I query next?*")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "recommender" not in st.session_state:
        st.session_state["recommender"] = RecommenderState()

    for role, content in st.session_state["chat_messages"]:
        with st.chat_message(role):
            st.markdown(content)

    config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        }
    }

    if prompt := st.chat_input("Ask about the Bitext dataset…"):
        st.session_state["chat_messages"].append(("user", prompt))

        with st.chat_message("user"):
            st.markdown(prompt)

        snapshot = graph.get_state(config)
        history = list(snapshot.values.get("messages", [])) if snapshot.values else []

        with st.spinner("Thinking…"):
            result = run_turn(
                graph,
                config,
                prompt,
                recommender=st.session_state["recommender"],
                history_messages=history,
            )

        st.session_state["recommender"] = result.recommender

        with st.chat_message("assistant"):
            st.markdown(result.final_answer)
            _render_reasoning(result.reasoning)

        st.session_state["chat_messages"].append(("assistant", result.final_answer))


if __name__ == "__main__":
    main()
