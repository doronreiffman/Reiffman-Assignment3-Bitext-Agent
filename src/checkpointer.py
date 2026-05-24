"""SQLite checkpointer for persistent conversation threads."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import CHECKPOINT_PATH, DATA_DIR


@contextmanager
def get_checkpointer() -> Iterator[SqliteSaver]:
    """Open a SqliteSaver backed by a file under data/ (survives restarts)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_PATH), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()
