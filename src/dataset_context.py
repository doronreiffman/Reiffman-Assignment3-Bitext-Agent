"""Mutable working view for dataset tools (synced to graph state after tool runs)."""

from __future__ import annotations

import pandas as pd

_FULL_DF: pd.DataFrame | None = None
_WORKING_ROW_IDS: list[int] | None = None


def set_full_dataset(df: pd.DataFrame) -> None:
    global _FULL_DF, _WORKING_ROW_IDS
    _FULL_DF = df.reset_index(drop=True)
    _WORKING_ROW_IDS = None


def reset_working() -> None:
    global _WORKING_ROW_IDS
    _WORKING_ROW_IDS = None


def set_working_from_df(df: pd.DataFrame) -> None:
    """Store the current view as row indices into the full dataset."""
    global _WORKING_ROW_IDS
    if df is None or df.empty:
        _WORKING_ROW_IDS = []
    else:
        _WORKING_ROW_IDS = df.index.tolist()


def get_working_row_ids() -> list[int] | None:
    return _WORKING_ROW_IDS


def current_df() -> pd.DataFrame:
    if _FULL_DF is None:
        raise RuntimeError("Dataset not loaded.")
    if _WORKING_ROW_IDS is not None:
        if not _WORKING_ROW_IDS:
            return _FULL_DF.iloc[0:0]
        return _FULL_DF.loc[_WORKING_ROW_IDS]
    return _FULL_DF


def sync_row_ids_from_state(working_row_ids: list[int] | None) -> None:
    """Restore tool context from checkpointed graph state."""
    global _WORKING_ROW_IDS
    _WORKING_ROW_IDS = working_row_ids
