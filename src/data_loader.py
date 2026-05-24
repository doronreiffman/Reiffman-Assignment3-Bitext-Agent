"""Load and cache the Bitext customer support dataset."""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

from src.config import CACHE_PATH, DATA_DIR, DATASET_ID


def load_bitext_dataframe() -> pd.DataFrame:
    """Load the dataset, using a local parquet cache when available."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
    else:
        dataset = load_dataset(DATASET_ID, split="train")
        df = dataset.to_pandas()
        df.to_parquet(CACHE_PATH, index=False)

    return df
