"""Application configuration."""

from pathlib import Path

# Nebius Token Factory endpoint
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Chosen after trying smaller models; 70B balances cost vs tool-calling quality.
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# Agent loop: assignment suggests 10–15 iterations
MAX_ITERATIONS = 12

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_PATH = DATA_DIR / "bitext.parquet"
CHECKPOINT_PATH = DATA_DIR / "checkpoints.sqlite"
PROFILES_DIR = DATA_DIR / "profiles"

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
