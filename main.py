#!/usr/bin/env python3
"""CLI entry point for the Bitext data analyst agent."""

import os
import sys

from dotenv import load_dotenv

# Ensure project root is on path when running as `python main.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_root = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_root, ".env"))
# Workspace root: Nebius Assignments/.env (two levels up from bitext-agent/)
load_dotenv(os.path.join(_root, "..", "..", ".env"))

if not os.environ.get("NEBIUS_API_KEY"):
    print("Error: Set NEBIUS_API_KEY in .env (see .env.example).", file=sys.stderr)
    sys.exit(1)

from src.cli import main

if __name__ == "__main__":
    main()
