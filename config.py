"""
InsightForge — environment and app config.

Reads PAID_OPENAI_API_KEY and DATA_PATH from the environment.
API key can be set in .zshrc (export PAID_OPENAI_API_KEY=...) or in a .env file.
"""

import os
import warnings
from pathlib import Path

# LangChain still uses Pydantic v1 internally; on Python 3.13+ this raises a
# UserWarning. Suppress it so the console stays clean until the ecosystem moves to v2.
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic V1.*",
    category=UserWarning,
    module="langchain_core.*",
)

# Load .env if present (optional; key may already be in shell env e.g. from .zshrc)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Project root (directory containing config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Path to the sales dataset (sales_data.csv)
DATA_PATH: Path = PROJECT_ROOT / "Dataset" / "sales_data.csv"

# OpenAI API key — must be set in environment (e.g. export in .zshrc or set in .env)
PAID_OPENAI_API_KEY: str | None = os.environ.get("PAID_OPENAI_API_KEY")


def has_api_key() -> bool:
    """Return True if PAID_OPENAI_API_KEY is set and non-empty."""
    return bool(PAID_OPENAI_API_KEY and PAID_OPENAI_API_KEY.strip())


if __name__ == "__main__":
    # (1) Print DATA_PATH
    print("DATA_PATH:", DATA_PATH)
    print("exists:", DATA_PATH.exists())

    # (2) Confirm PAID_OPENAI_API_KEY is set without printing the key
    if has_api_key():
        print("PAID_OPENAI_API_KEY: present")
    else:
        print("PAID_OPENAI_API_KEY: missing")
