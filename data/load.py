"""
Load raw dataset from config path or optional path.

Reads the CSV at config.DATA_PATH when path is None; otherwise uses the given path.
Run from project root: python data/load.py  (or python -m data.load)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so "config" can be found when running
# python data/load.py (script dir is data/, not project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from config import DATA_PATH


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV from path; if path is None, use config.DATA_PATH."""
    p = Path(path) if path is not None else DATA_PATH
    return pd.read_csv(p)


if __name__ == "__main__":
    # (1) Load and print shape
    df = load_raw_data()
    print("shape:", df.shape)

    # (2) Print column list
    print("columns:", list(df.columns))
