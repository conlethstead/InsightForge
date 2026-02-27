"""
Schema discovery for any CSV/DataFrame.

Given a DataFrame or path, returns a typed schema: row count, column list with
per-column dtype and optional stats. Same shape regardless of which CSV is loaded.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd


def _dtype_kind(dtype) -> str:
    """Map pandas dtype to a simple category: numeric, datetime, object."""
    kind = getattr(dtype, "kind", None) or ""
    if kind in ("i", "u", "f"):
        return "numeric"
    if kind == "M" or (hasattr(dtype, "name") and "datetime" in str(dtype).lower()):
        return "datetime"
    return "object"


@dataclass
class ColumnStats:
    """Per-column stats; fields depend on dtype."""

    name: str
    dtype: str  # "numeric" | "datetime" | "object"
    null_count: int = 0
    # numeric
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    # datetime
    min_dt: str | None = None
    max_dt: str | None = None
    # categorical/object
    distinct_count: int | None = None
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "dtype": self.dtype, "null_count": self.null_count}
        if self.dtype == "numeric":
            if self.min_val is not None:
                d["min"] = self.min_val
            if self.max_val is not None:
                d["max"] = self.max_val
            if self.mean_val is not None:
                d["mean"] = self.mean_val
        elif self.dtype == "datetime":
            if self.min_dt is not None:
                d["min"] = self.min_dt
            if self.max_dt is not None:
                d["max"] = self.max_dt
        else:
            if self.distinct_count is not None:
                d["distinct_count"] = self.distinct_count
            if self.sample_values:
                d["sample_values"] = self.sample_values
        return d


@dataclass
class DataSchema:
    """
    Stricter typed schema: row count, columns (order preserved), per-column stats.
    schema_str for backward-compatible use in prompts.
    """

    row_count: int
    columns: list[str]
    column_stats: list[ColumnStats]
    schema_str: str  # comma-separated column names

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "columns": self.columns,
            "column_stats": [c.to_dict() for c in self.column_stats],
            "schema_str": self.schema_str,
        }

    def to_json(self) -> str:
        """Serialize for storage or retrieval."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataSchema:
        """Deserialize from dict (e.g. from JSON)."""
        stats = [
            ColumnStats(
                name=s["name"],
                dtype=s["dtype"],
                null_count=s.get("null_count", 0),
                min_val=s.get("min") if s.get("dtype") == "numeric" else None,
                max_val=s.get("max") if s.get("dtype") == "numeric" else None,
                mean_val=s.get("mean"),
                min_dt=str(s["min"]) if s.get("dtype") == "datetime" and "min" in s else None,
                max_dt=str(s["max"]) if s.get("dtype") == "datetime" and "max" in s else None,
                distinct_count=s.get("distinct_count"),
                sample_values=s.get("sample_values", []),
            )
            for s in d.get("column_stats", [])
        ]
        return cls(
            row_count=d.get("row_count", 0),
            columns=d.get("columns", []),
            column_stats=stats,
            schema_str=d.get("schema_str", ", ".join(d.get("columns", []))),
        )


def discover_schema(
    df: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> DataSchema:
    """
    Discover schema from a DataFrame or CSV path.

    Returns a DataSchema with row count, column list (order preserved),
    per-column dtype and stats (min/max/mean for numeric; min/max for datetime;
    distinct count and sample values for categorical).
    """
    if df is None:
        if path is None:
            from config import DATA_PATH
            path = DATA_PATH
        df = pd.read_csv(path)
    df = df.copy()
    n = len(df)
    columns = df.columns.tolist()
    column_stats: list[ColumnStats] = []
    for col in columns:
        s = df[col]
        dt = _dtype_kind(s.dtype)
        null_count = int(s.isna().sum())
        if dt == "numeric":
            column_stats.append(
                ColumnStats(
                    name=col,
                    dtype="numeric",
                    null_count=null_count,
                    min_val=float(s.min()) if s.notna().any() else None,
                    max_val=float(s.max()) if s.notna().any() else None,
                    mean_val=float(s.mean()) if s.notna().any() else None,
                )
            )
        elif dt == "datetime":
            try:
                s_dt = pd.to_datetime(s)
                mn, mx = s_dt.min(), s_dt.max()
                column_stats.append(
                    ColumnStats(
                        name=col,
                        dtype="datetime",
                        null_count=null_count,
                        min_dt=str(mn) if pd.notna(mn) else None,
                        max_dt=str(mx) if pd.notna(mx) else None,
                    )
                )
            except Exception:
                # fallback to object
                uniq = s.dropna().unique()
                sample = uniq[:10].tolist() if len(uniq) > 10 else uniq.tolist()
                column_stats.append(
                    ColumnStats(
                        name=col,
                        dtype="object",
                        null_count=null_count,
                        distinct_count=len(uniq),
                        sample_values=sample,
                    )
                )
        else:
            uniq = s.dropna().unique()
            sample = uniq[:10].tolist() if len(uniq) > 10 else uniq.tolist()
            column_stats.append(
                ColumnStats(
                    name=col,
                    dtype="object",
                    null_count=null_count,
                    distinct_count=len(uniq),
                    sample_values=sample,
                )
            )
    schema_str = ", ".join(columns)
    return DataSchema(
        row_count=n,
        columns=columns,
        column_stats=column_stats,
        schema_str=schema_str,
    )


def discover_schema_dict(
    df: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> dict:
    """
    Legacy: discover schema and return a dict compatible with old callers.
    Keys: columns, schema_str, dtypes (col -> "numeric"|"datetime"|"object"),
    row_count, column_stats (list of dicts).
    """
    schema = discover_schema(df=df, path=path)
    dtypes = {c.name: c.dtype for c in schema.column_stats}
    return {
        "columns": schema.columns,
        "schema_str": schema.schema_str,
        "dtypes": dtypes,
        "row_count": schema.row_count,
        "column_stats": [c.to_dict() for c in schema.column_stats],
    }
