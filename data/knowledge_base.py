"""
Build and expose chunked documents from any CSV for retrieval.

Schema-agnostic: overview chunk(s) + optional row/summary chunks using actual
column names and dtypes. No vector DB; in-memory list of LangChain Documents.
Run from project root: python data/knowledge_base.py  or  python -m data.knowledge_base
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: F401 — load first so Pydantic v1 warning filter is applied
import pandas as pd
from langchain_core.documents import Document

from data.load import load_raw_data
from data.schema import DataSchema, discover_schema


def get_chunks(
    df: pd.DataFrame | None = None,
    path: str | Path | None = None,
) -> list[Document]:
    """
    Build document chunks from raw data for RAG retrieval (schema-agnostic).

    Produces (1) overview chunk(s) describing columns, row count, and per-column
    stats (min/max for numeric, sample values for object); (2) optional summary
    chunks using first categorical + first numeric for "breakdown by X" retrieval.

    Args:
        df: Optional DataFrame; if None and path is None, loads via load_raw_data().
        path: Optional path to CSV; used when df is None.

    Returns:
        List of Document instances with page_content and metadata (type: overview | sample).
    """
    if df is None:
        df = load_raw_data(path=path)

    df = df.copy()
    schema = discover_schema(df=df)
    columns = schema.columns
    chunks: list[Document] = []

    # --- Overview chunk(s) ---
    overview_parts = [
        f"Columns: {schema.schema_str}. Row count: {schema.row_count}.",
    ]
    for col_stat in schema.column_stats:
        col = col_stat.name
        dt = col_stat.dtype
        if dt == "numeric":
            mean_s = f"{col_stat.mean_val:.2f}" if col_stat.mean_val is not None else "n/a"
            overview_parts.append(
                f"Column {col} (numeric): min={col_stat.min_val}, max={col_stat.max_val}, mean={mean_s}."
            )
        elif dt == "datetime":
            overview_parts.append(
                f"Column {col} (datetime): range {col_stat.min_dt} to {col_stat.max_dt}."
            )
        else:
            sample = col_stat.sample_values[:10] if col_stat.sample_values else []
            overview_parts.append(f"Column {col} (categorical/object): values {sample}.")

    chunks.append(
        Document(
            page_content=" ".join(overview_parts),
            metadata={"type": "overview"},
        )
    )

    # --- Optional summary chunks: first datetime -> period; first categorical + first numeric ---
    dtypes_map = {c.name: c.dtype for c in schema.column_stats}
    dt_cols = [c for c in columns if dtypes_map.get(c) == "datetime"]
    cat_cols = [c for c in columns if dtypes_map.get(c) == "object"]
    num_cols = [c for c in columns if dtypes_map.get(c) == "numeric"]

    if dt_cols and (cat_cols or num_cols):
        time_col = dt_cols[0]
        df_per = df.copy()
        df_per[time_col] = pd.to_datetime(df_per[time_col])
        df_per["_period"] = df_per[time_col].dt.to_period("M").astype(str)
        for period, grp in df_per.groupby("_period"):
            parts = [f"Period {period}: rows={len(grp)}."]
            if num_cols:
                for col in num_cols[:2]:
                    parts.append(f"{col}: sum={grp[col].sum():.2f}, mean={grp[col].mean():.2f}.")
            if cat_cols:
                for col in cat_cols[:2]:
                    by_val = grp[col].value_counts()
                    parts.append(f"By {col}: {dict(by_val.head(5))}.")
            chunks.append(
                Document(
                    page_content=" ".join(parts),
                    metadata={"type": "sample", "period": period},
                )
            )

    if cat_cols and num_cols and not dt_cols:
        cat_col, num_col = cat_cols[0], num_cols[0]
        for val, grp in df.groupby(cat_col):
            total = grp[num_col].sum()
            chunks.append(
                Document(
                    page_content=f"{cat_col}={val}: {num_col} sum={total:.2f}, count={len(grp)}.",
                    metadata={"type": "sample", cat_col: val},
                )
            )

    # No sample row chunks: retrieval is dominated by overview and period/summary
    # so schema/column-level context is prioritized over raw rows.

    return chunks


if __name__ == "__main__":
    # (1) Call get_chunks(), print number of chunks
    chunks = get_chunks()
    print("number of chunks:", len(chunks))
    # (2) Print one sample chunk (content + metadata)
    sample = chunks[0]
    content_preview = sample.page_content[:200] + ("..." if len(sample.page_content) > 200 else "")
    print("sample chunk content:", content_preview)
    print("sample chunk metadata:", sample.metadata)
