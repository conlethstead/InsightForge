"""
Custom retriever: query -> relevant chunks via semantic search over Documents.

When chunks are provided (e.g. from session), embeds query and chunks and returns
top-k by cosine similarity. When chunks is None, builds chunks via get_chunks(path=path)
and runs semantic retrieval (no keyword routing or fixed aggregations).
Compatible with LangChain BaseRetriever for RAG.

Run from project root: python retrieval/retriever.py  or  python -m retrieval.retriever
"""

import logging
import sys
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("insightforge.retriever")
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from data.knowledge_base import get_chunks


def _default_embed_fn():
    """Return default embedding function using OpenAI (requires config.PAID_OPENAI_API_KEY)."""
    import config
    from langchain_openai import OpenAIEmbeddings
    if not config.has_api_key():
        raise ValueError("PAID_OPENAI_API_KEY is not set. Set it in .zshrc or .env.")
    emb = OpenAIEmbeddings(api_key=config.PAID_OPENAI_API_KEY, model="text-embedding-3-small")
    def _embed(texts: list[str]):
        return emb.embed_documents(texts)
    return _embed


def _cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and each row of B. Returns 1d array."""
    a = np.asarray(a, dtype=float).ravel()
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    dot = B @ a
    norm_a = np.linalg.norm(a)
    norm_B = np.linalg.norm(B, axis=1)
    norms = norm_a * norm_B
    norms = np.where(norms > 0, norms, 1.0)
    return dot / norms


def retrieve(
    query: str,
    chunks: list[Document] | None = None,
    path: str | Path | None = None,
    top_k: int = 10,
    embed_fn: Callable[..., list] | None = None,
) -> list[Document]:
    """
    Return top-k chunks most similar to the query (semantic retrieval).

    Args:
        query: User or internal query string.
        chunks: Prebuilt list of Documents (e.g. from session). If None, get_chunks(path=path) is used.
        path: Optional CSV path for get_chunks when chunks is None; if both None, config default.
        top_k: Max number of documents to return.
        embed_fn: Callable(list[str]) -> list of embedding vectors. If None, uses OpenAI embeddings.

    Returns:
        List of Document instances, ordered by relevance (highest first).
    """
    # Step 1: Input
    logger.info("[retrieve] step=input query=%r top_k=%d chunks_provided=%s path=%s", query, top_k, chunks is not None, path)
    if chunks is None:
        chunks = get_chunks(path=path)
        logger.info("[retrieve] step=chunks_built num_chunks=%d (from path/config)", len(chunks))
    else:
        logger.info("[retrieve] step=chunks_provided num_chunks=%d", len(chunks))
    if not chunks:
        logger.info("[retrieve] step=output num_results=0 (no chunks)")
        return []

    if embed_fn is None:
        embed_fn = _default_embed_fn()

    # Step 2: Embed query and chunks
    texts = [doc.page_content for doc in chunks]
    query_emb = np.array(embed_fn([query.strip()])[0])
    chunk_embs = np.array(embed_fn(texts))
    logger.info("[retrieve] step=embed query_embed_dim=%d chunk_embeddings=%d", len(query_emb), len(chunk_embs))

    # Step 3: Similarity and top-k
    scores = _cosine_similarity(query_emb, chunk_embs)
    indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[indices]
    logger.info("[retrieve] step=similarity top_scores=%s", [f"{s:.4f}" for s in top_scores])

    results = [chunks[i] for i in indices]
    # Step 4: Output
    logger.info("[retrieve] step=output num_results=%d", len(results))
    for i, doc in enumerate(results[:5]):
        preview = (doc.page_content[:100] + "…") if len(doc.page_content) > 100 else doc.page_content
        logger.info("[retrieve] step=output result[%d] preview=%s", i, preview)
    if len(results) > 5:
        logger.info("[retrieve] step=output ... and %d more results", len(results) - 5)
    return results


class InsightForgeRetriever(BaseRetriever):
    """
    LangChain BaseRetriever that delegates to retrieve() for RAG compatibility.
    """

    top_k: int = 10
    chunks: list[Document] | None = None
    path: str | Path | None = None
    embed_fn: Callable[..., list] | None = None

    def _get_relevant_documents(self, query: str, **kwargs: object) -> list[Document]:
        return retrieve(
            query,
            chunks=self.chunks,
            path=self.path,
            top_k=self.top_k,
            embed_fn=self.embed_fn,
        )


if __name__ == "__main__":
    # Semantic retrieval with default CSV and embeddings (requires API key)
    try:
        docs1 = retrieve("sales by region", top_k=5)
        print("Query: 'sales by region'")
        print("Number of results:", len(docs1))
        if docs1:
            preview = docs1[0].page_content[:150] + ("..." if len(docs1[0].page_content) > 150 else "")
            print("First doc preview:", preview)
    except ValueError as e:
        print("Error (e.g. missing API key):", e)
    print()

    try:
        docs2 = retrieve("top products", top_k=5)
        print("Query: 'top products'")
        print("Number of results:", len(docs2))
        if docs2:
            preview = docs2[0].page_content[:150] + ("..." if len(docs2[0].page_content) > 150 else "")
            print("First doc preview:", preview)
    except ValueError as e:
        print("Error:", e)
