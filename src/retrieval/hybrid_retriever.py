"""
Hybrid retriever: combines dense vector search + Qdrant full-text search via
client-side Reciprocal Rank Fusion (RRF).

Architecture change from SPLADE → Qdrant built-in BM25:
  Previously: Qdrant server-side RRF using a `prefetch` with a SPLADE sparse
  vector query.  This required pre-computing SPLADE embeddings (English-only).

  Now: Two independent async calls run in parallel --
    - Dense leg:   query_points with the OpenAI dense vector → ranked list
    - Keyword leg: scroll with Qdrant MatchText filter → unranked match list
  Results are merged in Python using RRF.

Why client-side RRF here?
  Qdrant's server-side `prefetch` fusion accepts only vector queries (dense or
  sparse SparseVector).  MatchText is a payload filter condition, not a vector
  query, so it cannot be placed inside `prefetch`.  Client-side RRF is the
  correct approach for mixing vector and text search result sets.

RRF formula:
  score(chunk) = Σ  1 / (k + rank_in_result_set)
  where k=2 dampens the impact of very high ranks.
  A chunk appearing at rank 1 in both legs gets the maximum combined score.
  A chunk appearing in only one leg still contributes via that leg's rank.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from src.config import settings
from src.models import Chunk, QueryRequest
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.vector_retriever import VectorRetriever


def _client_side_rrf(
    dense_results: list[Chunk],
    keyword_results: list[Chunk],
    k: int = 2,
) -> list[Chunk]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns a deduplicated list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(keyword_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank + 1)
        chunk_map[chunk.chunk_id] = chunk

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_map[cid] for cid in sorted_ids]


class HybridRetriever:
    """
    Combines dense ANN search and Qdrant full-text (BM25-style) search via RRF.

    Both retrieval legs run concurrently with asyncio.gather for minimal latency.
    """

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._dense = VectorRetriever()
        self._keyword = SparseRetriever()

    async def retrieve(self, request: QueryRequest) -> list[Chunk]:
        top_k = request.top_k or settings.retrieval_top_k

        # Run both legs concurrently
        dense_results, keyword_results = await asyncio.gather(
            self._dense.retrieve(request),
            self._keyword.retrieve(request),
        )

        merged = _client_side_rrf(dense_results, keyword_results)
        return merged[:top_k]
