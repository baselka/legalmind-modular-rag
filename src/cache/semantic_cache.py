"""
Redis-backed semantic cache.

Design:
- Each cache entry stores a serialised QueryResponse alongside the query embedding.
- On lookup, we compute cosine similarity between the incoming query embedding
  and all stored embeddings. A hit is declared when similarity >= threshold.
- Cache entries are tagged with their source document IDs so they can be
  invalidated when a document is re-ingested.

Why not exact-match caching?
  Legal queries are rarely word-for-word identical.  "Show me the indemnity
  clause" and "What are the indemnification obligations?" should hit the same
  cache entry.  Embedding similarity generalises across paraphrases.
"""

from __future__ import annotations

import json
import time

import numpy as np
import redis.asyncio as aioredis

from src.config import settings
from src.models import QueryResponse


_EMBED_PREFIX = "semcache:embed:"
_RESP_PREFIX = "semcache:resp:"
_DOC_INDEX_PREFIX = "semcache:doc:"  # set of cache keys that reference a doc


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class SemanticCache:
    """Async Redis-backed semantic cache."""

    def __init__(
        self,
        redis_url: str | None = None,
        similarity_threshold: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._url = redis_url or settings.redis_url
        self._threshold = similarity_threshold or settings.semantic_cache_similarity_threshold
        self._ttl = ttl_seconds or settings.redis_cache_ttl_seconds
        self._redis: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def get(
        self,
        query_embedding: list[float],
    ) -> QueryResponse | None:
        """Return a cached response if a similar query embedding exists."""
        client = await self._get_client()

        # Scan all stored embeddings and find the best match.
        # For production scale, replace with Redis Vector Similarity Search (VSS).
        keys = await client.keys(f"{_EMBED_PREFIX}*")
        best_sim = -1.0
        best_key = None

        for key in keys:
            raw = await client.get(key)
            if raw is None:
                continue
            stored_embed: list[float] = json.loads(raw)
            sim = _cosine_similarity(query_embedding, stored_embed)
            if sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is None or best_sim < self._threshold:
            return None

        # Derive the response key from the embedding key
        cache_id = best_key[len(_EMBED_PREFIX):]
        resp_raw = await client.get(f"{_RESP_PREFIX}{cache_id}")
        if resp_raw is None:
            return None

        return QueryResponse.model_validate_json(resp_raw)

    async def set(
        self,
        query_embedding: list[float],
        response: QueryResponse,
    ) -> None:
        """Store a response alongside its query embedding."""
        client = await self._get_client()
        cache_id = str(int(time.time() * 1000))

        await client.setex(
            f"{_EMBED_PREFIX}{cache_id}",
            self._ttl,
            json.dumps(query_embedding),
        )
        await client.setex(
            f"{_RESP_PREFIX}{cache_id}",
            self._ttl,
            response.model_dump_json(),
        )

        # Index by source document IDs for targeted invalidation
        for citation in response.citations:
            await client.sadd(f"{_DOC_INDEX_PREFIX}{citation.document_id}", cache_id)

    async def invalidate_by_document(self, document_id: str) -> int:
        """
        Purge all cache entries that reference *document_id*.
        Called when a document is re-ingested.
        Returns the number of entries purged.
        """
        client = await self._get_client()
        doc_key = f"{_DOC_INDEX_PREFIX}{document_id}"
        cache_ids: set[str] = await client.smembers(doc_key)

        purged = 0
        for cache_id in cache_ids:
            await client.delete(f"{_EMBED_PREFIX}{cache_id}")
            await client.delete(f"{_RESP_PREFIX}{cache_id}")
            purged += 1

        await client.delete(doc_key)
        return purged

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
