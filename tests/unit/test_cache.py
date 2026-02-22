"""
Unit tests for the semantic cache module.

Uses a mocked Redis client to test cache logic (lookup, set, invalidation)
without requiring a running Redis server.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.cache.semantic_cache import SemanticCache, _cosine_similarity
from src.models import QueryResponse, SourceCitation


# ---------------------------------------------------------------------------
# Pure function tests -- no mocking needed
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert _cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# Cache behaviour tests with mocked Redis
# ---------------------------------------------------------------------------

def _make_response(query: str = "test query") -> QueryResponse:
    return QueryResponse(
        query=query,
        answer="The indemnification clause requires the Service Provider to hold harmless the Client.",
        citations=[
            SourceCitation(
                document_id="doc-001",
                chunk_id="chunk-001",
                filename="contract.pdf",
                excerpt="Service Provider shall indemnify...",
                relevance_score=0.9,
            )
        ],
    )


@pytest.fixture
def mock_redis():
    """Returns a mock Redis client with async methods."""
    r = AsyncMock()
    r.keys = AsyncMock(return_value=[])
    r.get = AsyncMock(return_value=None)
    r.setex = AsyncMock()
    r.sadd = AsyncMock()
    r.smembers = AsyncMock(return_value=set())
    r.delete = AsyncMock()
    return r


@pytest.fixture
def cache(mock_redis) -> SemanticCache:
    c = SemanticCache(similarity_threshold=0.95, ttl_seconds=3600)
    c._redis = mock_redis
    return c


@pytest.mark.asyncio
async def test_cache_miss_when_empty(cache, mock_redis):
    """Empty cache should always return None (cache miss)."""
    mock_redis.keys.return_value = []
    result = await cache.get([0.1, 0.2, 0.3])
    assert result is None


@pytest.mark.asyncio
async def test_cache_hit_above_threshold(cache, mock_redis):
    """Should return cached response when similarity >= threshold."""
    embedding = [1.0, 0.0]
    response = _make_response()

    # Simulate one stored embedding (identical = similarity 1.0)
    mock_redis.keys.return_value = ["semcache:embed:1000"]
    mock_redis.get.side_effect = [
        json.dumps(embedding),          # embedding lookup
        response.model_dump_json(),     # response lookup
    ]

    result = await cache.get(embedding)
    assert result is not None
    assert result.query == "test query"


@pytest.mark.asyncio
async def test_cache_miss_below_threshold(cache, mock_redis):
    """Should return None when best similarity is below the threshold."""
    stored_embedding = [1.0, 0.0]
    query_embedding = [0.0, 1.0]  # orthogonal -- similarity 0.0

    mock_redis.keys.return_value = ["semcache:embed:1000"]
    mock_redis.get.return_value = json.dumps(stored_embedding)

    result = await cache.get(query_embedding)
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_calls_redis(cache, mock_redis):
    """set() should write both the embedding and response to Redis."""
    embedding = [0.5, 0.5]
    response = _make_response()

    await cache.set(embedding, response)

    # Should have called setex twice (embed + response) and sadd once per citation
    assert mock_redis.setex.call_count == 2
    assert mock_redis.sadd.call_count == 1  # one citation = one doc


@pytest.mark.asyncio
async def test_cache_invalidation(cache, mock_redis):
    """invalidate_by_document should delete all cache entries for a document."""
    mock_redis.smembers.return_value = {"ts1", "ts2"}

    purged = await cache.invalidate_by_document("doc-001")

    assert purged == 2
    # Each cache_id has embed + response key = 4 delete calls, plus the index key
    assert mock_redis.delete.call_count >= 4
