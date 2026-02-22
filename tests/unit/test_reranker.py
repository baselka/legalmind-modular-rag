"""
Unit tests for the re-ranker module.

Uses mocked Cohere/OpenAI clients to test re-ranking logic without API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.models import Chunk, DocumentMetadata, DocumentType
from src.retrieval.reranker import CohereReranker, get_reranker


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    meta = DocumentMetadata(
        filename="test.pdf",
        document_type=DocumentType.CONTRACT,
    )
    return [
        Chunk(
            chunk_id=f"chunk-{i:04d}-0000-0000-0000-000000000000",
            document_id="doc-0000-0000-0000-0000-000000000000",
            text=f"Legal clause {i}: This is relevant legal text number {i}.",
            chunk_index=i,
            metadata=meta,
        )
        for i in range(5)
    ]


@pytest.mark.asyncio
async def test_cohere_reranker_returns_top_n(sample_chunks):
    """CohereReranker should return exactly top_n results."""
    mock_result = MagicMock()
    mock_result.results = [
        MagicMock(index=2, relevance_score=0.95),
        MagicMock(index=0, relevance_score=0.88),
        MagicMock(index=4, relevance_score=0.71),
    ]

    with patch("src.retrieval.reranker.cohere") as mock_cohere:
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(return_value=mock_result)
        mock_cohere.AsyncClientV2.return_value = mock_client

        reranker = CohereReranker()
        reranker._client = mock_client
        results = await reranker.rerank("indemnity clause", sample_chunks, top_n=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_cohere_reranker_preserves_scores(sample_chunks):
    """The reranker should attach relevance scores to returned chunks."""
    mock_result = MagicMock()
    mock_result.results = [
        MagicMock(index=1, relevance_score=0.92),
        MagicMock(index=3, relevance_score=0.67),
    ]

    with patch("src.retrieval.reranker.cohere") as mock_cohere:
        mock_client = AsyncMock()
        mock_client.rerank = AsyncMock(return_value=mock_result)
        mock_cohere.AsyncClientV2.return_value = mock_client

        reranker = CohereReranker()
        reranker._client = mock_client
        results = await reranker.rerank("arbitration clause", sample_chunks, top_n=2)

    # Relevance score is stored in embedding[0]
    assert results[0].embedding is not None
    assert results[0].embedding[0] == pytest.approx(0.92)
    assert results[1].embedding[0] == pytest.approx(0.67)


@pytest.mark.asyncio
async def test_reranker_empty_input():
    """Passing an empty list of chunks should return an empty list."""
    reranker = CohereReranker()
    result = await reranker.rerank("query", [], top_n=5)
    assert result == []


def test_get_reranker_returns_cohere_by_default():
    """Factory function should return CohereReranker when reranker_type=cohere."""
    with patch("src.retrieval.reranker.settings") as mock_settings:
        mock_settings.reranker_type = "cohere"
        mock_settings.cohere_rerank_model = "rerank-v3.5"
        mock_settings.cohere_api_key = "test-key"
        reranker = get_reranker()
    assert isinstance(reranker, CohereReranker)
