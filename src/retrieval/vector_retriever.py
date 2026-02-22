"""
Dense vector retriever.

Queries Qdrant using the "dense" named vector (text-embedding-3-large).
Applies optional metadata payload filters (document type, client ID, date range)
before scoring, which prunes the candidate set before ANN search and thus
improves both speed and precision.

Why pre-filter?
  If a lawyer asks "show me indemnity clauses from 2022 contracts for client ABC",
  searching all 10,000 documents semantically then discarding 9,900 is wasteful.
  Qdrant's payload filters let us restrict the search space first.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

from src.config import settings
from src.models import Chunk, DocumentMetadata, DocumentType, QueryRequest


def _build_filter(request: QueryRequest) -> Filter | None:
    """Translate QueryRequest filter fields into a Qdrant Filter object."""
    conditions = []

    if request.filter_document_type:
        conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=request.filter_document_type.value),
            )
        )

    if request.filter_client_id:
        conditions.append(
            FieldCondition(
                key="client_id",
                match=MatchValue(value=request.filter_client_id),
            )
        )

    # Date range: stored as ISO-8601 string -- Qdrant supports range on strings
    if request.filter_date_from or request.filter_date_to:
        date_range: dict = {}
        if request.filter_date_from:
            date_range["gte"] = request.filter_date_from.isoformat()
        if request.filter_date_to:
            date_range["lte"] = request.filter_date_to.isoformat()
        conditions.append(
            FieldCondition(key="date", range=Range(**date_range))
        )

    if not conditions:
        return None

    return Filter(must=conditions)


async def _embed_query_async(query: str, client: AsyncOpenAI) -> list[float]:
    """Shared helper: embed a query string using the configured OpenAI model."""
    response = await client.embeddings.create(
        model=settings.openai_embedding_model,
        input=query,
    )
    return response.data[0].embedding


def _point_to_chunk(point) -> Chunk:
    """Reconstruct a Chunk from a Qdrant ScoredPoint payload."""
    payload = point.payload or {}
    metadata = DocumentMetadata(
        document_id=payload.get("document_id", ""),
        filename=payload.get("filename", ""),
        document_type=DocumentType(payload.get("document_type", "unknown")),
        client_id=payload.get("client_id"),
        parties=payload.get("parties", []),
    )
    return Chunk(
        chunk_id=payload.get("chunk_id", str(point.id)),
        document_id=payload.get("document_id", ""),
        text=payload.get("text", ""),
        chunk_index=payload.get("chunk_index", 0),
        metadata=metadata,
        # Use the Qdrant score as a proxy relevance signal
        embedding=[point.score] if point.score is not None else None,
    )


class VectorRetriever:
    """Retrieves chunks via dense ANN search in Qdrant."""

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def retrieve(self, request: QueryRequest) -> list[Chunk]:
        top_k = request.top_k or settings.retrieval_top_k
        query_vector = await _embed_query_async(request.query, self._openai)
        qdrant_filter = _build_filter(request)

        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        try:
            results = await client.query_points(
                collection_name=settings.qdrant_collection_name,
                query=query_vector,
                using="dense",
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        finally:
            await client.close()

        return [_point_to_chunk(p) for p in results.points]
