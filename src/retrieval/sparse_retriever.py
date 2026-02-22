"""
Sparse (keyword) retriever using Qdrant's built-in multilingual full-text index.

Replaces SPLADE (English-only learned sparse vectors) with Qdrant's native
BM25-style full-text search backed by a multilingual tokenizer.

Why this is better for Arabic legal documents:
  SPLADE was trained on English MS-MARCO data.  For Arabic queries, it either
  generates meaningless sparse vectors or falls back to a hash-based dummy.
  Qdrant's multilingual tokenizer handles Arabic (including tashkeel), Hebrew,
  CJK, and Latin scripts natively -- no model weights, no download.

How it works:
  1. During ingestion, `ensure_collection_exists` creates a TextIndexParams
     payload index on the "text" field using TokenizerType.MULTILINGUAL.
     Qdrant builds an inverted index over the existing payload data.
  2. At query time, a MatchText filter finds all chunks whose "text" payload
     contains the query terms.  Qdrant tokenizes both the stored text and the
     query using the same multilingual tokenizer, enabling exact and stem matches.
  3. The matched chunks are returned (unranked by keyword score -- ranking
     is handled by the RRF step in HybridRetriever that combines this result
     set with the dense ANN results).

Note on ranking:
  Qdrant's MatchText filter returns matching chunks in arbitrary order
  (not by BM25 score). Final relevance ranking is achieved by RRF in
  HybridRetriever, which combines ranks from both the dense and this leg.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText

from src.config import settings
from src.models import Chunk, DocumentMetadata, DocumentType, QueryRequest
from src.retrieval.vector_retriever import _build_filter


def _record_to_chunk(record) -> Chunk:
    """Reconstruct a Chunk from a Qdrant scroll Record (no score field)."""
    payload = record.payload or {}
    metadata = DocumentMetadata(
        document_id=payload.get("document_id", ""),
        filename=payload.get("filename", ""),
        document_type=DocumentType(payload.get("document_type", "unknown")),
        client_id=payload.get("client_id"),
        parties=payload.get("parties", []),
    )
    return Chunk(
        chunk_id=payload.get("chunk_id", str(record.id)),
        document_id=payload.get("document_id", ""),
        text=payload.get("text", ""),
        chunk_index=payload.get("chunk_index", 0),
        metadata=metadata,
    )


class SparseRetriever:
    """
    Retrieves chunks via Qdrant's built-in multilingual full-text search.
    Uses MatchText filter on the indexed "text" payload field.
    """

    async def retrieve(self, request: QueryRequest) -> list[Chunk]:
        top_k = request.top_k or settings.retrieval_top_k

        # Build MatchText condition for the query text
        text_condition = FieldCondition(
            key="text",
            match=MatchText(text=request.query),
        )

        # Merge with metadata filters (doc_type, client_id, date range)
        base_filter = _build_filter(request)
        if base_filter is not None:
            must_conditions = list(base_filter.must or []) + [text_condition]
            scroll_filter = Filter(must=must_conditions)
        else:
            scroll_filter = Filter(must=[text_condition])

        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        try:
            results, _ = await client.scroll(
                collection_name=settings.qdrant_collection_name,
                scroll_filter=scroll_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            return [_record_to_chunk(r) for r in results]
        finally:
            await client.close()
