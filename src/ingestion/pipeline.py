"""
Ingestion pipeline orchestrator.

Coordinates:
  1. PDF parsing      (parser.py)
  2. Metadata enrichment  (enricher.py)
  3. Semantic chunking    (chunker.py)
  4. Embedding generation (OpenAI text-embedding-3-large for dense,
                           FastEmbed SPLADE for sparse)
  5. Upsert into Qdrant   (both vector types in a single collection)

The pipeline is intentionally decoupled from the API layer -- it can be
invoked from a CLI script, a background task queue, or an HTTP handler.

Qdrant collection schema
------------------------
Each point in the collection stores:
  - id:       chunk_id (UUID string, stored as Qdrant UUID)
  - vectors:
      "dense":  3072-dim float32  (text-embedding-3-large)
      "sparse": SparseVector       (SPLADE via FastEmbed)
  - payload:  all fields from Chunk + DocumentMetadata (for filtering)
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import structlog
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    TextIndexParams,
    TokenizerType,
    VectorParams,
    VectorsConfig,
    SparseVectorsConfig,
)

from src.config import settings
from src.ingestion.chunker import chunk_document
from src.ingestion.enricher import enrich_metadata
from src.ingestion.parser import extract_text_from_bytes, extract_text_from_pdf
from src.models import Chunk, DocumentMetadata

log = structlog.get_logger(__name__)

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"
_EMBED_BATCH_SIZE = 100  # OpenAI batch limit


async def _get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=settings.qdrant_timeout,
    )


async def ensure_collection_exists(client: AsyncQdrantClient) -> None:
    """
    Create the Qdrant collection if it doesn't already exist.
    Uses a single collection with both dense and sparse named vectors.
    """
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}

    if settings.qdrant_collection_name not in existing:
        await client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config={
                _DENSE_VECTOR_NAME: VectorParams(
                    size=3072,  # text-embedding-3-large dimension
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        log.info("qdrant_collection_created", name=settings.qdrant_collection_name)

    # Ensure multilingual full-text index exists on the "text" payload field.
    # This is idempotent — safe to call on both new and existing collections.
    # The index enables Qdrant's built-in BM25-style keyword search for Arabic
    # and other languages without needing an external tokenizer (e.g. SPLADE).
    try:
        await client.create_payload_index(
            collection_name=settings.qdrant_collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )
        log.info("text_index_ensured", field="text", tokenizer="multilingual")
    except Exception as exc:
        log.warning("text_index_creation_skipped", error=str(exc))


async def _embed_dense(texts: list[str]) -> list[list[float]]:
    """Generate dense embeddings via OpenAI in batches."""
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i : i + _EMBED_BATCH_SIZE]
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings


def _embed_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """
    Generate sparse vectors via FastEmbed (SPLADE model).
    Returns a list of (indices, values) tuples.
    """
    try:
        from fastembed import SparseTextEmbedding

        model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        results = []
        for embedding in model.embed(texts):
            indices = embedding.indices.tolist()
            values = embedding.values.tolist()
            results.append((indices, values))
        return results
    except Exception as exc:
        log.warning("sparse_embedding_failed", error=str(exc))
        # Return empty sparse vectors -- dense-only retrieval will still work
        return [([0], [0.0])] * len(texts)


def _chunk_to_payload(chunk: Chunk) -> dict:
    """Serialize a Chunk to a Qdrant payload dict."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "filename": chunk.metadata.filename,
        "document_type": chunk.metadata.document_type.value,
        "date": chunk.metadata.date.isoformat() if chunk.metadata.date else None,
        "parties": chunk.metadata.parties,
        "client_id": chunk.metadata.client_id,
    }


async def _upsert_chunks(client: AsyncQdrantClient, chunks: list[Chunk]) -> None:
    """Embed and upsert a batch of chunks into Qdrant."""
    if not chunks:
        return

    texts = [c.text for c in chunks]

    # Run dense embedding (async) and sparse embedding (sync in thread) concurrently
    dense_embeddings, sparse_vectors = await asyncio.gather(
        _embed_dense(texts),
        asyncio.get_event_loop().run_in_executor(None, _embed_sparse, texts),
    )

    points: list[PointStruct] = []
    for chunk, dense, (sp_idx, sp_val) in zip(chunks, dense_embeddings, sparse_vectors):
        points.append(
            PointStruct(
                id=str(uuid.UUID(chunk.chunk_id)),
                vector={
                    _DENSE_VECTOR_NAME: dense,
                    _SPARSE_VECTOR_NAME: {"indices": sp_idx, "values": sp_val},
                },
                payload=_chunk_to_payload(chunk),
            )
        )

    batch_size = settings.qdrant_upsert_batch_size
    total_upserted = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=batch,
            wait=True,
        )
        total_upserted += len(batch)
        log.debug("chunks_upserted_batch", batch_size=len(batch), total=total_upserted)
    log.info("chunks_upserted", count=total_upserted)


async def ingest_file(file_path: str | Path) -> list[Chunk]:
    """
    Full ingestion pipeline for a single PDF file.
    Returns the list of chunks stored in Qdrant.
    """
    path = Path(file_path)
    log.info("ingestion_started", file=str(path))

    text = extract_text_from_pdf(path)
    base_metadata = DocumentMetadata(filename=path.name)
    metadata = await enrich_metadata(text, base_metadata)
    chunks = chunk_document(text, metadata, semantic=True)

    qdrant = await _get_qdrant_client()
    await ensure_collection_exists(qdrant)
    await _upsert_chunks(qdrant, chunks)
    await qdrant.close()

    log.info("ingestion_complete", file=str(path), chunks=len(chunks))
    return chunks


async def ingest_bytes(content: bytes, filename: str) -> list[Chunk]:
    """
    Full ingestion pipeline for PDF bytes (e.g., from an HTTP upload).
    Returns the list of chunks stored in Qdrant.
    """
    log.info("ingestion_started", file=filename)

    text = extract_text_from_bytes(content, filename)
    base_metadata = DocumentMetadata(filename=filename)
    metadata = await enrich_metadata(text, base_metadata)
    chunks = chunk_document(text, metadata, semantic=True)

    qdrant = await _get_qdrant_client()
    await ensure_collection_exists(qdrant)
    await _upsert_chunks(qdrant, chunks)
    await qdrant.close()

    log.info("ingestion_complete", file=filename, chunks=len(chunks))
    return chunks
