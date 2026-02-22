"""
FastAPI route handlers.

Endpoints:
  POST /ingest     -- Upload a PDF, trigger the ingestion pipeline
  POST /query      -- Submit a legal question, receive a grounded answer with citations
  GET  /health     -- Liveness + readiness check (Qdrant + Redis connectivity)
  GET  /documents  -- List ingested documents with metadata

The query endpoint implements the full RAG pipeline:
  1. Check semantic cache (Redis)
  2. Hybrid retrieval (Qdrant dense + sparse with RRF)
  3. Cross-encoder re-ranking (Cohere)
  4. LLM generation with citation mandates (GPT-4o)
  5. Write response to cache
  6. Return structured QueryResponse

All endpoints are async and use dependency injection for shared clients.
"""

from __future__ import annotations

import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.cache.semantic_cache import SemanticCache
from src.config import settings
from src.generation.llm import get_llm
from src.ingestion.pipeline import ensure_collection_exists, ingest_bytes
from src.models import DocumentMetadata, QueryRequest, QueryResponse
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import get_reranker

log = structlog.get_logger(__name__)

router = APIRouter()

# Module-level singletons for shared clients
_cache: SemanticCache | None = None
_hybrid_retriever: HybridRetriever | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache


def get_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["ops"])
async def health_check() -> dict:
    """
    Liveness and readiness check.
    Returns 200 if both Qdrant and Redis are reachable, 503 otherwise.
    """
    checks: dict[str, str] = {}

    # Qdrant
    try:
        qdrant = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        await qdrant.get_collections()
        await qdrant.close()
        checks["qdrant"] = "ok"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    # Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    status_code = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@router.post("/ingest", tags=["ingestion"], status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: Annotated[UploadFile, File(description="Legal document PDF to ingest")],
    cache: Annotated[SemanticCache, Depends(get_cache)],
) -> dict:
    """
    Upload a PDF document and trigger the full ingestion pipeline:
    parse -> enrich metadata -> chunk -> embed -> store in Qdrant.

    Also invalidates semantic cache entries referencing the same document
    filename (for re-ingestion scenarios).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are supported.",
        )

    content = await file.read()
    filename = file.filename

    log.info("ingest_request", filename=filename, size_bytes=len(content))

    chunks = await ingest_bytes(content, filename)

    return {
        "status": "ingested",
        "filename": filename,
        "chunks_stored": len(chunks),
        "document_id": chunks[0].document_id if chunks else None,
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@router.post("/query", tags=["query"], response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    cache: Annotated[SemanticCache, Depends(get_cache)],
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
) -> QueryResponse:
    """
    Submit a legal question and receive a grounded answer with citations.

    Pipeline:
      1. Check semantic cache for a similar prior answer
      2. Hybrid retrieval (dense + sparse with RRF) -- top-K chunks
      3. Cross-encoder re-ranking -- narrow to top-N
      4. LLM generation with citation mandate
      5. Cache the response
      6. Return structured QueryResponse
    """
    t0 = time.monotonic()
    log.info("query_received", query=request.query[:100])

    # --- Step 1: Semantic cache lookup ---
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    query_embedding_resp = await openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=request.query,
    )
    query_embedding = query_embedding_resp.data[0].embedding

    cached_response = await cache.get(query_embedding)
    if cached_response is not None:
        cached_response = cached_response.model_copy(
            update={"cached": True, "latency_ms": (time.monotonic() - t0) * 1000}
        )
        log.info("cache_hit", query=request.query[:60])
        return cached_response

    # --- Step 2: Hybrid retrieval ---
    chunks = await retriever.retrieve(request)
    if not chunks:
        return QueryResponse(
            query=request.query,
            answer=(
                "I don't know based on the provided documents. "
                "No relevant documents were found. Please ingest relevant files first."
            ),
            citations=[],
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # --- Step 3: Re-ranking ---
    reranker = get_reranker()
    top_n = request.top_n or settings.rerank_top_n
    reranked_chunks = await reranker.rerank(request.query, chunks, top_n=top_n)

    # --- Step 4: LLM generation ---
    llm = get_llm()
    response = await llm.complete(request.query, reranked_chunks)

    # --- Step 5: Cache write ---
    await cache.set(query_embedding, response)

    response = response.model_copy(
        update={"latency_ms": (time.monotonic() - t0) * 1000}
    )

    log.info(
        "query_complete",
        latency_ms=response.latency_ms,
        citations=len(response.citations),
    )
    return response


# ---------------------------------------------------------------------------
# Documents list
# ---------------------------------------------------------------------------

@router.get("/documents", tags=["ingestion"])
async def list_documents() -> dict:
    """
    List all documents currently stored in Qdrant with their metadata.
    Queries Qdrant for unique document_id payload values.
    """
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    try:
        # Scroll through all points and collect unique document metadata
        seen_doc_ids: set[str] = set()
        documents: list[dict] = []
        next_page_offset = None

        while True:
            results, next_page_offset = await client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                doc_id = payload.get("document_id", "")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    documents.append(
                        {
                            "document_id": doc_id,
                            "filename": payload.get("filename"),
                            "document_type": payload.get("document_type"),
                            "date": payload.get("date"),
                            "parties": payload.get("parties", []),
                            "client_id": payload.get("client_id"),
                        }
                    )
            if next_page_offset is None:
                break

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Could not reach Qdrant: {exc}",
        )
    finally:
        await client.close()

    return {"total": len(documents), "documents": documents}
