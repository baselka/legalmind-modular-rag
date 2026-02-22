"""
Re-ranking layer -- narrows the retrieved top-K to the top-N most relevant chunks.

Two implementations:

1. CohereReranker (default)
   Uses Cohere's rerank-v3.5 API (cross-encoder model, 4096 context window).
   Reads query + document text jointly, unlike bi-encoders which encode them
   separately.  Joint encoding produces much higher relevance accuracy at the
   cost of ~100ms API latency for 20 docs.

2. CrossEncoderReranker (local fallback)
   Uses sentence-transformers cross-encoder/ms-marco-MiniLM-L-12-v2.
   No API dependency -- runs on CPU.  Useful for offline dev or CI environments.

Why two-stage retrieval?
  ANN (Approximate Nearest Neighbor) search via bi-encoders is designed for
  speed at scale.  It retrieves top-20 candidates in <10ms across 10M vectors.
  But bi-encoders encode query and document independently, missing fine-grained
  interaction signals.  The cross-encoder at stage 2 sees both together and
  produces a precise relevance score.  This pattern (retrieve-then-rerank) is
  the industry standard for high-accuracy RAG.

Factory function `get_reranker()` reads `settings.reranker_type` and returns
the appropriate implementation -- a config change, not a code change.
"""

from __future__ import annotations

import cohere

from src.config import settings
from src.models import Chunk
from src.retrieval.base import BaseReranker


import structlog

log = structlog.get_logger(__name__)

class CohereReranker(BaseReranker):
    """Re-ranks chunks using the Cohere Rerank v3.5 API."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.cohere_rerank_model
        self._client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
        log.info("cohere_reranker_initialized", model=self._model)

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []

        documents = [chunk.text for chunk in chunks]

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=top_n,
            )
        except Exception as exc:
            log.error("cohere_rerank_failed", error=str(exc))
            # Critical failure: return original chunks as fallback rather than crashing
            return chunks[:top_n]

        reranked: list[Chunk] = []
        for result in response.results:
            chunk = chunks[result.index]
            # Attach the rerank score as a signal (stored in embedding[0] by convention)
            chunk = chunk.model_copy(
                update={"embedding": [result.relevance_score]}
            )
            reranked.append(chunk)

        log.debug("reranking_complete", engine="cohere", count=len(reranked))
        return reranked


class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder re-ranker using sentence-transformers.
    No external API required -- suitable for offline/CI environments.
    Default model is BAAI/bge-reranker-v2-m3 (multilingual, Arabic-capable).
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.reranker_model
        self._model = None  # lazy-loaded on first use
        log.info("local_reranker_initialized", model=self._model_name)

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        import asyncio

        if not chunks:
            return []

        model = self._get_model()
        pairs = [(query, chunk.text) for chunk in chunks]

        # Run blocking CPU inference in a thread pool
        loop = asyncio.get_event_loop()
        scores: list[float] = await loop.run_in_executor(
            None, lambda: model.predict(pairs).tolist()
        )

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        log.debug("reranking_complete", engine="local", count=min(len(chunks), top_n))
        return [
            chunk.model_copy(update={"embedding": [score]})
            for chunk, score in scored[:top_n]
        ]


class PassthroughReranker(BaseReranker):
    """No-op reranker -- returns the top_n chunks as-is from retrieval."""

    def __init__(self) -> None:
        log.warning("passthrough_reranker_initialized", reason="explicitly_configured")

    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        return chunks[:top_n]


def get_reranker() -> BaseReranker:
    """Factory: returns the configured reranker implementation."""
    if settings.reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    if settings.reranker_type == "none":
        return PassthroughReranker()
    
    # Auto-fallback to local CrossEncoder if Cohere API key is missing
    if not settings.cohere_api_key:
        log.info("cohere_key_missing_falling_back_to_local_reranker")
        return CrossEncoderReranker()
        
    return CohereReranker()
