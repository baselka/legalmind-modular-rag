"""
Response builder: parses the raw LLM output and extracts structured citations.

The LLM is instructed to include [SOURCE: document_id:chunk_id] markers.
This module:
  1. Extracts all citation markers from the answer text
  2. Cross-references them against the retrieved chunks to build SourceCitation objects
  3. Returns a fully structured QueryResponse with citations verified against context

Why parse citations from text?
  We could ask the LLM to return JSON directly, but that risks the model
  embedding reasoning in JSON strings or hallucinating structure.  The inline
  citation format is simpler for the model and more robust.  We do the
  structured extraction post-hoc in code.
"""

from __future__ import annotations

import re

from src.models import Chunk, QueryResponse, SourceCitation


_CITATION_PATTERN = re.compile(
    r"\[SOURCE:\s*(?P<doc_id>[a-f0-9\-]+):(?P<chunk_id>[a-f0-9\-]+)\]",
    re.IGNORECASE,
)


def extract_citations(
    answer_text: str,
    context_chunks: list[Chunk],
) -> list[SourceCitation]:
    """
    Parse [SOURCE: doc_id:chunk_id] markers from *answer_text* and resolve
    them to SourceCitation objects using the provided context chunks.

    Only citations that match a chunk in *context_chunks* are included,
    which guards against the model fabricating chunk IDs.
    """
    chunk_map = {chunk.chunk_id: chunk for chunk in context_chunks}
    seen_chunk_ids: set[str] = set()
    citations: list[SourceCitation] = []

    for match in _CITATION_PATTERN.finditer(answer_text):
        chunk_id = match.group("chunk_id")

        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            # Model cited a chunk_id that wasn't in the context -- skip it.
            # The Shepardizer agent will flag this as a broken citation.
            continue

        # Compute a simple relevance score from the reranker (stored in embedding[0])
        relevance = chunk.embedding[0] if chunk.embedding else 0.0

        citations.append(
            SourceCitation(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                filename=chunk.metadata.filename,
                # Include full chunk text (up to 2000 chars) for evaluation and display
                excerpt=chunk.text[:2000].replace("\n", " ").strip(),
                relevance_score=relevance,
            )
        )

    return citations


def build_query_response(
    query: str,
    raw_answer: str,
    context_chunks: list[Chunk],
    latency_ms: float = 0.0,
    cached: bool = False,
) -> QueryResponse:
    """
    Combine the raw LLM answer with extracted citations into a QueryResponse.
    """
    citations = extract_citations(raw_answer, context_chunks)

    # If the model answered but cited nothing, fall back to including all
    # context chunks as citations (better than an uncited answer).
    if not citations and "I don't know" not in raw_answer:
        citations = [
            SourceCitation(
                document_id=c.document_id,
                chunk_id=c.chunk_id,
                filename=c.metadata.filename,
                excerpt=c.text[:2000].replace("\n", " ").strip(),
                relevance_score=c.embedding[0] if c.embedding else 0.0,
            )
            for c in context_chunks
        ]

    return QueryResponse(
        query=query,
        answer=raw_answer,
        citations=citations,
        cached=cached,
        latency_ms=latency_ms,
    )
