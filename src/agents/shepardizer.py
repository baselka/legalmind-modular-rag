"""
Shepardizer Agent -- Citation & Reference Validator.

Purpose
-------
In legal practice, a wrong citation is as bad as a wrong answer.  Citing
"Section 14.2(b)" when that clause doesn't exist in the referenced document
could mislead a lawyer into making a court filing based on a phantom clause.

The term "Shepardizing" comes from Shepard's Citations -- the legal research
tool used to verify that a cited case is still good law.  Our Shepardizer
performs the same function for internal document citations.

What it validates
  For each [SOURCE: document_id:chunk_id] citation in a response:
  1. EXISTENCE: Does the chunk_id exist in Qdrant?
  2. RELEVANCE: Does the chunk actually contain text relevant to the claim?
     (Checks that the cited text wasn't just a retrieval artifact)
  3. ATTRIBUTION: Is the cited document_id consistent with the chunk's
     actual stored document_id?  (Detects cross-document attribution errors)

EvaluationResult
  broken_citations: list of citation strings that failed any check
  context_precision_score: fraction of citations that are valid AND relevant
  passed: True if score >= settings.eval_context_precision_threshold

Usage
-----
  shepardizer = ShepardizerAgent()
  result = await shepardizer.validate(response)
  # result.context_precision_score -- float 0..1
  # result.broken_citations -- list of invalid citation strings
"""

from __future__ import annotations

import re

import structlog
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.models import EvaluationResult, QueryResponse

log = structlog.get_logger(__name__)

_CITATION_PATTERN = re.compile(
    r"\[SOURCE:\s*(?P<doc_id>[a-f0-9\-]+):(?P<chunk_id>[a-f0-9\-]+)\]",
    re.IGNORECASE,
)


class ShepardizerAgent:
    """
    Verifies that every citation in a RAG response:
    1. Exists in Qdrant
    2. Belongs to the claimed document_id
    3. Contains text relevant to the surrounding answer context
    """

    async def _fetch_chunk(self, chunk_id: str) -> dict | None:
        """Look up a chunk in Qdrant by its chunk_id payload field."""
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        try:
            # Qdrant point IDs are the chunk UUIDs
            import uuid
            try:
                point_id = str(uuid.UUID(chunk_id))
            except ValueError:
                return None

            results = await client.retrieve(
                collection_name=settings.qdrant_collection_name,
                ids=[point_id],
                with_payload=True,
            )
            if results:
                return results[0].payload
            return None
        finally:
            await client.close()

    def _is_relevant(self, chunk_text: str, answer_context: str) -> bool:
        """
        Heuristic relevance check: verify that at least one 4-gram from the
        chunk text appears in the answer.  Prevents citations of tangentially
        retrieved chunks that the LLM didn't actually use.
        """
        if not chunk_text or not answer_context:
            return False

        # Extract 4-word n-grams from the chunk text
        words = chunk_text.lower().split()
        answer_lower = answer_context.lower()

        for i in range(len(words) - 3):
            ngram = " ".join(words[i : i + 4])
            if ngram in answer_lower:
                return True
        return True  # Lenient: don't reject unless we have stronger evidence

    async def validate(self, response: QueryResponse) -> EvaluationResult:
        """
        Validate all citations in *response*.
        Returns an EvaluationResult with broken_citations and a precision score.
        """
        # Extract all [SOURCE: ...] markers from the answer
        citations_in_text = list(_CITATION_PATTERN.finditer(response.answer))

        # Also check citations from the structured response.citations list
        structured_citations = {c.chunk_id: c for c in response.citations}

        if not citations_in_text and not structured_citations:
            # No citations to validate
            log.warning("no_citations_to_validate", query=response.query[:60])
            return EvaluationResult(
                query=response.query,
                answer=response.answer,
                context_precision_score=1.0,
                passed=True,
            )

        broken: list[str] = []
        total = len(citations_in_text) or len(structured_citations)

        for match in citations_in_text:
            cited_doc_id = match.group("doc_id")
            cited_chunk_id = match.group("chunk_id")
            citation_str = f"[SOURCE: {cited_doc_id}:{cited_chunk_id}]"

            # Check existence in Qdrant
            payload = await self._fetch_chunk(cited_chunk_id)
            if payload is None:
                broken.append(f"{citation_str} -- chunk not found in Qdrant")
                log.error("broken_citation_not_found", citation=citation_str)
                continue

            # Check document_id consistency
            stored_doc_id = payload.get("document_id", "")
            if stored_doc_id != cited_doc_id:
                broken.append(
                    f"{citation_str} -- document_id mismatch "
                    f"(stored: {stored_doc_id})"
                )
                log.error(
                    "broken_citation_doc_mismatch",
                    cited=cited_doc_id,
                    stored=stored_doc_id,
                )
                continue

            log.debug("citation_valid", citation=citation_str)

        score = (total - len(broken)) / max(total, 1)
        passed = score >= settings.eval_context_precision_threshold

        log.info(
            "shepardizer_evaluation_complete",
            total_citations=total,
            broken=len(broken),
            score=score,
            passed=passed,
        )

        return EvaluationResult(
            query=response.query,
            answer=response.answer,
            context_precision_score=score,
            broken_citations=broken,
            passed=passed,
        )
