"""
Compliance Auditor Agent -- Faithfulness / Hallucination Detector.

Purpose
-------
For a law firm, an AI that confidently states incorrect facts is worse than
one that admits uncertainty.  This agent performs "LLM-as-judge" evaluation
to detect hallucinations before they reach end users.

How it works (RAG Triad: Faithfulness)
  1. Extract individual factual claims from the RAG response (LLM-powered)
  2. For each claim, ask a judge LLM: "Is this claim supported by the provided
     retrieved chunks?"
  3. Faithfulness score = (supported claims) / (total claims)
  4. Any unsupported claim is flagged and logged

Integration with DeepEval
  The `evaluate_with_deepeval` method wraps DeepEval's FaithfulnessMetric,
  which implements the same logic using Confident AI's grading rubric.
  Both are available; DeepEval is used in CI/CD pytest runs.

Usage
-----
  auditor = ComplianceAuditorAgent()
  result = await auditor.evaluate(query, answer, retrieved_chunks)
  # result.faithfulness_score -- float 0..1
  # result.unsupported_claims -- list of hallucinated statements
"""

from __future__ import annotations

import json

import structlog
from openai import AsyncOpenAI

from src.config import settings
from src.models import Chunk, EvaluationResult

log = structlog.get_logger(__name__)


_CLAIM_EXTRACTION_PROMPT = """You are a legal fact-checker.
Extract every individual factual claim from the AI response below.
A "claim" is any specific assertion about a date, party, clause number, dollar amount,
legal obligation, or outcome.

AI Response:
\"\"\"
{answer}
\"\"\"

Respond in JSON:
{{
  "claims": ["<claim 1>", "<claim 2>", ...]
}}
Return an empty list if there are no specific claims."""


_CLAIM_VERIFICATION_PROMPT = """You are a legal fact-checker.
Your task: determine if the following claim is FULLY SUPPORTED by the context below.

Claim: {claim}

Context (retrieved legal document excerpts):
\"\"\"
{context}
\"\"\"

Respond in JSON:
{{
  "supported": true or false,
  "reason": "<brief explanation citing specific text from context, or explaining why it's not supported>"
}}"""


class ComplianceAuditorAgent:
    """
    Detects hallucinations by cross-referencing response claims against
    retrieved chunks.  Returns a Faithfulness score and a list of
    unsupported claims.
    """

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _extract_claims(self, answer: str) -> list[str]:
        """Use an LLM to extract individual factual claims from the answer."""
        response = await self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": _CLAIM_EXTRACTION_PROMPT.format(answer=answer),
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return data.get("claims", [])

    async def _verify_claim(self, claim: str, context: str) -> tuple[bool, str]:
        """Check whether a single claim is supported by the context."""
        response = await self._openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": _CLAIM_VERIFICATION_PROMPT.format(
                        claim=claim, context=context[:3000]
                    ),
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return data.get("supported", False), data.get("reason", "")

    async def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[Chunk],
    ) -> EvaluationResult:
        """
        Full faithfulness evaluation:
          1. Extract claims from answer
          2. Verify each claim against retrieved chunks
          3. Compute score and return EvaluationResult
        """
        context = "\n\n---\n\n".join(
            f"[{c.metadata.filename} | chunk {c.chunk_id}]\n{c.text}"
            for c in retrieved_chunks
        )

        claims = await self._extract_claims(answer)
        if not claims:
            # No claims = no hallucinations possible (usually an "I don't know")
            return EvaluationResult(
                query=query,
                answer=answer,
                faithfulness_score=1.0,
                passed=True,
            )

        unsupported: list[str] = []
        for claim in claims:
            supported, reason = await self._verify_claim(claim, context)
            if not supported:
                unsupported.append(f"{claim} (reason: {reason})")
                log.warning("hallucination_detected", claim=claim, reason=reason)

        score = (len(claims) - len(unsupported)) / len(claims)
        passed = score >= settings.eval_faithfulness_threshold

        log.info(
            "faithfulness_evaluation_complete",
            score=score,
            claims=len(claims),
            unsupported=len(unsupported),
            passed=passed,
        )

        return EvaluationResult(
            query=query,
            answer=answer,
            faithfulness_score=score,
            unsupported_claims=unsupported,
            passed=passed,
        )

    async def evaluate_with_deepeval(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[Chunk],
        expected_output: str = "",
    ) -> EvaluationResult:
        """
        DeepEval-based faithfulness evaluation (used in pytest CI/CD runs).
        Returns the same EvaluationResult shape for consistency.
        """
        try:
            from deepeval.metrics import FaithfulnessMetric
            from deepeval.test_case import LLMTestCase

            context_texts = [c.text for c in retrieved_chunks]
            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_output or answer,
                retrieval_context=context_texts,
            )
            metric = FaithfulnessMetric(
                threshold=settings.eval_faithfulness_threshold,
                model="gpt-4o",
            )
            metric.measure(test_case)

            score = metric.score or 0.0
            passed = metric.is_successful()
            reason = metric.reason or ""

            return EvaluationResult(
                query=query,
                answer=answer,
                faithfulness_score=score,
                unsupported_claims=[reason] if not passed else [],
                passed=passed,
            )

        except Exception as exc:
            log.warning("deepeval_faithfulness_failed", error=str(exc))
            return await self.evaluate(query, answer, retrieved_chunks)
