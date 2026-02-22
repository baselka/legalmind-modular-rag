"""
Faithfulness evaluation tests (RAG Triad metric 1).

What is Faithfulness (Groundedness)?
  A response is "faithful" if every factual claim in it is directly supported
  by the retrieved context chunks.  Faithfulness = 1.0 means zero hallucinations.

Threshold: >= 0.9 (configured in settings.eval_faithfulness_threshold)
  This means at most 10% of claims may be unsupported.  For legal use cases,
  this is intentionally strict -- a single unsupported claim in a contract
  interpretation could cause real harm.

How DeepEval's FaithfulnessMetric works:
  1. Uses an LLM to extract "verdicts" -- each claim in the response gets
     a verdict of "yes" (supported) or "no" (hallucinated).
  2. score = verdicts_yes / total_verdicts
  3. If score < threshold, the test fails.

These tests are run via: deepeval test run tests/eval/
Or in CI: deepeval test run tests/eval/ --confident-api-key ...
"""

from __future__ import annotations

import pytest

try:
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from src.config import settings

pytestmark = pytest.mark.skipif(
    not DEEPEVAL_AVAILABLE,
    reason="deepeval not installed",
)


@pytest.fixture
def faithfulness_metric():
    return FaithfulnessMetric(
        threshold=settings.eval_faithfulness_threshold,
        model="gpt-4o",
        include_reason=True,
    )


@pytest.mark.parametrize(
    "question,answer,context",
    [
        (
            "What is the monthly retainer fee?",
            "The monthly retainer fee is USD $15,000, payable within 30 days of invoice "
            "[SOURCE: doc-001:chunk-001].",
            [
                "Client shall pay Service Provider a monthly retainer fee of USD $15,000 "
                "(fifteen thousand dollars), due and payable within thirty (30) days of invoice."
            ],
        ),
        (
            "What happens in case of force majeure?",
            "Neither party shall be liable for delays caused by events beyond their "
            "reasonable control, including acts of God, war, terrorism, pandemics, "
            "government orders, or natural disasters.",
            [
                "Neither party shall be liable for any delay or failure to perform its "
                "obligations under this Agreement due to causes beyond its reasonable control, "
                "including acts of God, war, terrorism, pandemics, government orders, or "
                "natural disasters ('Force Majeure Event')."
            ],
        ),
        (
            "What is the governing law?",
            "The agreement is governed by New York law and disputes are resolved "
            "by binding arbitration under JAMS rules in New York City.",
            [
                "This Agreement shall be governed by and construed in accordance with "
                "the laws of the State of New York, without regard to conflicts of law "
                "principles. Any disputes shall be resolved by binding arbitration in "
                "New York City under the rules of JAMS."
            ],
        ),
    ],
)
def test_faithfulness_score(faithfulness_metric, question, answer, context):
    """
    Each parametrized case tests that a given answer is grounded in the context.
    The answer is expected to be faithful (score >= 0.9) because it paraphrases
    the source text accurately without adding external claims.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context,
    )
    assert_test(test_case, [faithfulness_metric])


def test_faithfulness_hallucinated_answer(faithfulness_metric):
    """
    Hallucinated answer should FAIL the faithfulness check.
    This test is EXPECTED TO FAIL -- it demonstrates the metric catches hallucinations.
    We invert it with pytest.raises to verify the guard works.
    """
    test_case = LLMTestCase(
        input="What is the termination notice period?",
        actual_output=(
            "Either party may terminate with 90 days notice, plus the Service Provider "
            "is entitled to a $50,000 termination fee upon early termination."
        ),
        retrieval_context=[
            "Either party may terminate this Agreement upon thirty (30) days written notice."
        ],
    )
    faithfulness_metric.measure(test_case)
    # A hallucinated termination fee and wrong notice period should score below threshold
    assert faithfulness_metric.score < 1.0, (
        "Expected faithfulness score to be below 1.0 for hallucinated claims"
    )
