"""
Context Precision evaluation tests (RAG Triad metric 3).

What is Context Precision?
  Are the most relevant chunks ranked FIRST in the retrieved results?
  Context Precision measures retrieval quality -- specifically whether the
  re-ranker is doing its job.

  Formula: ContextPrecision@k = Σ (Precision@i × rel_i) / Σ rel_i
  where rel_i = 1 if the i-th result is relevant, 0 otherwise.

  A score of 1.0 means all relevant chunks are ranked before all irrelevant ones.

Threshold: >= 0.8 (configured in settings.eval_context_precision_threshold)

Why does this matter?
  The LLM's context window has a size limit.  We feed it the top-N chunks.
  If the most relevant chunk is ranked 19th (just below the cutoff), the LLM
  never sees it.  Context Precision measures whether our retrieval + re-ranking
  pipeline surfaces the right content early enough.

These tests use the Shepardizer agent's output and DeepEval's ContextualPrecisionMetric.
"""

from __future__ import annotations

import pytest

try:
    from deepeval import assert_test
    from deepeval.metrics import ContextualPrecisionMetric
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
def precision_metric():
    return ContextualPrecisionMetric(
        threshold=settings.eval_context_precision_threshold,
        model="gpt-4o",
        include_reason=True,
    )


@pytest.mark.parametrize(
    "question,expected_answer,context",
    [
        (
            "What is the indemnification obligation?",
            "The Service Provider must indemnify and hold harmless the Client from "
            "claims arising from negligence, misconduct, or breach.",
            [
                # Most relevant chunk first -- should be ranked #1
                "Service Provider shall indemnify, defend, and hold harmless Client and its "
                "officers, directors, employees, and agents from and against any and all claims, "
                "damages, losses, costs, and expenses arising out of or related to Service "
                "Provider's negligence, willful misconduct, or breach of this Agreement.",
                # Less relevant -- background context
                "This Agreement shall be governed by the laws of the State of New York.",
                # Irrelevant -- should be ranked last
                "Client shall pay Service Provider a monthly retainer fee of USD $15,000.",
            ],
        ),
        (
            "Under what circumstances may Henderson recover back pay?",
            "Henderson may recover back pay if the court finds Blackwood terminated him "
            "in retaliation for filing an SEC complaint under the Sarbanes-Oxley Act.",
            [
                "Henderson was terminated sixty days after filing an SEC whistleblower "
                "complaint, creating a strong inference of retaliatory intent. Back pay "
                "is estimated at $340,000 for ten months.",
                "Henderson alleges breach of the employment contract requiring 'cause' "
                "for termination, defined in Section 4.2.",
                "The Parties have executed this Agreement as of the Effective Date.",
            ],
        ),
    ],
)
def test_context_precision(precision_metric, question, expected_answer, context):
    """
    The first context item is the most relevant one.
    DeepEval should confirm it is ranked appropriately.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=expected_answer,
        expected_output=expected_answer,
        retrieval_context=context,
    )
    assert_test(test_case, [precision_metric])


def test_poor_context_precision_scores_low(precision_metric):
    """
    When an irrelevant chunk is ranked first, precision score should drop.
    Demonstrates the metric catches poor retrieval ranking.
    """
    test_case = LLMTestCase(
        input="What is the termination notice period?",
        actual_output="Either party may terminate with 30 days written notice.",
        expected_output="Either party may terminate with 30 days written notice.",
        retrieval_context=[
            # Irrelevant chunk ranked first (bad retrieval)
            "Client shall pay Service Provider a monthly retainer fee of USD $15,000.",
            # Irrelevant
            "The Agreement is governed by New York law.",
            # Relevant chunk buried last
            "Either party may terminate this Agreement upon thirty (30) days written notice.",
        ],
    )
    precision_metric.measure(test_case)
    assert precision_metric.score < 1.0, (
        "Expected precision to be < 1.0 when relevant chunk is not ranked first"
    )
