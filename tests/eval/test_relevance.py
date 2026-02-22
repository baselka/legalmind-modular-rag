"""
Answer Relevance evaluation tests (RAG Triad metric 2).

What is Answer Relevance?
  The response must actually answer the user's question.  A response that is
  100% grounded in the context but doesn't address the query is still a
  failure -- it's an evasive non-answer.

Threshold: >= 0.8 (configured in settings.eval_relevance_threshold)

How DeepEval's AnswerRelevancyMetric works:
  1. Generates statements from the response
  2. For each statement, asks: "Is this relevant to the input question?"
  3. score = relevant_statements / total_statements

This is distinct from faithfulness: relevance checks if you answered the
right question, faithfulness checks if you answered truthfully.
"""

from __future__ import annotations

import pytest

try:
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric
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
def relevance_metric():
    return AnswerRelevancyMetric(
        threshold=settings.eval_relevance_threshold,
        model="gpt-4o",
        include_reason=True,
    )


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "Who are the parties to the Professional Services Agreement?",
            "The parties are Acme Corporation as the Client and LexTech Solutions LLC "
            "as the Service Provider, as stated in the opening of the Agreement.",
        ),
        (
            "What is the limitation of liability under the contract?",
            "Neither party shall be liable for indirect, incidental, or consequential "
            "damages. The total cumulative liability of the Service Provider shall not "
            "exceed the total fees paid by Client in the twelve months preceding the claim.",
        ),
        (
            "What are the whistleblower's claims against Blackwood Industries?",
            "Henderson alleges wrongful termination in violation of the Sarbanes-Oxley "
            "Act, breach of employment contract requiring 'cause' for termination, and "
            "defamation based on a false statement to a prospective employer.",
        ),
    ],
)
def test_answer_relevance(relevance_metric, question, answer):
    """
    Each answer should score >= threshold for relevance to its question.
    These are examples of on-topic, directly responsive legal answers.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
    )
    assert_test(test_case, [relevance_metric])


def test_irrelevant_answer_scores_low(relevance_metric):
    """
    An off-topic answer should score low on relevance.
    Demonstrates the metric catches evasive responses.
    """
    test_case = LLMTestCase(
        input="What is the termination notice period?",
        actual_output=(
            "The agreement contains clauses about compensation, intellectual property, "
            "and governing law. The parties are Acme Corporation and LexTech Solutions."
        ),
    )
    relevance_metric.measure(test_case)
    # An answer that doesn't mention termination should score below 0.8
    assert relevance_metric.score < settings.eval_relevance_threshold, (
        "Expected low relevance score for an off-topic answer"
    )
