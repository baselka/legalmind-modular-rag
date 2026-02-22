"""
RAG System Benchmark
====================
This test suite benchmarks the LIVE RAG system against the Golden Dataset.
It satisfies the Case Study requirement: "Use GitHub Actions to run these evals 
on every Pull Request... fail the build if Faithfulness drops below 0.9".

Agents Integrated:
- Compliance Auditor (via FaithfulnessMetric)
- Shepardizer (via ContextualPrecisionMetric)
"""

import os
import json
import pytest
import httpx
from pathlib import Path

from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from src.config import settings

# --- Discovery ---
DATASET_PATH = Path("golden_dataset/rag_eval_questions.json")
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/query")

def load_golden_dataset():
    if not DATASET_PATH.exists():
        pytest.skip(f"Golden dataset not found at {DATASET_PATH}")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]

# --- Metrics ---
@pytest.fixture
def faithfulness_metric():
    return FaithfulnessMetric(
        threshold=settings.eval_faithfulness_threshold, # 0.9
        model="gpt-4o",
        include_reason=True
    )

@pytest.fixture
def precision_metric():
    return ContextualPrecisionMetric(
        threshold=settings.eval_context_precision_threshold, # 0.8
        model="gpt-4o",
        include_reason=True
    )

# --- Benchmark ---
@pytest.mark.parametrize("q_entry", load_golden_dataset())
def test_rag_system_performance(q_entry, faithfulness_metric, precision_metric):
    """
    Test the live RAG system's response for a golden dataset question.
    """
    # 1. Query the live API
    try:
        response = httpx.post(
            API_URL, 
            json={"query": q_entry["question"]},
            timeout=120.0
        )
        response.raise_for_status()
        res_data = response.json()
    except Exception as exc:
        pytest.fail(f"API query failed for {q_entry['id']}: {exc}")

    actual_answer = res_data["answer"]
    retrieved_contexts = [c["excerpt"] for c in res_data.get("citations", [])]
    
    # 2. Construct evaluation test case
    test_case = LLMTestCase(
        input=q_entry["question"],
        actual_output=actual_answer,
        expected_output=q_entry["ground_truth"],
        retrieval_context=retrieved_contexts
    )
    
    # 3. Assert using Compliance Auditor and Shepardizer metrics
    # This will fail the pytest run if faithfulness < 0.9 or precision < 0.8
    assert_test(test_case, [faithfulness_metric, precision_metric])
