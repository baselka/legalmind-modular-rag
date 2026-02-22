"""
Pytest shared fixtures for unit and evaluation tests.

Provides:
  - sample_chunks: a list of Chunk objects without needing Qdrant
  - sample_query_response: a QueryResponse with inline citations
  - golden_dataset: loads golden_dataset/golden.json if it exists, else returns minimal stub
  - app_client: async httpx TestClient for API endpoint tests
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from httpx import AsyncClient, ASGITransport

from src.models import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    GoldenDatasetEntry,
    QueryResponse,
    SourceCitation,
)

# Use a fixed document_id and chunk_id so tests are deterministic
_DOC_ID = "aaaaaaaa-0000-0000-0000-aaaaaaaaaaaa"
_CHUNK_ID_1 = "bbbbbbbb-0000-0000-0000-bbbbbbbbbbbb"
_CHUNK_ID_2 = "cccccccc-0000-0000-0000-cccccccccccc"


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        document_id=_DOC_ID,
        filename="contract_acme_services_2023.pdf",
        document_type=DocumentType.CONTRACT,
        parties=["Acme Corporation", "LexTech Solutions LLC"],
        client_id="ACM-2023-0315",
    )


@pytest.fixture
def sample_chunks(sample_metadata) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=_CHUNK_ID_1,
            document_id=_DOC_ID,
            text=(
                "Service Provider shall indemnify, defend, and hold harmless Client "
                "and its officers, directors, employees, and agents from and against "
                "any and all claims, damages, losses, costs, and expenses "
                "(including reasonable attorneys' fees) arising out of or related to "
                "Service Provider's negligence, willful misconduct, or breach of this Agreement. "
                "The total cumulative liability shall not exceed USD $15,000."
            ),
            chunk_index=0,
            metadata=sample_metadata,
            embedding=[0.85],
        ),
        Chunk(
            chunk_id=_CHUNK_ID_2,
            document_id=_DOC_ID,
            text=(
                "This Agreement shall be governed by and construed in accordance with "
                "the laws of the State of New York, without regard to conflicts of law. "
                "Any disputes shall be resolved by binding arbitration in New York City "
                "under the rules of JAMS."
            ),
            chunk_index=1,
            metadata=sample_metadata,
            embedding=[0.72],
        ),
    ]


@pytest.fixture
def sample_query_response(sample_chunks) -> QueryResponse:
    return QueryResponse(
        query="What is the indemnification obligation of the Service Provider?",
        answer=(
            f"The Service Provider must indemnify the Client against all claims "
            f"arising from negligence or breach of contract "
            f"[SOURCE: {_DOC_ID}:{_CHUNK_ID_1}]. "
            f"Disputes are resolved in New York via JAMS arbitration "
            f"[SOURCE: {_DOC_ID}:{_CHUNK_ID_2}].\n\n"
            f"⚠️ This response is for informational purposes only and does not constitute legal advice."
        ),
        citations=[
            SourceCitation(
                document_id=_DOC_ID,
                chunk_id=_CHUNK_ID_1,
                filename="contract_acme_services_2023.pdf",
                excerpt="Service Provider shall indemnify, defend, and hold harmless...",
                relevance_score=0.85,
            ),
            SourceCitation(
                document_id=_DOC_ID,
                chunk_id=_CHUNK_ID_2,
                filename="contract_acme_services_2023.pdf",
                excerpt="This Agreement shall be governed by the laws of New York...",
                relevance_score=0.72,
            ),
        ],
    )


@pytest.fixture
def golden_dataset() -> list[GoldenDatasetEntry]:
    """Load the golden dataset from disk, or return a minimal stub for unit tests."""
    golden_path = Path("golden_dataset/golden.json")
    if golden_path.exists():
        raw = json.loads(golden_path.read_text(encoding="utf-8"))
        return [GoldenDatasetEntry.model_validate(e) for e in raw]

    # Minimal stub -- used when golden dataset hasn't been generated yet
    return [
        GoldenDatasetEntry(
            question="What is the indemnification obligation?",
            reference_context=(
                "Service Provider shall indemnify, defend, and hold harmless Client..."
            ),
            expected_answer=(
                "The Service Provider must indemnify the Client against claims arising "
                "from its negligence, misconduct, or breach of agreement."
            ),
            source_document_ids=[_DOC_ID],
            is_multi_hop=False,
        )
    ]


@pytest.fixture
async def app_client():
    """Async HTTP client for API tests. Sets dummy env vars to bypass settings validation."""
    os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
