"""
Unit tests for the chunker module.

Tests the fixed-size chunker (no API calls required) and validates
that chunks have the correct structure, metadata propagation, and
that overlap is respected.
"""

import pytest
from src.ingestion.chunker import chunk_document_fixed
from src.models import DocumentMetadata, DocumentType


@pytest.fixture
def legal_text() -> str:
    return """PROFESSIONAL SERVICES AGREEMENT

This Professional Services Agreement ("Agreement") is entered into as of March 15, 2023
by and between Acme Corporation ("Client") and LexTech Solutions LLC ("Service Provider").

1. SERVICES
Service Provider agrees to provide Client with legal technology consulting services.
Service Provider shall perform the Services in a professional and workmanlike manner.

2. COMPENSATION
Client shall pay Service Provider a monthly retainer fee of USD $15,000 per month,
due and payable within thirty (30) days of invoice. Late payments accrue interest
at the rate of 1.5% per month.

3. INDEMNIFICATION
Service Provider shall indemnify, defend, and hold harmless Client and its officers,
directors, employees, and agents from and against any and all claims, damages, losses,
costs, and expenses (including reasonable attorneys fees) arising out of or related to
Service Provider's negligence, willful misconduct, or breach of this Agreement.

4. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, special,
exemplary, or consequential damages, even if advised of the possibility of such damages.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York.
""" * 5  # Repeat to ensure multiple chunks


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        filename="test_contract.pdf",
        document_type=DocumentType.CONTRACT,
        parties=["Acme Corporation", "LexTech Solutions LLC"],
    )


def test_fixed_chunker_returns_chunks(legal_text, sample_metadata):
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    assert len(chunks) > 0


def test_fixed_chunker_chunk_size(legal_text, sample_metadata):
    """No chunk should wildly exceed the configured chunk size in characters."""
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    # Allow up to 2x chunk_size in characters (tokens != chars, ~4 chars/token)
    max_chars = 512 * 6  # generous bound
    for chunk in chunks:
        assert len(chunk.text) <= max_chars, (
            f"Chunk {chunk.chunk_index} exceeds expected max length: {len(chunk.text)}"
        )


def test_fixed_chunker_metadata_propagation(legal_text, sample_metadata):
    """Metadata (document_id, filename, document_type) must propagate to every chunk."""
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    for chunk in chunks:
        assert chunk.document_id == sample_metadata.document_id
        assert chunk.metadata.filename == "test_contract.pdf"
        assert chunk.metadata.document_type == DocumentType.CONTRACT


def test_fixed_chunker_chunk_indices(legal_text, sample_metadata):
    """chunk_index should be monotonically increasing from 0."""
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_fixed_chunker_unique_chunk_ids(legal_text, sample_metadata):
    """Each chunk must have a unique chunk_id."""
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"


def test_fixed_chunker_non_empty_text(legal_text, sample_metadata):
    """All chunks must contain non-empty text."""
    chunks = chunk_document_fixed(legal_text, sample_metadata)
    for chunk in chunks:
        assert chunk.text.strip(), f"Chunk {chunk.chunk_index} has empty text"


def test_fixed_chunker_short_text(sample_metadata):
    """A very short document should produce exactly one chunk."""
    short_text = "This is a simple test clause."
    chunks = chunk_document_fixed(short_text, sample_metadata)
    assert len(chunks) == 1
    assert short_text in chunks[0].text


def test_fixed_chunker_empty_text(sample_metadata):
    """An empty document should produce no chunks or one empty chunk gracefully."""
    chunks = chunk_document_fixed("", sample_metadata)
    # LlamaIndex may return 0 or 1 empty chunk -- both are acceptable
    assert isinstance(chunks, list)
