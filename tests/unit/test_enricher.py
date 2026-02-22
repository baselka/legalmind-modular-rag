"""
Unit tests for the metadata enricher.

Tests the regex-based fallback date extraction (no API calls required).
LLM-based enrichment is integration-tested separately (requires API key).
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.ingestion.enricher import _fallback_date, enrich_metadata
from src.models import DocumentMetadata, DocumentType


def test_fallback_date_mdy_slash():
    text = "Signed on 03/15/2023 between the parties."
    date = _fallback_date(text)
    assert date is not None


def test_fallback_date_written_month():
    text = "This Agreement is entered into as of March 15, 2023."
    date = _fallback_date(text)
    assert date is not None
    assert date.year == 2023
    assert date.month == 3
    assert date.day == 15


def test_fallback_date_no_date():
    text = "This agreement contains no date information whatsoever."
    date = _fallback_date(text)
    assert date is None


def test_fallback_date_first_date_found():
    text = "Filed January 5, 2022. Amended June 8, 2022."
    date = _fallback_date(text)
    assert date is not None
    # Should find the first date
    assert date.year == 2022


@pytest.mark.asyncio
async def test_enrich_metadata_uses_llm_response():
    """Mock the OpenAI call to test enrichment logic without API."""
    base = DocumentMetadata(filename="contract_test.pdf")
    mock_content = '{"document_type": "contract", "date": "2023-03-15", "parties": ["Acme Corp", "LexTech LLC"], "client_id": "ACM-2023"}'

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = mock_content

    with patch("src.ingestion.enricher.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_cls.return_value = mock_client

        result = await enrich_metadata("Some legal document text about a contract.", base)

    assert result.document_type == DocumentType.CONTRACT
    assert result.date is not None
    assert result.date.year == 2023
    assert "Acme Corp" in result.parties
    assert result.client_id == "ACM-2023"


@pytest.mark.asyncio
async def test_enrich_metadata_fallback_on_api_error():
    """When the LLM call fails, enricher should gracefully fall back."""
    base = DocumentMetadata(filename="scanned_contract.pdf")
    text = "Agreement dated March 15, 2023 between parties."

    with patch("src.ingestion.enricher.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API unavailable")
        )
        mock_openai_cls.return_value = mock_client

        result = await enrich_metadata(text, base)

    # Should still have a date from regex fallback
    assert result.date is not None
    # Document type should default to UNKNOWN on failure
    assert result.document_type == DocumentType.UNKNOWN
