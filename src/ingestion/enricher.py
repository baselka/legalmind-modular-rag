"""
Metadata enricher.

Uses an LLM to extract structured metadata from the raw document text:
  - Document type (contract, case file, pleading, etc.)
  - Execution/filing date
  - Named parties (law firm clients, opposing counsel, court names)
  - Client ID if present

Why LLM-based enrichment?
  Legal documents have wildly inconsistent headers.  A contract signed in 2019
  might not say "Contract" anywhere -- it says "Services Agreement".  A regex
  approach would need hundreds of rules.  An LLM classifies document type with
  near-perfect accuracy from the opening paragraph alone.

The enricher uses structured output (response_format=JSON) to guarantee
parseable results.  We limit the input to the first 2000 characters (header
region) to keep token costs low.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from openai import AsyncOpenAI

from src.config import settings
from src.models import DocumentMetadata, DocumentType


_ENRICHMENT_PROMPT = """You are a legal document analysis assistant.
Analyze the document excerpt below and extract the following metadata in JSON format.

Fields:
- document_type: one of [contract, case_file, pleading, brief, correspondence, unknown]
- date: ISO-8601 date string (YYYY-MM-DD) if a signing/filing date is present, else null
- parties: array of entity names (individuals, companies, courts) mentioned as parties
- client_id: alphanumeric client or matter ID if present, else null

Respond ONLY with a valid JSON object. No explanation.

Document excerpt:
\"\"\"
{excerpt}
\"\"\"
"""

_DATE_PATTERNS = [
    r"\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})\b",
]


def _fallback_date(text: str) -> datetime | None:
    """Simple regex date extraction used when LLM extraction fails."""
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            # Try common formats that match our regexes
            for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%B %d, %Y", "%m/%d/%y", "%m-%d-%y"]:
                try:
                    # strptime is case-sensitive for %B, so we capitalise
                    val = m.group(0).replace(",", ", ").replace("  ", " ").strip()
                    # If it contains letters, title case it for %B
                    if any(c.isalpha() for c in val):
                        val = val.title()
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
    return None


async def enrich_metadata(
    text: str,
    base_metadata: DocumentMetadata,
) -> DocumentMetadata:
    """
    Run LLM-based metadata extraction on *text* and merge results into
    *base_metadata* (which already has document_id and filename).
    Returns an updated copy of the metadata.
    """
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    excerpt = text[:2000]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # cheap model -- metadata extraction is simple
            messages=[
                {
                    "role": "user",
                    "content": _ENRICHMENT_PROMPT.format(excerpt=excerpt),
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data: dict = json.loads(raw)

        doc_type_str = data.get("document_type", "unknown")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.UNKNOWN

        date_str = data.get("date")
        date = None
        if date_str:
            try:
                date = datetime.fromisoformat(date_str)
            except ValueError:
                date = _fallback_date(text)

        parties = data.get("parties", [])
        client_id = data.get("client_id")

        return base_metadata.model_copy(
            update={
                "document_type": doc_type,
                "date": date,
                "parties": parties if isinstance(parties, list) else [],
                "client_id": client_id,
            }
        )

    except Exception:
        # Non-critical: fall back to regex date and unknown type
        return base_metadata.model_copy(
            update={
                "date": _fallback_date(text),
            }
        )
