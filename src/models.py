"""
Shared Pydantic data models used across every layer of the system.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    CONTRACT = "contract"
    CASE_FILE = "case_file"
    PLEADING = "pleading"
    BRIEF = "brief"
    CORRESPONDENCE = "correspondence"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata extracted from a legal document during ingestion."""

    document_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    document_type: DocumentType = DocumentType.UNKNOWN
    date: datetime | None = None
    parties: list[str] = Field(default_factory=list)
    client_id: str | None = None
    # Any extra fields extracted by the enricher
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single chunk produced by the chunker, ready for embedding and storage."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    # Position within the source document (0-based)
    chunk_index: int = 0
    metadata: DocumentMetadata
    # Populated after embedding
    embedding: list[float] | None = None
    sparse_indices: list[int] | None = None
    sparse_values: list[float] | None = None


class SourceCitation(BaseModel):
    """A single cited source included in a RAG response."""

    document_id: str
    chunk_id: str
    filename: str
    # Verbatim excerpt from the chunk that supports the answer
    excerpt: str
    relevance_score: float


class QueryRequest(BaseModel):
    """Incoming query from the user (via API or internal call)."""

    query: str = Field(..., min_length=1, max_length=4096)
    # Optional metadata filters applied before vector search
    filter_document_type: DocumentType | None = None
    filter_client_id: str | None = None
    filter_date_from: datetime | None = None
    filter_date_to: datetime | None = None
    # Override per-request retrieval limits (falls back to settings)
    top_k: int | None = None
    top_n: int | None = None


class QueryResponse(BaseModel):
    """Final answer returned to the user."""

    query: str
    answer: str
    citations: list[SourceCitation]
    # Whether this response was served from the semantic cache
    cached: bool = False
    latency_ms: float = 0.0


class GoldenDatasetEntry(BaseModel):
    """A single entry in the evaluation golden dataset."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    reference_context: str
    expected_answer: str
    # Source document IDs that informed the question
    source_document_ids: list[str]
    # Multi-hop: requires reasoning over multiple documents
    is_multi_hop: bool = False


class EvaluationResult(BaseModel):
    """Result of running an evaluation agent over a single response."""

    query: str
    answer: str
    faithfulness_score: float | None = None
    relevance_score: float | None = None
    context_precision_score: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    broken_citations: list[str] = Field(default_factory=list)
    passed: bool = True
