"""
Application entry point.
Run with:  uvicorn main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.ingestion.pipeline import _get_qdrant_client, ensure_collection_exists

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if settings.app_env == "development"
        else structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: ensure the Qdrant collection and multilingual text index exist.
    This is idempotent -- safe to call against an already-populated collection.
    It creates the BM25 keyword index on the existing 'text' payload field
    without requiring any re-ingestion.
    """
    try:
        client = await _get_qdrant_client()
        await ensure_collection_exists(client)
        await client.close()
        log.info("startup_complete", status="qdrant_ready")
    except Exception as exc:
        log.warning("startup_qdrant_unavailable", error=str(exc))
    yield  # Application runs here


app = FastAPI(
    lifespan=lifespan,
    title="LegalMind Knowledge Assistant",
    description=(
        "Modular RAG system for querying 10,000+ internal legal case files and contracts. "
        "Features hybrid retrieval (dense + sparse), cross-encoder re-ranking, "
        "semantic caching, and hallucination-detection agents."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root() -> dict:
    return {
        "service": "LegalMind Knowledge Assistant",
        "version": "0.1.0",
        "docs": "/docs",
    }
