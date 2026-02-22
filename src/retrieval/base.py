"""
Abstract base classes for the retrieval layer (Strategy Pattern).
Swap any implementation without touching the rest of the system.
"""

from abc import ABC, abstractmethod

from src.models import Chunk, QueryRequest


class BaseRetriever(ABC):
    """Interface every retriever must satisfy."""

    @abstractmethod
    async def retrieve(self, request: QueryRequest) -> list[Chunk]:
        """Return the top-K most relevant chunks for the query."""
        ...


class BaseReranker(ABC):
    """Interface every re-ranker must satisfy."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """
        Re-rank *chunks* using a cross-encoder or equivalent model and return
        the *top_n* most relevant in descending order.
        """
        ...
