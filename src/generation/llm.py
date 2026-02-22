"""
LLM abstraction layer (Strategy Pattern).
All generation code talks to BaseLLM -- making the underlying model swappable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from openai import AsyncOpenAI

from src.config import settings
from src.models import Chunk, QueryResponse
from src.generation.prompts import build_system_prompt, build_user_message
from src.generation.response import build_query_response


class BaseLLM(ABC):
    """Interface every LLM backend must satisfy."""

    @abstractmethod
    async def complete(
        self,
        query: str,
        context_chunks: list[Chunk],
    ) -> QueryResponse:
        """Generate an answer grounded in *context_chunks* for *query*."""
        ...


class OpenAILLM(BaseLLM):
    """
    GPT-4o (or any OpenAI-compatible model) implementation.

    Temperature is set to 0.0 for deterministic, factually-grounded output.
    Legal accuracy requires reproducibility -- creative variation is a liability.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or settings.openai_chat_model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def complete(
        self,
        query: str,
        context_chunks: list[Chunk],
    ) -> QueryResponse:
        t0 = time.monotonic()

        system_prompt = build_system_prompt()
        user_message = build_user_message(query, context_chunks)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
        )

        raw_answer = response.choices[0].message.content or ""
        latency_ms = (time.monotonic() - t0) * 1000

        return build_query_response(
            query=query,
            raw_answer=raw_answer,
            context_chunks=context_chunks,
            latency_ms=latency_ms,
        )


def get_llm() -> BaseLLM:
    """Factory: returns the configured LLM implementation."""
    return OpenAILLM()
