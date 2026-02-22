"""
Adversarial Lawyer Agent -- Synthetic Golden Dataset Generator.

Purpose
-------
A high-quality evaluation suite requires ground-truth question-answer pairs.
Manually creating 50+ pairs from 10,000 documents is impractical.  This agent
uses an LLM to automatically scan ingested chunks and generate:

  - Simple single-hop questions  (e.g., "What is the indemnity cap in contract X?")
  - Complex multi-hop questions  (e.g., "How does the limitation of liability in
    Contract A interact with the force majeure clause in Contract B?")

The output is a JSON "golden dataset" that becomes the benchmark used by
pytest and DeepEval in CI/CD.

Why "adversarial"?
  The agent is instructed to find questions that are HARD for the RAG system --
  questions that require precise retrieval and reasoning, not just keyword matching.
  Easy questions don't expose system weaknesses.

Usage
-----
  from src.agents.adversarial_lawyer import AdversarialLawyerAgent
  agent = AdversarialLawyerAgent()
  dataset = await agent.generate(min_questions=50)
  agent.save(dataset, "golden_dataset/golden.json")
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import structlog
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient

from src.config import settings
from src.models import GoldenDatasetEntry

log = structlog.get_logger(__name__)


_SINGLE_HOP_PROMPT = """You are a demanding litigation attorney reviewing an internal legal document.
Your task: generate ONE challenging question that:
  1. Can be answered using ONLY the text below
  2. Requires precise reading (not just keyword search)
  3. Asks about a specific clause, date, party obligation, or dollar amount
  4. Would expose a RAG system's weaknesses if it retrieved the wrong chunk

Document: {filename}
Text excerpt:
\"\"\"
{text}
\"\"\"

Respond in JSON:
{{
  "question": "<your question>",
  "reference_context": "<the exact quote from the text that answers it, verbatim>",
  "expected_answer": "<a concise, precise answer citing the specific clause/number/date>"
}}
"""

_MULTI_HOP_PROMPT = """You are a senior partner at a law firm analyzing two internal documents.
Your task: generate ONE complex multi-hop question that:
  1. REQUIRES reading BOTH documents to answer
  2. Involves reasoning about how a clause in one document interacts with a clause in the other
  3. Cannot be answered from either document alone
  4. Tests the RAG system's ability to synthesize information across documents

Document A: {filename_a}
Excerpt A:
\"\"\"
{text_a}
\"\"\"

Document B: {filename_b}
Excerpt B:
\"\"\"
{text_b}
\"\"\"

Respond in JSON:
{{
  "question": "<your multi-hop question>",
  "reference_context": "<combined relevant text from both excerpts>",
  "expected_answer": "<precise answer citing both documents>"
}}
"""


class AdversarialLawyerAgent:
    """Generates a golden evaluation dataset from ingested legal documents."""

    def __init__(self) -> None:
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _fetch_sample_chunks(self, n: int = 200) -> list[dict]:
        """Retrieve a sample of chunks from Qdrant for question generation."""
        client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        try:
            results, _ = await client.scroll(
                collection_name=settings.qdrant_collection_name,
                limit=n,
                with_payload=True,
                with_vectors=False,
            )
            return [p.payload for p in results if p.payload]
        finally:
            await client.close()

    async def _generate_single_hop(self, chunk: dict) -> GoldenDatasetEntry | None:
        """Ask the LLM to generate one single-hop Q&A from a single chunk."""
        text = chunk.get("text", "")
        if len(text) < 200:
            return None

        prompt = _SINGLE_HOP_PROMPT.format(
            filename=chunk.get("filename", "unknown"),
            text=text[:1500],
        )
        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return GoldenDatasetEntry(
                question=data["question"],
                reference_context=data["reference_context"],
                expected_answer=data["expected_answer"],
                source_document_ids=[chunk.get("document_id", "")],
                is_multi_hop=False,
            )
        except Exception as exc:
            log.warning("single_hop_generation_failed", error=str(exc))
            return None

    async def _generate_multi_hop(
        self, chunk_a: dict, chunk_b: dict
    ) -> GoldenDatasetEntry | None:
        """Generate one multi-hop Q&A requiring reasoning over two chunks."""
        if chunk_a.get("document_id") == chunk_b.get("document_id"):
            return None  # Must be from different documents

        prompt = _MULTI_HOP_PROMPT.format(
            filename_a=chunk_a.get("filename", "doc_a"),
            text_a=chunk_a.get("text", "")[:800],
            filename_b=chunk_b.get("filename", "doc_b"),
            text_b=chunk_b.get("text", "")[:800],
        )
        try:
            response = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            return GoldenDatasetEntry(
                question=data["question"],
                reference_context=data["reference_context"],
                expected_answer=data["expected_answer"],
                source_document_ids=[
                    chunk_a.get("document_id", ""),
                    chunk_b.get("document_id", ""),
                ],
                is_multi_hop=True,
            )
        except Exception as exc:
            log.warning("multi_hop_generation_failed", error=str(exc))
            return None

    async def generate(self, min_questions: int = 50) -> list[GoldenDatasetEntry]:
        """
        Generate a golden dataset with at least *min_questions* entries.
        Roughly 70% single-hop, 30% multi-hop.
        """
        log.info("golden_dataset_generation_started", target=min_questions)
        chunks = await self._fetch_sample_chunks(n=200)

        if not chunks:
            log.warning("no_chunks_found_in_qdrant")
            return []

        dataset: list[GoldenDatasetEntry] = []
        single_hop_target = int(min_questions * 0.7)
        multi_hop_target = min_questions - single_hop_target

        # Single-hop: one chunk at a time
        random.shuffle(chunks)
        for chunk in chunks[:single_hop_target * 2]:  # 2x to account for failures
            if len(dataset) >= single_hop_target:
                break
            entry = await self._generate_single_hop(chunk)
            if entry:
                dataset.append(entry)

        # Multi-hop: pair chunks from different documents
        doc_chunk_map: dict[str, list[dict]] = {}
        for chunk in chunks:
            doc_id = chunk.get("document_id", "")
            doc_chunk_map.setdefault(doc_id, []).append(chunk)

        doc_ids = list(doc_chunk_map.keys())
        multi_hop_generated = 0
        for _ in range(multi_hop_target * 3):
            if multi_hop_generated >= multi_hop_target or len(doc_ids) < 2:
                break
            doc_a, doc_b = random.sample(doc_ids, 2)
            chunk_a = random.choice(doc_chunk_map[doc_a])
            chunk_b = random.choice(doc_chunk_map[doc_b])
            entry = await self._generate_multi_hop(chunk_a, chunk_b)
            if entry:
                dataset.append(entry)
                multi_hop_generated += 1

        log.info(
            "golden_dataset_generation_complete",
            total=len(dataset),
            single_hop=single_hop_target,
            multi_hop=multi_hop_generated,
        )
        return dataset

    def save(self, dataset: list[GoldenDatasetEntry], path: str | Path) -> None:
        """Persist the golden dataset to a JSON file."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = [entry.model_dump(mode="json") for entry in dataset]
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("golden_dataset_saved", path=str(output), entries=len(data))

    def load(self, path: str | Path) -> list[GoldenDatasetEntry]:
        """Load a previously saved golden dataset."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return [GoldenDatasetEntry.model_validate(entry) for entry in raw]
