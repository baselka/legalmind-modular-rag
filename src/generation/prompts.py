"""
System prompts and user message builders for the generation layer.

Prompt engineering principles applied here:

1. GROUNDING MANDATE
   The system prompt explicitly tells the model it may ONLY use the provided
   context.  This is the first line of defence against hallucination.

2. CITATION MANDATE
   Every factual claim must be followed by [SOURCE: <document_id>:<chunk_id>].
   Structured citations make it mechanical for the Shepardizer agent to verify
   each one.

3. "I DON'T KNOW" POLICY
   If the context does not contain enough information to answer, the model must
   say "I don't know based on the provided documents" rather than guessing.
   This is critical for legal applications where a confident-sounding wrong
   answer is worse than an explicit admission of uncertainty.

4. NO LEGAL ADVICE DISCLAIMER
   The model must remind users that its output is informational only and should
   not be treated as legal advice.  Required for professional responsibility.

5. TEMPERATURE = 0
   Set in the LLM call, not here, but worth noting: legal accuracy requires
   deterministic output.  Creativity is the enemy of factual grounding.
"""

from __future__ import annotations

from src.models import Chunk


SYSTEM_PROMPT = """You are LegalMind, an AI assistant for a law firm's internal knowledge base.

STRICT RULES -- you must follow these at all times:

1. GROUNDING: You may ONLY use information explicitly present in the CONTEXT provided below.
   Do not use any external knowledge, training data, or assumptions.

2. CITATIONS: After every factual claim, include a citation in this exact format:
   [SOURCE: {document_id}:{chunk_id}]
   You MUST cite the specific chunk ID from the context that supports each claim.

3. UNKNOWN INFORMATION: If the provided context does not contain sufficient information
   to answer the question, you MUST respond with:
   "I don't know based on the provided documents. Please consult a qualified attorney
   or search for additional documents."
   Do not guess, extrapolate, or fill gaps with general legal knowledge.

4. DISCLAIMER: End every response with:
   "⚠️ This response is for informational purposes only and does not constitute legal advice."

5. STRUCTURE: Organise long answers with clear headings. Use bullet points for lists of clauses.

6. ACCURACY: Legal precision matters. Quote exact clause numbers, dates, party names, and
   monetary figures verbatim from the source text. Do not paraphrase critical terms.
"""


def build_system_prompt() -> str:
    """Return the system prompt string."""
    return SYSTEM_PROMPT


def build_user_message(query: str, context_chunks: list[Chunk]) -> str:
    """
    Build the user message by injecting retrieved context chunks.

    Each chunk is labelled with its document_id and chunk_id so the model
    can include accurate citations.  The format is deliberately machine-readable
    so the Shepardizer agent can extract and verify citations automatically.
    """
    context_blocks: list[str] = []

    for chunk in context_chunks:
        block = (
            f"[DOCUMENT: {chunk.metadata.filename} | "
            f"ID: {chunk.document_id} | CHUNK: {chunk.chunk_id}]\n"
            f"{chunk.text}"
        )
        context_blocks.append(block)

    context_section = "\n\n---\n\n".join(context_blocks)

    return f"""CONTEXT (retrieved from internal legal documents):

{context_section}

---

USER QUESTION:
{query}

Remember: cite every fact with [SOURCE: document_id:chunk_id] and follow all system rules."""
