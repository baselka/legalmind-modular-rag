"""
Document chunking strategies.

Primary: SemanticSplitterNodeParser (LlamaIndex)
  Measures embedding cosine distance between consecutive sentences.
  Only splits when the semantic distance crosses the configured percentile
  threshold, keeping legally-related content in the same chunk.

Fallback: SentenceSplitter (recursive, fixed-size)
  Used when the document is too short for semantic splitting to be meaningful,
  or as a fast alternative when semantic splitting is disabled via config.

Why semantic over fixed-size for legal text?
  A 512-token fixed window often bisects a clause mid-sentence.  A cross-
  reference like "as defined in Section 14.2(b) above" loses its antecedent
  if that section is in a different chunk.  Semantic chunking keeps clauses
  together because the embedding distance between the start and end of a
  clause is small, while the distance to the next clause is large.
"""

from __future__ import annotations

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument, TextNode

from src.config import settings
from src.models import Chunk, DocumentMetadata


def _llama_doc(text: str, metadata: DocumentMetadata) -> LlamaDocument:
    return LlamaDocument(
        text=text,
        metadata={
            "document_id": metadata.document_id,
            "filename": metadata.filename,
            "document_type": metadata.document_type.value,
        },
        doc_id=metadata.document_id,
    )


def _node_to_chunk(node: TextNode, metadata: DocumentMetadata, index: int) -> Chunk:
    # Prepend filename and other metadata to help with retrieval of factual content
    # This prevents uninformative headers from being the only chunks with document identity.
    prefix = f"[Document: {metadata.filename}] "
    augmented_text = prefix + node.get_content()
    
    return Chunk(
        chunk_id=node.node_id,
        document_id=metadata.document_id,
        text=augmented_text,
        chunk_index=index,
        metadata=metadata,
    )


def chunk_document_semantic(text: str, metadata: DocumentMetadata) -> list[Chunk]:
    """
    Semantic chunking via LlamaIndex SemanticSplitterNodeParser.
    Falls back to fixed-size chunking if the import fails or text is too short.
    """
    try:
        from llama_index.packs.node_parser_semantic_chunking import SemanticSplitterNodeParser
        from llama_index.embeddings.openai import OpenAIEmbedding

        embed_model = OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
        splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=settings.semantic_breakpoint_percentile,
            buffer_size=1,
        )
        doc = _llama_doc(text, metadata)
        nodes = splitter.get_nodes_from_documents([doc])
        return [_node_to_chunk(n, metadata, i) for i, n in enumerate(nodes)]

    except Exception:
        # Graceful fallback to fixed-size chunking
        return chunk_document_fixed(text, metadata)


def chunk_document_fixed(text: str, metadata: DocumentMetadata) -> list[Chunk]:
    """
    Fixed-size recursive chunking (512 tokens, ~10% overlap).
    Fast and deterministic -- used as a fallback and in tests.
    """
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    doc = _llama_doc(text, metadata)
    nodes = splitter.get_nodes_from_documents([doc])
    return [_node_to_chunk(n, metadata, i) for i, n in enumerate(nodes)]


def chunk_document(text: str, metadata: DocumentMetadata, semantic: bool = True) -> list[Chunk]:
    """Entry point: choose the chunking strategy based on *semantic* flag."""
    if semantic:
        return chunk_document_semantic(text, metadata)
    return chunk_document_fixed(text, metadata)
