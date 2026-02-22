"""
Application configuration via Pydantic Settings.
All values can be overridden via environment variables or a .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key")
    openai_embedding_model: str = "text-embedding-3-large"
    openai_chat_model: str = "gpt-4o"

    # --- Cohere ---
    cohere_api_key: str = Field(default="", description="Cohere API key for re-ranking")
    cohere_rerank_model: str = "rerank-v3.5"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "legal_documents"
    qdrant_timeout: int = 120  # seconds for large upserts
    qdrant_upsert_batch_size: int = 100  # points per upsert batch

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl_seconds: int = 86400
    semantic_cache_similarity_threshold: float = 0.95

    # --- Retrieval ---
    retrieval_top_k: int = 30
    rerank_top_n: int = 7

    # --- Chunking ---
    chunk_size: int = 512
    chunk_overlap: int = 102
    semantic_breakpoint_percentile: int = 95

    # --- Evaluation thresholds ---
    eval_faithfulness_threshold: float = 0.9
    eval_relevance_threshold: float = 0.8
    eval_context_precision_threshold: float = 0.8

    # --- Application ---
    app_env: str = "development"
    log_level: str = "INFO"
    # "cohere" | "cross_encoder" | "none"  -- controls which reranker is injected
    reranker_type: str = "cohere"
    # HuggingFace model ID used by CrossEncoderReranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


# Module-level singleton so all imports share the same instance.
settings = Settings()  # type: ignore[call-arg]
