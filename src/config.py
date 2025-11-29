"""
Configuration management for the multi-agent routing system.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # LLM Configuration
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_base_url: Optional[str] = Field(None, env="OPENROUTER_BASE_URL")
    model_name: str = Field(
        "google/gemini-2.0-flash-exp:free", env="MODEL_NAME")

    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = Field(None, env="LANGFUSE_SECRET_KEY")
    langfuse_public_key: Optional[str] = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_base_url: str = Field(
        "https://cloud.langfuse.com", env="LANGFUSE_BASE_URL")

    # RAG Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    similarity_top_k: int = Field(3, env="SIMILARITY_TOP_K")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # Document Paths
    data_dir: str = Field("./data", env="DATA_DIR")
    hr_docs_dir: str = Field("./data/hr_docs", env="HR_DOCS_DIR")
    tech_docs_dir: str = Field("./data/tech_docs", env="TECH_DOCS_DIR")
    finance_docs_dir: str = Field(
        "./data/finance_docs", env="FINANCE_DOCS_DIR")

    # Application Configuration
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # Agent Configuration
    department_classes: List[str] = ["hr", "tech", "finance"]
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")

    # Evaluator Configuration
    evaluator_model: str = Field("gpt-3.5-turbo", env="EVALUATOR_MODEL")
    quality_dimensions: List[str] = ["relevance", "completeness", "accuracy"]

    # Mock Mode Configuration
    mock_mode: bool = Field(False, env="MOCK_MODE")

    # Cache Configuration
    cache_dir: str = Field("./cache", env="CACHE_DIR")
    embeddings_cache_file: str = Field("embeddings.npy", env="EMBEDDINGS_CACHE_FILE")
    faiss_index_file: str = Field("faiss.index", env="FAISS_INDEX_FILE")
    metadata_cache_file: str = Field("metadata.json", env="METADATA_CACHE_FILE")

    # Persistent FAISS Storage Configuration
    store_dir: str = Field("./store", env="STORE_DIR")
    faiss_indices_dir: str = Field("./store/faiss_indices", env="FAISS_INDICES_DIR")
    embeddings_dir: str = Field("./store/embeddings", env="EMBEDDINGS_DIR")
    metadata_dir: str = Field("./store/metadata", env="METADATA_DIR")
    use_persistent_storage: bool = Field(True, env="USE_PERSISTENT_STORAGE")
    force_rebuild_indices: bool = Field(False, env="FORCE_REBUILD_INDICES")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields in .env
    }


# Global settings instance
settings = Settings()
