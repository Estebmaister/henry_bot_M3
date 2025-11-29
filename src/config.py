"""
Configuration management for the multi-agent routing system.
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # LLM Configuration
    openrouter_api_key: str
    openrouter_base_url: Optional[str] = None
    model_name: str = "google/gemini-2.0-flash-exp:free"

    # Langfuse Configuration
    langfuse_secret_key: Optional[str] = Field(None)
    langfuse_public_key: Optional[str] = Field(None)
    langfuse_base_url: str = Field(
        "https://cloud.langfuse.com")

    # RAG Configuration
    embedding_model: str = Field("all-MiniLM-L6-v2")
    similarity_top_k: int = Field(3)
    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(200)

    # Document Paths
    data_dir: str = Field("./data")
    hr_docs_dir: str = Field("./data/hr_docs")
    tech_docs_dir: str = Field("./data/tech_docs")
    finance_docs_dir: str = Field(
        "./data/finance_docs")

    # Application Configuration
    debug: bool = Field(False)
    log_level: str = Field("INFO")

    # Agent Configuration
    department_classes: List[str] = ["hr", "tech", "finance"]
    confidence_threshold: float = Field(0.7)

    # Evaluator Configuration
    evaluator_model: str = Field("gpt-3.5-turbo")
    quality_dimensions: List[str] = ["relevance", "completeness", "accuracy"]

    # Mock Mode Configuration
    mock_mode: bool = Field(False)

    # Cache Configuration
    cache_dir: str = Field("./cache")
    embeddings_cache_file: str = Field("embeddings.npy")
    faiss_index_file: str = Field("faiss.index")
    metadata_cache_file: str = Field("metadata.json")

    # Persistent FAISS Storage Configuration
    store_dir: str = Field("./store")
    faiss_indices_dir: str = Field("./store/faiss_indices")
    embeddings_dir: str = Field("./store/embeddings")
    metadata_dir: str = Field("./store/metadata")
    use_persistent_storage: bool = Field(True)
    force_rebuild_indices: bool = Field(False)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Ignore extra fields in .env
    }


# Global settings instance
settings = Settings()
