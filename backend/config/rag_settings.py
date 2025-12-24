"""
Configuration settings for the Phase 2 RAG backend
"""
import os
from typing import Optional


class Settings:
    """Application settings for Phase 2 RAG services"""

    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Cohere settings
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    COHERE_EMBEDDING_MODEL: str = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
    COHERE_GENERATION_MODEL: str = os.getenv("COHERE_GENERATION_MODEL", "command-r-plus")

    # Qdrant settings
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))

    # Database settings
    NEON_DATABASE_URL: Optional[str] = os.getenv("NEON_DATABASE_URL")

    # Text processing settings
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
    DEFAULT_OVERLAP_SIZE: int = int(os.getenv("DEFAULT_OVERLAP_SIZE", "100"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))


settings = Settings()