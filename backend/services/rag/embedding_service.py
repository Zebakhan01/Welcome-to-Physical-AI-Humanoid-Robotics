"""
Cohere Embedding Service for Phase 2
Handles embedding generation using Cohere API with environment-based configuration
"""
import os
import asyncio
from typing import List, Optional
import cohere
from pydantic import BaseModel
from ...utils.logger import logger


class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "embed-english-v3.0"
    input_type: str = "search_document"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    embedding_type: str


class CohereEmbeddingService:
    """
    Service class for generating embeddings using Cohere API
    """

    def __init__(self):
        """Initialize the Cohere embedding service"""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.client = cohere.Client(api_key)
        self.default_model = os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0")
        logger.info("✅ Cohere Embedding Service initialized")

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "search_document"
    ) -> List[List[float]]:
        """
        Generate embeddings for the given texts using Cohere API

        Args:
            texts: List of text strings to embed
            model: Cohere model to use (optional, defaults to configured model)
            input_type: Type of input for embedding context (search_document, search_query, etc.)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        try:
            model = model or self.default_model

            # Use asyncio to make the API call non-blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=texts,
                    model=model,
                    input_type=input_type
                )
            )

            embeddings = [embedding for embedding in response.embeddings]
            logger.info(f"Generated embeddings for {len(texts)} texts using model: {model}")

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        """
        Generate a single embedding for a text string

        Args:
            text: Text string to embed
            model: Cohere model to use (optional)

        Returns:
            Embedding as a list of floats
        """
        embeddings = await self.generate_embeddings([text], model)
        return embeddings[0]

    async def embed_texts(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple text strings

        Args:
            texts: List of text strings to embed
            model: Cohere model to use (optional)

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return await self.generate_embeddings(texts, model)