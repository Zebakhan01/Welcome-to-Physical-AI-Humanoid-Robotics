from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from ...utils.logger import logger

router = APIRouter()

# Initialize the sentence transformer model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: str = "all-MiniLM-L6-v2"

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str

@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """
    Generate embeddings for the given texts using sentence transformer model
    """
    try:
        logger.info(f"Embedding generation requested for {len(request.texts)} texts")

        # Generate embeddings using the sentence transformer model
        embeddings = model.encode(request.texts)

        # Convert to list format for JSON serialization
        embeddings_list = [embedding.tolist() for embedding in embeddings]

        return EmbeddingResponse(
            embeddings=embeddings_list,
            model=request.model
        )
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

class TextChunk(BaseModel):
    text: str
    metadata: dict = {}

class ChunkEmbeddingRequest(BaseModel):
    chunks: List[TextChunk]
    collection_name: str = "textbook_content"

@router.post("/chunk-embeddings")
async def generate_and_store_chunk_embeddings(request: ChunkEmbeddingRequest):
    """
    Generate embeddings for text chunks and prepare for storage
    """
    try:
        logger.info(f"Chunk embedding generation requested for {len(request.chunks)} chunks")

        # Extract texts for embedding generation
        texts = [chunk.text for chunk in request.chunks]

        # Generate embeddings using the sentence transformer model
        embeddings = model.encode(texts)

        # Prepare the result with embeddings and metadata
        processed_chunks = []
        for i, chunk in enumerate(request.chunks):
            processed_chunk = {
                "embedding": embeddings[i].tolist(),
                "metadata": chunk.metadata,
                "text": chunk.text,
                "text_hash": hash(chunk.text)  # For potential deduplication
            }
            processed_chunks.append(processed_chunk)

        return {"processed_chunks": processed_chunks, "collection": request.collection_name}
    except Exception as e:
        logger.error(f"Error in chunk embedding generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunk embedding generation failed: {str(e)}")