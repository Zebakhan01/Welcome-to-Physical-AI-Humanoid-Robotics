"""
Content Indexer for Phase 3: Integrating content loader with embeddings and Qdrant
This module provides functionality to take content chunks from the content loader,
generate embeddings, and store them in Qdrant vector database.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
from ...utils.logger import logger
import os
from ..content.content_loader import ContentChunk
import asyncio

router = APIRouter()

# Initialize the sentence transformer model for embedding generation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
collection_name = os.getenv("QDRANT_COLLECTION", "textbook_content")

try:
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=10
    )
except Exception as e:
    logger.warning(f"Could not connect to Qdrant: {e}. Embedding storage will fail until Qdrant is available.")
    qdrant_client = None


class IndexContentRequest(BaseModel):
    """Request to index content chunks into Qdrant"""
    chunks: List[ContentChunk]
    collection_name: str = "textbook_content"
    batch_size: int = 10  # Number of chunks to process in each batch


class IndexContentResponse(BaseModel):
    """Response for content indexing"""
    indexed_chunks: int
    failed_chunks: int
    collection_name: str
    processing_time: float
    details: List[Dict[str, Any]]


class IndexQueryRequest(BaseModel):
    """Request to query indexed content using embeddings"""
    query: str
    top_k: int = 5
    collection_name: str = "textbook_content"
    filters: Dict[str, Any] = {}


class IndexedContentResult(BaseModel):
    """Result of an indexed content query"""
    id: str
    content: str
    title: str
    source_file: str
    chapter_id: str
    score: float
    metadata: Dict[str, Any]


class IndexQueryResponse(BaseModel):
    """Response for indexed content query"""
    results: List[IndexedContentResult]
    query: str
    collection_name: str


@router.post("/index-chunks", response_model=IndexContentResponse)
async def index_content_chunks(request: IndexContentRequest):
    """
    Index content chunks into Qdrant vector database with embeddings
    """
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    try:
        import time
        start_time = time.time()

        logger.info(f"Indexing {len(request.chunks)} content chunks into Qdrant collection: {request.collection_name}")

        # Ensure collection exists
        try:
            qdrant_client.get_collection(request.collection_name)
        except:
            # Create collection if it doesn't exist
            qdrant_client.create_collection(
                collection_name=request.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)  # all-MiniLM-L6-v2 outputs 384-dim vectors
            )
            logger.info(f"Created new Qdrant collection: {request.collection_name}")

        indexed_count = 0
        failed_count = 0
        details = []

        # Process chunks in batches
        for i in range(0, len(request.chunks), request.batch_size):
            batch = request.chunks[i:i + request.batch_size]

            # Extract content for embedding generation
            texts = [chunk.content for chunk in batch]

            try:
                # Generate embeddings for the batch
                embeddings = model.encode(texts)

                # Create Qdrant points
                points = []
                for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    point = models.PointStruct(
                        id=chunk.id or str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload={
                            "content": chunk.content,
                            "title": chunk.title,
                            "source_file": chunk.source_file,
                            "chapter_id": chunk.chapter_id,
                            "section_level": chunk.section_level,
                            "word_count": chunk.word_count,
                            "char_count": chunk.char_count,
                            "metadata": chunk.metadata,
                            "chunk_id": chunk.id
                        }
                    )
                    points.append(point)

                # Upload points to Qdrant
                qdrant_client.upload_points(
                    collection_name=request.collection_name,
                    points=points
                )

                indexed_count += len(batch)
                logger.info(f"Indexed batch {i//request.batch_size + 1}: {len(batch)} chunks")

            except Exception as batch_error:
                logger.error(f"Error indexing batch {i//request.batch_size + 1}: {str(batch_error)}")
                failed_count += len(batch)

                # Add failure details
                for chunk in batch:
                    details.append({
                        "chunk_id": chunk.id,
                        "title": chunk.title,
                        "status": "failed",
                        "error": str(batch_error)
                    })

        processing_time = time.time() - start_time

        response = IndexContentResponse(
            indexed_chunks=indexed_count,
            failed_chunks=failed_count,
            collection_name=request.collection_name,
            processing_time=processing_time,
            details=details
        )

        logger.info(f"Content indexing completed. Indexed: {indexed_count}, Failed: {failed_count}, Time: {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error in content indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content indexing failed: {str(e)}")


@router.post("/query-indexed", response_model=IndexQueryResponse)
async def query_indexed_content(request: IndexQueryRequest):
    """
    Query content from the indexed vector database using embeddings
    """
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    try:
        logger.info(f"Querying indexed content: {request.query}")

        # Generate embedding for the query
        query_embedding = model.encode([request.query])[0].tolist()

        # Prepare filters if provided
        search_filters = None
        if request.filters:
            filter_conditions = []
            for key, value in request.filters.items():
                if isinstance(value, str):
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
                elif isinstance(value, (int, float)):
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            range=models.Range(gte=value, lte=value)
                        )
                    )
                elif isinstance(value, list):
                    # Handle list of values for "in" queries
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchAny(any=value)
                        )
                    )

            if filter_conditions:
                search_filters = models.Filter(must=filter_conditions)

        # Perform search in Qdrant
        search_results = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=query_embedding,
            limit=request.top_k,
            query_filter=search_filters,
            with_payload=True,
            with_vectors=False
        )

        # Format results
        results = []
        for hit in search_results:
            payload = hit.payload
            result = IndexedContentResult(
                id=hit.id,
                content=payload.get("content", ""),
                title=payload.get("title", ""),
                source_file=payload.get("source_file", ""),
                chapter_id=payload.get("chapter_id", ""),
                score=hit.score,
                metadata=payload.get("metadata", {})
            )
            results.append(result)

        response = IndexQueryResponse(
            results=results,
            query=request.query,
            collection_name=request.collection_name
        )

        logger.info(f"Indexed content query completed. Found {len(results)} results.")
        return response

    except Exception as e:
        logger.error(f"Error in indexed content query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Indexed content query failed: {str(e)}")


class BulkIndexRequest(BaseModel):
    """Request for bulk indexing from a source path"""
    source_path: str
    collection_name: str = "textbook_content"
    chunk_size: int = 1000
    overlap_size: int = 100
    batch_size: int = 10


@router.post("/bulk-index", response_model=IndexContentResponse)
async def bulk_index_content(request: BulkIndexRequest):
    """
    Bulk index content from a source path by loading, chunking, embedding, and storing in Qdrant
    """
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    try:
        logger.info(f"Bulk indexing content from: {request.source_path}")

        # Import content loader to load and chunk the content
        from ..content.content_loader import load_content, ContentLoadRequest

        # Load content using the content loader
        load_request = ContentLoadRequest(
            source_path=request.source_path,
            recursive=True,
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size
        )

        content_response = await load_content(load_request)

        # Index the loaded chunks
        index_request = IndexContentRequest(
            chunks=content_response.chunks,
            collection_name=request.collection_name,
            batch_size=request.batch_size
        )

        # Use the index_content_chunks function to do the actual indexing
        return await index_content_chunks(index_request)

    except Exception as e:
        logger.error(f"Error in bulk content indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Bulk content indexing failed: {str(e)}")


class IndexStatsRequest(BaseModel):
    """Request for index statistics"""
    collection_name: str = "textbook_content"


class IndexStatsResponse(BaseModel):
    """Response for index statistics"""
    collection_name: str
    point_count: int
    vectors_size: int
    segments_count: int
    indexed_vectors_count: int


@router.post("/stats", response_model=IndexStatsResponse)
async def get_index_stats(request: IndexStatsRequest):
    """
    Get statistics about the indexed content in Qdrant
    """
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    try:
        logger.info(f"Getting statistics for collection: {request.collection_name}")

        # Get collection info
        collection_info = qdrant_client.get_collection(request.collection_name)

        response = IndexStatsResponse(
            collection_name=request.collection_name,
            point_count=collection_info.points_count,
            vectors_size=collection_info.config.params.vectors.size if collection_info.config.params.vectors else 0,
            segments_count=len(collection_info.segments) if hasattr(collection_info, 'segments') else 0,
            indexed_vectors_count=collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else 0
        )

        logger.info(f"Index statistics retrieved for {request.collection_name}")
        return response

    except Exception as e:
        logger.error(f"Error getting index statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Index statistics retrieval failed: {str(e)}")