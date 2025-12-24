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
import os
from ...utils.logger import logger
from ..content.content_loader import ContentChunk

router = APIRouter()

# ===============================
# EMBEDDING MODEL
# ===============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ===============================
# QDRANT CONFIG
# ===============================
QDRANT_URL = os.getenv("QDRANT_URL", "https://1798859e-3ed7-4dd9-a919-6f737f25a0d5.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7q9Z1c6vzvjz__OLVgwsH7J6D5b4d8ez3QKe6P62rIU")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

# ===============================
# QDRANT CLIENT
# ===============================
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10,
    )
    logger.info("Connected to Qdrant Cloud successfully")
except Exception as e:
    logger.error(f"Qdrant connection failed: {e}")
    qdrant_client = None


# ===============================
# REQUEST / RESPONSE MODELS
# ===============================
class IndexContentRequest(BaseModel):
    chunks: List[ContentChunk]
    collection_name: str = DEFAULT_COLLECTION
    batch_size: int = 10


class IndexContentResponse(BaseModel):
    indexed_chunks: int
    failed_chunks: int
    collection_name: str
    processing_time: float
    details: List[Dict[str, Any]]


class IndexQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection_name: str = DEFAULT_COLLECTION
    filters: Dict[str, Any] = {}


class IndexedContentResult(BaseModel):
    id: str
    content: str
    title: str
    source_file: str
    chapter_id: str
    score: float
    metadata: Dict[str, Any]


class IndexQueryResponse(BaseModel):
    results: List[IndexedContentResult]
    query: str
    collection_name: str


# ===============================
# INDEX CONTENT
# ===============================
@router.post("/index-chunks", response_model=IndexContentResponse)
async def index_content_chunks(request: IndexContentRequest):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    import time
    start_time = time.time()

    try:
        # Ensure collection exists
        try:
            qdrant_client.get_collection(request.collection_name)
        except:
            qdrant_client.create_collection(
                collection_name=request.collection_name,
                vectors_config=models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                ),
            )
            logger.info(f"Created Qdrant collection: {request.collection_name}")

        indexed = 0
        failed = 0
        details = []

        for i in range(0, len(request.chunks), request.batch_size):
            batch = request.chunks[i : i + request.batch_size]
            texts = [chunk.content for chunk in batch]

            try:
                embeddings = model.encode(texts)

                points = []
                for chunk, embedding in zip(batch, embeddings):
                    points.append(
                        models.PointStruct(
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
                                "chunk_id": chunk.id,
                            },
                        )
                    )

                qdrant_client.upload_points(
                    collection_name=request.collection_name, points=points
                )
                indexed += len(batch)

            except Exception as e:
                failed += len(batch)
                for chunk in batch:
                    details.append(
                        {
                            "chunk_id": chunk.id,
                            "title": chunk.title,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

        return IndexContentResponse(
            indexed_chunks=indexed,
            failed_chunks=failed,
            collection_name=request.collection_name,
            processing_time=time.time() - start_time,
            details=details,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# QUERY INDEX
# ===============================
@router.post("/query-indexed", response_model=IndexQueryResponse)
async def query_indexed_content(request: IndexQueryRequest):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    query_embedding = model.encode([request.query])[0].tolist()

    search_results = qdrant_client.search(
        collection_name=request.collection_name,
        query_vector=query_embedding,
        limit=request.top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []
    for hit in search_results:
        payload = hit.payload
        results.append(
            IndexedContentResult(
                id=str(hit.id),
                content=payload.get("content", ""),
                title=payload.get("title", ""),
                source_file=payload.get("source_file", ""),
                chapter_id=payload.get("chapter_id", ""),
                score=hit.score,
                metadata=payload.get("metadata", {}),
            )
        )

    return IndexQueryResponse(
        results=results,
        query=request.query,
        collection_name=request.collection_name,
    )


# ===============================
# STATS
# ===============================
class IndexStatsRequest(BaseModel):
    collection_name: str = DEFAULT_COLLECTION


class IndexStatsResponse(BaseModel):
    collection_name: str
    point_count: int
    vectors_size: int


@router.post("/stats", response_model=IndexStatsResponse)
async def get_index_stats(request: IndexStatsRequest):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not available")

    info = qdrant_client.get_collection(request.collection_name)

    return IndexStatsResponse(
        collection_name=request.collection_name,
        point_count=info.points_count,
        vectors_size=info.config.params.vectors.size,
    )
