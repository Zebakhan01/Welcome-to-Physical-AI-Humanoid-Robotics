from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
import os
from ...utils.logger import logger

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
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

# ===============================
# QDRANT CLIENT
# ===============================
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )
    logger.info("Connected to Qdrant Cloud successfully")
except Exception as e:
    logger.warning(f"Could not connect to Qdrant: {e}")
    qdrant_client = None


# ===============================
# REQUEST / RESPONSE MODELS
# ===============================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str


# ===============================
# QUERY CONTENT
# ===============================
@router.post("/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    """
    Query the vector database for relevant content
    """
    try:
        logger.info(f"Content query requested: {request.query}")

        # Generate embedding for the query
        query_embedding = model.encode([request.query])[0].tolist()

        if not qdrant_client:
            logger.warning("Qdrant client not available, returning empty results")
            return QueryResponse(results=[], query=request.query)

        # Prepare filters
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

            if filter_conditions:
                search_filters = models.Filter(must=filter_conditions)

        # Perform search
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=request.top_k,
            query_filter=search_filters,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for hit in search_results:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "title": hit.payload.get("title", "")
            })

        return QueryResponse(results=results, query=request.query)

    except Exception as e:
        logger.error(f"Error in content query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content retrieval failed: {str(e)}")


# ===============================
# VECTOR STORE REQUEST
# ===============================
class VectorStoreRequest(BaseModel):
    content: str
    title: str
    chapter_id: str
    metadata: Dict[str, Any] = {}
    vector_id: str | None = None


# ===============================
# STORE CONTENT
# ===============================
@router.post("/store")
async def store_content(request: VectorStoreRequest):
    """
    Store content in the vector database
    """
    try:
        logger.info(f"Content storage requested: {request.title}")

        # Generate embedding
        content_embedding = model.encode([request.content])[0].tolist()

        point = models.PointStruct(
            id=request.vector_id or str(uuid.uuid4()),
            vector=content_embedding,
            payload={
                "content": request.content,
                "title": request.title,
                "chapter_id": request.chapter_id,
                "metadata": request.metadata
            }
        )

        if not qdrant_client:
            logger.warning("Qdrant client not available, simulating storage")
            return {
                "status": "success",
                "vector_id": point.id,
                "collection": COLLECTION_NAME
            }

        # Ensure collection exists
        try:
            qdrant_client.get_collection(COLLECTION_NAME)
        except:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

        # Upload point
        qdrant_client.upload_points(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        return {
            "status": "success",
            "vector_id": point.id,
            "collection": COLLECTION_NAME
        }

    except Exception as e:
        logger.error(f"Error in content storage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content storage failed: {str(e)}")
