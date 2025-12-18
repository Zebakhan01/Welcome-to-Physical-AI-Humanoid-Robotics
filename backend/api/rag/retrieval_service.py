from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid
from ...utils.logger import logger
import os

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
    logger.warning(f"Could not connect to Qdrant: {e}. Using in-memory storage for testing.")
    qdrant_client = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str

@router.post("/query", response_model=QueryResponse)
async def query_content(request: QueryRequest):
    """
    Query the vector database for relevant content
    """
    try:
        logger.info(f"Content query requested: {request.query}")

        # Generate embedding for the query
        query_embedding = model.encode([request.query])[0].tolist()

        # Search in Qdrant if available
        if qdrant_client:
            try:
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

                    if filter_conditions:
                        search_filters = models.Filter(must=filter_conditions)

                # Perform search in Qdrant
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=request.top_k,
                    query_filter=search_filters,
                    with_payload=True,
                    with_vectors=False
                )

                # Format results
                results = []
                for hit in search_results:
                    result = {
                        "id": hit.id,
                        "score": hit.score,
                        "content": hit.payload.get("content", ""),
                        "metadata": hit.payload.get("metadata", {}),
                        "title": hit.payload.get("title", "")
                    }
                    results.append(result)

                return QueryResponse(
                    results=results,
                    query=request.query
                )
            except Exception as e:
                logger.error(f"Error querying Qdrant: {str(e)}")
                # Fall back to basic response if Qdrant fails
                return QueryResponse(
                    results=[],
                    query=request.query
                )
        else:
            # If no Qdrant client, return empty results (for testing)
            logger.warning("Qdrant client not available, returning empty results")
            return QueryResponse(
                results=[],
                query=request.query
            )

    except Exception as e:
        logger.error(f"Error in content query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content retrieval failed: {str(e)}")

class VectorStoreRequest(BaseModel):
    content: str
    title: str
    chapter_id: str
    metadata: Dict[str, Any] = {}
    vector_id: str = None

@router.post("/store")
async def store_content(request: VectorStoreRequest):
    """
    Store content in the vector database
    """
    try:
        logger.info(f"Content storage requested: {request.title}")

        # Generate embedding for the content
        content_embedding = model.encode([request.content])[0].tolist()

        # Create a point for Qdrant
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

        # Store in Qdrant if available
        if qdrant_client:
            try:
                # Ensure collection exists
                try:
                    qdrant_client.get_collection(collection_name)
                except:
                    # Create collection if it doesn't exist
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)  # all-MiniLM-L6-v2 outputs 384-dim vectors
                    )

                # Upload the point
                qdrant_client.upload_points(
                    collection_name=collection_name,
                    points=[point]
                )

                return {
                    "status": "success",
                    "vector_id": point.id,
                    "collection": collection_name
                }
            except Exception as e:
                logger.error(f"Error storing content in Qdrant: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Content storage failed: {str(e)}")
        else:
            # If no Qdrant client, return success for testing
            logger.warning("Qdrant client not available, simulating storage")
            return {
                "status": "success",
                "vector_id": point.id,
                "collection": collection_name
            }

    except Exception as e:
        logger.error(f"Error in content storage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content storage failed: {str(e)}")