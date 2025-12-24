from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from ...utils.logger import logger

router = APIRouter()

# ===============================
# QDRANT CONFIG
# ===============================
QDRANT_URL = os.getenv("QDRANT_URL", "https://1798859e-3ed7-4dd9-a919-6f737f25a0d5.us-east4-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7q9Z1c6vzvjz__OLVgwsH7J6D5b4d8ez3QKe6P62rIU")
DEFAULT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "textbook_content")

# ===============================
# QDRANT CLIENT
# ===============================
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=10
    )
    logger.info("Qdrant Cloud connected successfully")
except Exception as e:
    logger.warning(
        f"Could not connect to Qdrant: {e}. Vector store operations will fail until Qdrant is available."
    )
    qdrant_client = None


# ===============================
# MODELS
# ===============================
class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    distance: str
    point_count: int


class VectorStoreHealth(BaseModel):
    connected: bool
    collections: List[CollectionInfo]
    status: str


# ===============================
# HEALTH CHECK
# ===============================
@router.get("/health", response_model=VectorStoreHealth)
async def vector_store_health():
    """
    Check the health and status of the vector store
    """
    try:
        if not qdrant_client:
            return VectorStoreHealth(
                connected=False,
                collections=[],
                status="Qdrant client not initialized"
            )

        collections_response = qdrant_client.get_collections()
        collections_info = []

        for collection in collections_response.collections:
            try:
                info = qdrant_client.get_collection(collection.name)
                collections_info.append(
                    CollectionInfo(
                        name=collection.name,
                        vector_size=info.config.params.vectors.size,
                        distance=info.config.params.vectors.distance.value,
                        point_count=info.points_count
                    )
                )
            except Exception:
                collections_info.append(
                    CollectionInfo(
                        name=collection.name,
                        vector_size=0,
                        distance="unknown",
                        point_count=0
                    )
                )

        return VectorStoreHealth(
            connected=True,
            collections=collections_info,
            status="connected and operational"
        )

    except Exception as e:
        logger.error(f"Error in vector store health check: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Vector store health check failed: {str(e)}"
        )


# ===============================
# CREATE COLLECTION
# ===============================
@router.post("/create-collection")
async def create_collection(
    collection_name: str,
    vector_size: int = 384,
    distance: str = "Cosine"
):
    """
    Create a new collection in the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        distance_enum = models.Distance[distance.upper()]

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=distance_enum
            )
        )

        logger.info(f"Collection created: {collection_name}")

        return {
            "status": "success",
            "collection_name": collection_name,
            "vector_size": vector_size,
            "distance": distance
        }

    except Exception as e:
        logger.error(f"Error in collection creation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Collection creation failed: {str(e)}"
        )


# ===============================
# DELETE COLLECTION
# ===============================
@router.delete("/delete-collection/{collection_name}")
async def delete_collection(collection_name: str):
    """
    Delete a collection from the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        qdrant_client.delete_collection(collection_name)

        logger.info(f"Collection deleted: {collection_name}")

        return {
            "status": "success",
            "collection_name": collection_name
        }

    except Exception as e:
        logger.error(f"Error in collection deletion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Collection deletion failed: {str(e)}"
        )


# ===============================
# LIST COLLECTIONS
# ===============================
@router.get("/collections")
async def list_collections():
    """
    List all collections in the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        collections_response = qdrant_client.get_collections()
        collection_names = [col.name for col in collections_response.collections]

        return {"collections": collection_names}

    except Exception as e:
        logger.error(f"Error in collections listing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Collections listing failed: {str(e)}"
        )


# ===============================
# INDEX TEXTBOOK CONTENT (PLACEHOLDER)
# ===============================
@router.post("/index-textbook-content")
async def index_textbook_content():
    """
    Index all textbook content into the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        return {
            "status": "success",
            "message": "Textbook content indexing hook is ready (parser + embeddings integration pending)"
        }

    except Exception as e:
        logger.error(f"Error in textbook content indexing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Textbook content indexing failed: {str(e)}"
        )
