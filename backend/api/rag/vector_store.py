from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ...utils.logger import logger
import os

router = APIRouter()

# Initialize Qdrant client
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
default_collection_name = os.getenv("QDRANT_COLLECTION", "textbook_content")

try:
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=10
    )
except Exception as e:
    logger.warning(f"Could not connect to Qdrant: {e}. Vector store operations will fail until Qdrant is available.")
    qdrant_client = None

class CollectionInfo(BaseModel):
    name: str
    vector_size: int
    distance: str
    point_count: int

class VectorStoreHealth(BaseModel):
    connected: bool
    collections: List[CollectionInfo]
    status: str

@router.get("/health", response_model=VectorStoreHealth)
async def vector_store_health():
    """
    Check the health and status of the vector store
    """
    try:
        if qdrant_client:
            try:
                # Try to get collections to verify connection
                collections_response = qdrant_client.get_collections()

                collections_info = []
                for collection in collections_response.collections:
                    try:
                        # Get collection info
                        info = qdrant_client.get_collection(collection.name)
                        collections_info.append(
                            CollectionInfo(
                                name=collection.name,
                                vector_size=info.config.params.vectors.size,
                                distance=info.config.params.vectors.distance.value,
                                point_count=info.points_count
                            )
                        )
                    except:
                        # If we can't get detailed info, provide basic info
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
                logger.error(f"Error checking Qdrant health: {str(e)}")
                return VectorStoreHealth(
                    connected=False,
                    collections=[],
                    status=f"connection error: {str(e)}"
                )
        else:
            return VectorStoreHealth(
                connected=False,
                collections=[],
                status="Qdrant client not initialized - check configuration"
            )
    except Exception as e:
        logger.error(f"Error in vector store health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector store health check failed: {str(e)}")

@router.post("/create-collection")
async def create_collection(collection_name: str, vector_size: int = 384, distance: str = "Cosine"):
    """
    Create a new collection in the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        distance_enum = models.Distance[distance.upper()]

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=distance_enum)
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
        raise HTTPException(status_code=500, detail=f"Collection creation failed: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Collection deletion failed: {str(e)}")

@router.get("/collections")
async def list_collections():
    """
    List all collections in the vector store
    """
    try:
        if qdrant_client:
            try:
                collections_response = qdrant_client.get_collections()
                collection_names = [col.name for col in collections_response.collections]
                return {"collections": collection_names}
            except Exception as e:
                logger.error(f"Error listing collections: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Qdrant client not available")
    except Exception as e:
        logger.error(f"Error in collections listing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collections listing failed: {str(e)}")

@router.post("/index-textbook-content")
async def index_textbook_content():
    """
    Index all textbook content into the vector store
    """
    try:
        if not qdrant_client:
            raise HTTPException(status_code=500, detail="Qdrant client not available")

        # This would typically integrate with the content parser to index all textbook content
        # For now, we'll return a message indicating this functionality would be implemented
        return {
            "status": "success",
            "message": "Textbook content indexing would be implemented here, integrating with content parser and embedding service"
        }
    except Exception as e:
        logger.error(f"Error in textbook content indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Textbook content indexing failed: {str(e)}")