"""
Qdrant Vector Store Service for Phase 2
Handles vector storage and retrieval using Qdrant Cloud with environment-based configuration
"""
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel
from ...utils.logger import logger


class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    url: str
    api_key: str
    collection_name: str = "textbook_content"
    vector_size: int = 1024  # Cohere embeddings are 1024-dim for embed-english-v3.0


class VectorRecord(BaseModel):
    """Represents a vector record with metadata"""
    id: str
    vector: List[float]
    content: str
    metadata: Dict[str, Any]


class QdrantVectorStoreService:
    """
    Service class for managing vector storage using Qdrant Cloud
    """

    def __init__(self):
        """Initialize the Qdrant vector store service"""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY environment variables are required")

        self.config = VectorStoreConfig(
            url=url,
            api_key=api_key,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME", "textbook_content"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
        )

        try:
            self.client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=30
            )
            logger.info("✅ Qdrant Vector Store Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {str(e)}")
            raise

    async def ensure_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Ensure the specified collection exists, create if it doesn't

        Args:
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            True if collection exists or was created successfully
        """
        try:
            collection_name = collection_name or self.config.collection_name

            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                logger.info(f"Collection {collection_name} already exists")
                return True
            except:
                # Collection doesn't exist, create it
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.vector_size,
                        distance=models.Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
                return True

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    async def store_embedding(self, record: VectorRecord, collection_name: Optional[str] = None) -> bool:
        """
        Store a single embedding with metadata in the vector store

        Args:
            record: Vector record to store
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            True if storage was successful
        """
        try:
            collection_name = collection_name or self.config.collection_name
            await self.ensure_collection_exists(collection_name)

            point = models.PointStruct(
                id=record.id,
                vector=record.vector,
                payload={
                    "content": record.content,
                    "metadata": record.metadata
                }
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )

            logger.info(f"Stored vector with ID: {record.id}")
            return True

        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise

    async def store_embeddings(self, records: List[VectorRecord], collection_name: Optional[str] = None) -> bool:
        """
        Store multiple embeddings with metadata in the vector store

        Args:
            records: List of vector records to store
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            True if storage was successful
        """
        try:
            collection_name = collection_name or self.config.collection_name
            await self.ensure_collection_exists(collection_name)

            points = []
            for record in records:
                point = models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload={
                        "content": record.content,
                        "metadata": record.metadata
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Stored {len(records)} vectors")
            return True

        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise

    async def search_similar(self,
                           query_vector: List[float],
                           top_k: int = 5,
                           filters: Optional[Dict[str, Any]] = None,
                           collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the vector store

        Args:
            query_vector: Vector to search for similar ones
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            List of similar vectors with metadata and scores
        """
        try:
            collection_name = collection_name or self.config.collection_name

            # Prepare filters
            search_filter = None
            if filters:
                filter_conditions = []
                for key, value in filters.items():
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
                    search_filter = models.Filter(must=filter_conditions)

            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("content", ""),
                    "metadata": hit.payload.get("metadata", {})
                })

            logger.info(f"Search completed, found {len(results)} similar vectors")
            return results

        except Exception as e:
            logger.error(f"Error searching similar vectors: {str(e)}")
            raise

    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a collection

        Args:
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_name = collection_name or self.config.collection_name

            info = self.client.get_collection(collection_name)

            stats = {
                "collection_name": collection_name,
                "point_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
                "status": "active"
            }

            logger.info(f"Retrieved stats for collection: {collection_name}")
            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    async def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection from the vector store

        Args:
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            True if deletion was successful
        """
        try:
            collection_name = collection_name or self.config.collection_name

            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    async def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Clear all points from a collection (but keep the collection)

        Args:
            collection_name: Name of the collection (uses default if not provided)

        Returns:
            True if clearing was successful
        """
        try:
            collection_name = collection_name or self.config.collection_name

            # Get all point IDs
            all_points = self.client.scroll(
                collection_name=collection_name,
                limit=10000  # Adjust as needed
            )[0]

            if all_points:
                point_ids = [point.id for point in all_points]
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )

            logger.info(f"Cleared all points from collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise