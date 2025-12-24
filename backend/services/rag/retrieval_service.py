"""
Retrieval Service for Phase 2
Handles similarity search and content retrieval with two modes:
- Mode A: Standard question → similarity search
- Mode B: Selected-text question → restricted retrieval
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ...utils.logger import logger


class RetrievalRequest(BaseModel):
    """Request for content retrieval"""
    query: str
    top_k: int = 5
    mode: str = "standard"  # "standard" or "selected_text"
    filters: Dict[str, Any] = {}
    selected_text: Optional[str] = None  # For selected-text mode


class RetrievedChunk(BaseModel):
    """Represents a retrieved content chunk"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class RetrievalResponse(BaseModel):
    """Response for content retrieval"""
    results: List[RetrievedChunk]
    query: str
    retrieved_count: int
    mode: str


class RetrievalService:
    """
    Service class for retrieving relevant content based on queries
    """

    def __init__(self, embedding_service, vector_store_service, database_service):
        """
        Initialize the retrieval service with required dependencies

        Args:
            embedding_service: Cohere embedding service instance
            vector_store_service: Qdrant vector store service instance
            database_service: Neon database service instance
        """
        self.embedding_service = embedding_service
        self.vector_store_service = vector_store_service
        self.database_service = database_service
        logger.info("✅ Retrieval Service initialized")

    async def retrieve_content(self, request: RetrievalRequest) -> RetrievalResponse:
        """
        Retrieve relevant content based on the query and mode

        Args:
            request: Retrieval request with query and parameters

        Returns:
            Retrieval response with results
        """
        try:
            if request.mode == "selected_text":
                # Mode B: Selected-text question → restricted retrieval
                results = await self._retrieve_with_selected_text(request)
            else:
                # Mode A: Standard question → similarity search
                results = await self._retrieve_standard(request)

            logger.info(f"Retrieved {len(results)} results for query: {request.query[:50]}...")
            return RetrievalResponse(
                results=results,
                query=request.query,
                retrieved_count=len(results),
                mode=request.mode
            )

        except Exception as e:
            logger.error(f"Error in content retrieval: {str(e)}")
            raise

    async def _retrieve_standard(self, request: RetrievalRequest) -> List[RetrievedChunk]:
        """
        Standard retrieval mode: search for similar content to the query

        Args:
            request: Retrieval request

        Returns:
            List of retrieved chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_service.embed_text(request.query)

            # Search in vector store
            search_results = await self.vector_store_service.search_similar(
                query_vector=query_embedding,
                top_k=request.top_k,
                filters=request.filters
            )

            # Convert to RetrievedChunk format
            retrieved_chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result["metadata"]
                )
                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in standard retrieval: {str(e)}")
            raise

    async def _retrieve_with_selected_text(self, request: RetrievalRequest) -> List[RetrievedChunk]:
        """
        Selected-text retrieval mode: combine selected text with query for targeted search

        Args:
            request: Retrieval request with selected_text

        Returns:
            List of retrieved chunks
        """
        try:
            if not request.selected_text:
                # If no selected text provided, fall back to standard retrieval
                return await self._retrieve_standard(request)

            # Combine selected text with query for more targeted search
            combined_query = f"{request.selected_text} {request.query}"

            # Generate embedding for the combined query
            query_embedding = await self.embedding_service.embed_text(combined_query)

            # Apply additional filters based on selected text context if available
            filters = request.filters.copy()
            if "chapter" in request.filters or "section" in request.filters:
                # If specific filters are provided, use them
                search_filters = filters
            else:
                # Otherwise, try to infer filters from the selected text if possible
                search_filters = filters

            # Search in vector store
            search_results = await self.vector_store_service.search_similar(
                query_vector=query_embedding,
                top_k=request.top_k,
                filters=search_filters
            )

            # Convert to RetrievedChunk format
            retrieved_chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result["metadata"]
                )
                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in selected-text retrieval: {str(e)}")
            # Fall back to standard retrieval if selected-text mode fails
            return await self._retrieve_standard(request)

    async def retrieve_by_document_id(self, doc_id: str, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve content specifically from a document based on a query

        Args:
            doc_id: Document ID to restrict search to
            query: Query string
            top_k: Number of results to return

        Returns:
            List of retrieved chunks from the specific document
        """
        try:
            # Get document chunks from database
            chunks = await self.database_service.get_chunks_by_document_id(doc_id)

            if not chunks:
                logger.warning(f"No chunks found for document ID: {doc_id}")
                return []

            # Generate embedding for the query
            query_embedding = await self.embedding_service.embed_text(query)

            # Create a temporary vector store search with document-specific content
            # For this implementation, we'll do a simple similarity search within the document chunks
            # In a production system, you might want to have document-specific vector collections

            # Get all content embeddings from the document chunks
            chunk_contents = [chunk.content for chunk in chunks]
            chunk_embeddings = await self.embedding_service.embed_texts(chunk_contents)

            # Calculate similarity scores manually
            import numpy as np
            query_array = np.array(query_embedding)
            similarities = []

            for i, chunk_embedding in enumerate(chunk_embeddings):
                chunk_array = np.array(chunk_embedding)
                # Calculate cosine similarity
                similarity = np.dot(query_array, chunk_array) / (
                    np.linalg.norm(query_array) * np.linalg.norm(chunk_array)
                )
                similarities.append((i, similarity))

            # Sort by similarity score
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get top-k results
            top_indices = similarities[:top_k]

            retrieved_chunks = []
            for idx, score in top_indices:
                chunk = chunks[idx]
                retrieved_chunk = RetrievedChunk(
                    id=chunk.id,
                    content=chunk.content,
                    score=score,
                    metadata=chunk.metadata
                )
                retrieved_chunks.append(retrieved_chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} results from document {doc_id}")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in document-specific retrieval: {str(e)}")
            raise

    async def retrieve_by_filters(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Retrieve content based on specific filters

        Args:
            query: Query string
            filters: Dictionary of filters to apply
            top_k: Number of results to return

        Returns:
            List of retrieved chunks
        """
        try:
            # Generate embedding for the query
            query_embedding = await self.embedding_service.embed_text(query)

            # Search in vector store with filters
            search_results = await self.vector_store_service.search_similar(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters
            )

            # Convert to RetrievedChunk format
            retrieved_chunks = []
            for result in search_results:
                chunk = RetrievedChunk(
                    id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result["metadata"]
                )
                retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error in filtered retrieval: {str(e)}")
            raise

    async def get_relevant_context(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Get a concatenated string of relevant context for answer generation

        Args:
            query: Query string
            top_k: Number of results to retrieve
            filters: Optional filters to apply

        Returns:
            Concatenated context string
        """
        try:
            request = RetrievalRequest(
                query=query,
                top_k=top_k,
                filters=filters or {}
            )

            response = await self.retrieve_content(request)

            # Concatenate all retrieved content
            context_parts = []
            for result in response.results:
                context_parts.append(result.content)

            context = "\n\n".join(context_parts)
            logger.info(f"Generated context with {len(context_parts)} parts for query: {query[:30]}...")

            return context

        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            raise