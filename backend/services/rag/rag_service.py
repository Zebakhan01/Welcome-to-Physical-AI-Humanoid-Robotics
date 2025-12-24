"""
Main RAG Service for Phase 2
Orchestrates the complete RAG pipeline: loading, embedding, storing, retrieving, and answering
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .content_loader import ContentLoaderService, ContentLoaderConfig, ContentChunk
from .embedding_service import CohereEmbeddingService
from .vector_store import QdrantVectorStoreService
from .database_service import NeonDatabaseService, DocumentMetadata, ChunkMetadata
from .retrieval_service import RetrievalService, RetrievalRequest, RetrievedChunk
from .answer_generation_service import AnswerGenerationService, AnswerRequest
from ...utils.logger import logger


class RAGPipelineRequest(BaseModel):
    """Request for the complete RAG pipeline"""
    source_path: str
    query: str
    mode: str = "standard"  # "standard" or "selected_text"
    selected_text: Optional[str] = None
    top_k: int = 5
    chunk_size: int = 1000
    overlap_size: int = 100


class RAGPipelineResponse(BaseModel):
    """Response from the complete RAG pipeline"""
    answer: str
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    sources: List[str]
    grounded: bool
    processing_time: float


class RAGService:
    """
    Main RAG service that orchestrates the complete pipeline
    """

    def __init__(self):
        """Initialize the main RAG service with all required sub-services"""
        self.content_loader = ContentLoaderService()
        self.embedding_service = CohereEmbeddingService()
        self.vector_store = QdrantVectorStoreService()
        self.database_service = NeonDatabaseService()
        self.retrieval_service = RetrievalService(
            self.embedding_service,
            self.vector_store,
            self.database_service
        )
        self.answer_generation_service = AnswerGenerationService()

        logger.info("✅ Main RAG Service initialized with all sub-services")

    async def process_content(self, source_path: str, chunk_size: int = 1000, overlap_size: int = 100) -> bool:
        """
        Process content from source path: load, embed, store, and index

        Args:
            source_path: Path to the content to process
            chunk_size: Size of content chunks
            overlap_size: Overlap between chunks

        Returns:
            True if processing was successful
        """
        try:
            logger.info(f"Starting content processing for: {source_path}")

            # Load content
            config = ContentLoaderConfig(
                source_path=source_path,
                chunk_size=chunk_size,
                overlap_size=overlap_size
            )
            chunks = await self.content_loader.load_content(config)

            if not chunks:
                logger.warning(f"No content found at: {source_path}")
                return False

            logger.info(f"Loaded {len(chunks)} chunks from {source_path}")

            # Process each chunk: embed and store
            for chunk in chunks:
                # Generate embedding for the chunk content
                embedding = await self.embedding_service.embed_text(chunk.content)

                # Create vector record
                from .vector_store import VectorRecord
                vector_record = VectorRecord(
                    id=chunk.id,
                    vector=embedding,
                    content=chunk.content,
                    metadata={
                        "title": chunk.title,
                        "source_file": chunk.source_file,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        **chunk.metadata
                    }
                )

                # Store in vector database
                await self.vector_store.store_embedding(vector_record)

                # Store chunk metadata in PostgreSQL
                chunk_metadata = ChunkMetadata(
                    id=chunk.id,
                    document_id=chunk.source_file_path,  # Using file path as document ID
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    vector_id=chunk.id,  # Using the same ID for now
                    word_count=len(chunk.content.split()),
                    char_count=len(chunk.content),
                    metadata={
                        "title": chunk.title,
                        "source_file": chunk.source_file,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        **chunk.metadata
                    }
                )

                # Store document metadata if not already stored
                doc_metadata = DocumentMetadata(
                    id=chunk.source_file_path,
                    title=chunk.title,
                    source_file_path=chunk.source_file_path,
                    chapter=chunk.chapter,
                    section=chunk.section,
                    total_chunks=chunk.total_chunks,
                    word_count=len(chunk.content.split()),
                    char_count=len(chunk.content),
                    metadata={
                        "chapter": chunk.chapter,
                        "section": chunk.section
                    }
                )

                await self.database_service.store_document_metadata(doc_metadata)
                await self.database_service.store_chunk_metadata(chunk_metadata)

            logger.info(f"Completed content processing for: {source_path}")
            return True

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            raise

    async def query(self, request: RAGPipelineRequest) -> RAGPipelineResponse:
        """
        Execute the complete RAG pipeline: retrieve relevant content and generate answer

        Args:
            request: RAG pipeline request with query and parameters

        Returns:
            RAG pipeline response with answer and metadata
        """
        import time
        start_time = time.time()

        try:
            logger.info(f"Processing RAG query: {request.query[:50]}...")

            # Create retrieval request
            retrieval_request = RetrievalRequest(
                query=request.query,
                top_k=request.top_k,
                mode=request.mode,
                selected_text=request.selected_text
            )

            # Retrieve relevant content
            retrieval_response = await self.retrieval_service.retrieve_content(retrieval_request)

            # Get context from retrieved chunks
            context_parts = []
            retrieved_chunks_data = []

            for chunk in retrieval_response.results:
                context_parts.append(chunk.content)
                retrieved_chunks_data.append({
                    "id": chunk.id,
                    "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,  # Truncate for response
                    "score": chunk.score,
                    "metadata": chunk.metadata
                })

            context = "\n\n".join(context_parts)

            # Generate answer based on context
            answer_request = AnswerRequest(
                question=request.query,
                context=context
            )
            answer_response = await self.answer_generation_service.generate_answer(answer_request)

            processing_time = time.time() - start_time

            logger.info(f"RAG query completed in {processing_time:.2f}s")
            return RAGPipelineResponse(
                answer=answer_response.answer,
                question=request.query,
                retrieved_chunks=retrieved_chunks_data,
                sources=answer_response.sources,
                grounded=answer_response.grounded,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            processing_time = time.time() - start_time

            # Return fallback response
            return RAGPipelineResponse(
                answer="This information is not available in the textbook.",
                question=request.query,
                retrieved_chunks=[],
                sources=[],
                grounded=False,
                processing_time=processing_time
            )

    async def load_and_index_content(self, source_path: str) -> bool:
        """
        Load content from source and index it in both vector store and database

        Args:
            source_path: Path to the content to load and index

        Returns:
            True if loading and indexing was successful
        """
        try:
            logger.info(f"Loading and indexing content from: {source_path}")

            # Load content
            config = ContentLoaderConfig(source_path=source_path)
            chunks = await self.content_loader.load_content(config)

            if not chunks:
                logger.warning(f"No content found at: {source_path}")
                return False

            # Prepare for embedding and storage
            chunk_embeddings = []
            for chunk in chunks:
                # Generate embedding
                embedding = await self.embedding_service.embed_text(chunk.content)

                # Create vector record
                from .vector_store import VectorRecord
                vector_record = VectorRecord(
                    id=chunk.id,
                    vector=embedding,
                    content=chunk.content,
                    metadata={
                        "title": chunk.title,
                        "source_file": chunk.source_file,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        **chunk.metadata
                    }
                )

                chunk_embeddings.append((chunk, vector_record))

            # Store all embeddings in vector store
            vector_records = [record for _, record in chunk_embeddings]
            await self.vector_store.store_embeddings(vector_records)

            # Store metadata in database
            for chunk, _ in chunk_embeddings:
                # Store document metadata
                doc_metadata = DocumentMetadata(
                    id=chunk.source_file_path,
                    title=chunk.title,
                    source_file_path=chunk.source_file_path,
                    chapter=chunk.chapter,
                    section=chunk.section,
                    total_chunks=chunk.total_chunks,
                    word_count=len(chunk.content.split()),
                    char_count=len(chunk.content),
                    metadata={
                        "chapter": chunk.chapter,
                        "section": chunk.section
                    }
                )

                chunk_metadata = ChunkMetadata(
                    id=chunk.id,
                    document_id=chunk.source_file_path,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    vector_id=chunk.id,
                    word_count=len(chunk.content.split()),
                    char_count=len(chunk.content),
                    metadata={
                        "title": chunk.title,
                        "source_file": chunk.source_file,
                        "chapter": chunk.chapter,
                        "section": chunk.section,
                        **chunk.metadata
                    }
                )

                await self.database_service.store_document_metadata(doc_metadata)
                await self.database_service.store_chunk_metadata(chunk_metadata)

            logger.info(f"Successfully loaded and indexed {len(chunks)} chunks from: {source_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading and indexing content: {str(e)}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all services

        Returns:
            Dictionary with health status of all services
        """
        try:
            health_status = {
                "content_loader": True,
                "embedding_service": True,
                "vector_store": True,
                "database_service": True,
                "retrieval_service": True,
                "answer_generation_service": True,
                "overall": True
            }

            # Test each service
            try:
                # Test embedding service
                test_embedding = await self.embedding_service.embed_text("test")
                health_status["embedding_service"] = len(test_embedding) > 0
            except:
                health_status["embedding_service"] = False
                health_status["overall"] = False

            try:
                # Test vector store
                stats = await self.vector_store.get_collection_stats()
                health_status["vector_store"] = "collection_name" in stats
            except:
                health_status["vector_store"] = False
                health_status["overall"] = False

            try:
                # Test database
                docs = await self.database_service.get_all_documents()
                health_status["database_service"] = True  # If we can query, it's working
            except:
                health_status["database_service"] = False
                health_status["overall"] = False

            return health_status

        except Exception as e:
            logger.error(f"Error in health check: {str(e)}")
            return {
                "content_loader": False,
                "embedding_service": False,
                "vector_store": False,
                "database_service": False,
                "retrieval_service": False,
                "answer_generation_service": False,
                "overall": False,
                "error": str(e)
            }