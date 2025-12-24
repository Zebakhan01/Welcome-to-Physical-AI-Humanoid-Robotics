"""
RAG API Endpoints for Phase 2
REST endpoints for the complete RAG pipeline with clear service separation
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from ...services.rag.rag_service import RAGService, RAGPipelineRequest, RAGPipelineResponse
from ...utils.logger import logger


router = APIRouter(prefix="/rag", tags=["rag"])


class HealthResponse(BaseModel):
    """Response for health check"""
    status: str
    services: Dict[str, bool]
    timestamp: float


class ProcessContentRequest(BaseModel):
    """Request to process content"""
    source_path: str
    chunk_size: int = 1000
    overlap_size: int = 100


class ProcessContentResponse(BaseModel):
    """Response for content processing"""
    success: bool
    processed_files: int
    processed_chunks: int
    processing_time: float


class QueryRequest(BaseModel):
    """Request for RAG query"""
    query: str
    mode: str = "standard"
    selected_text: Optional[str] = None
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response for RAG query"""
    answer: str
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    sources: List[str]
    grounded: bool
    processing_time: float


class IndexContentRequest(BaseModel):
    """Request to index content"""
    source_path: str


class IndexContentResponse(BaseModel):
    """Response for content indexing"""
    success: bool
    indexed_chunks: int
    processing_time: float


# Initialize the main RAG service
rag_service = RAGService()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint for RAG services
    """
    try:
        start_time = time.time()
        health_status = await rag_service.health_check()
        processing_time = time.time() - start_time

        status = "healthy" if health_status.get("overall", False) else "unhealthy"

        logger.info(f"Health check completed: {status}")
        return HealthResponse(
            status=status,
            services=health_status,
            timestamp=processing_time
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/process-content", response_model=ProcessContentResponse)
async def process_content(request: ProcessContentRequest):
    """
    Process content from source path: load, embed, store, and index
    """
    try:
        start_time = time.time()

        success = await rag_service.process_content(
            source_path=request.source_path,
            chunk_size=request.chunk_size,
            overlap_size=request.overlap_size
        )

        processing_time = time.time() - start_time

        if not success:
            raise HTTPException(status_code=400, detail="Content processing failed")

        # For now, we'll return a simple success response
        # In a real implementation, you might want to return more detailed stats
        logger.info(f"Content processed successfully from: {request.source_path}")
        return ProcessContentResponse(
            success=True,
            processed_files=1,  # Simplified - in reality, this would be calculated
            processed_chunks=0,  # Simplified - in reality, this would be calculated
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content processing failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question
    """
    try:
        start_time = time.time()

        rag_request = RAGPipelineRequest(
            source_path="",  # Not needed for querying
            query=request.query,
            mode=request.mode,
            selected_text=request.selected_text,
            top_k=request.top_k
        )

        response = await rag_service.query(rag_request)

        processing_time = time.time() - start_time

        logger.info(f"Query processed: {request.query[:50]}...")
        return QueryResponse(
            answer=response.answer,
            question=response.question,
            retrieved_chunks=response.retrieved_chunks,
            sources=response.sources,
            grounded=response.grounded,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/index-content", response_model=IndexContentResponse)
async def index_content(request: IndexContentRequest):
    """
    Index content from source path into vector store and database
    """
    try:
        start_time = time.time()

        success = await rag_service.load_and_index_content(request.source_path)

        processing_time = time.time() - start_time

        if not success:
            raise HTTPException(status_code=400, detail="Content indexing failed")

        # In a real implementation, you would return the actual number of indexed chunks
        # For now, we'll use a placeholder
        logger.info(f"Content indexed successfully from: {request.source_path}")
        return IndexContentResponse(
            success=True,
            indexed_chunks=0,  # Placeholder - would be actual count in real implementation
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Error indexing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content indexing failed: {str(e)}")


@router.get("/collections/stats")
async def get_collection_stats():
    """
    Get statistics about the vector store collections
    """
    try:
        stats = await rag_service.vector_store.get_collection_stats()
        logger.info("Collection stats retrieved")
        return stats
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Collection stats retrieval failed: {str(e)}")


class ContentStructureRequest(BaseModel):
    """Request for content structure"""
    source_path: str
    recursive: bool = True


@router.post("/content-structure")
async def get_content_structure(request: ContentStructureRequest):
    """
    Get the structure of content at the specified path
    """
    try:
        structure = await rag_service.content_loader.get_content_structure(
            request.source_path,
            request.recursive
        )
        logger.info(f"Content structure retrieved from: {request.source_path}")
        return structure
    except Exception as e:
        logger.error(f"Error getting content structure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content structure retrieval failed: {str(e)}")