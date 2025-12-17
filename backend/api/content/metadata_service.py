from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from ...utils.logger import logger

router = APIRouter()

class ChapterMetadata(BaseModel):
    id: str
    title: str
    week_number: Optional[int] = None
    module: Optional[str] = None
    learning_objectives: List[str]
    prerequisites: List[str]
    estimated_reading_time: int  # in minutes
    created_at: datetime
    updated_at: datetime
    tags: List[str]

class ChapterMetadataCreate(BaseModel):
    title: str
    week_number: Optional[int] = None
    module: Optional[str] = None
    learning_objectives: List[str] = []
    prerequisites: List[str] = []
    estimated_reading_time: int = 10  # default 10 minutes
    tags: List[str] = []

class ChapterMetadataUpdate(BaseModel):
    title: Optional[str] = None
    week_number: Optional[int] = None
    module: Optional[str] = None
    learning_objectives: Optional[List[str]] = None
    prerequisites: Optional[List[str]] = None
    estimated_reading_time: Optional[int] = None
    tags: Optional[List[str]] = None

# In-memory storage for metadata (in production, this would use a database)
chapter_metadata_store: Dict[str, ChapterMetadata] = {}

@router.post("/chapters", response_model=ChapterMetadata)
async def create_chapter_metadata(metadata: ChapterMetadataCreate):
    """
    Create metadata for a textbook chapter
    """
    try:
        chapter_id = str(uuid.uuid4())

        chapter_metadata = ChapterMetadata(
            id=chapter_id,
            title=metadata.title,
            week_number=metadata.week_number,
            module=metadata.module,
            learning_objectives=metadata.learning_objectives,
            prerequisites=metadata.prerequisites,
            estimated_reading_time=metadata.estimated_reading_time,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=metadata.tags
        )

        chapter_metadata_store[chapter_id] = chapter_metadata

        logger.info(f"Created metadata for chapter '{metadata.title}' with ID {chapter_id}")

        return chapter_metadata

    except Exception as e:
        logger.error(f"Error creating chapter metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating chapter metadata: {str(e)}")

@router.get("/chapters/{chapter_id}", response_model=ChapterMetadata)
async def get_chapter_metadata(chapter_id: str):
    """
    Get metadata for a specific textbook chapter
    """
    try:
        if chapter_id not in chapter_metadata_store:
            raise HTTPException(status_code=404, detail=f"Chapter with ID {chapter_id} not found")

        logger.info(f"Retrieved metadata for chapter {chapter_id}")

        return chapter_metadata_store[chapter_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chapter metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chapter metadata: {str(e)}")

@router.put("/chapters/{chapter_id}", response_model=ChapterMetadata)
async def update_chapter_metadata(chapter_id: str, metadata: ChapterMetadataUpdate):
    """
    Update metadata for a textbook chapter
    """
    try:
        if chapter_id not in chapter_metadata_store:
            raise HTTPException(status_code=404, detail=f"Chapter with ID {chapter_id} not found")

        existing_metadata = chapter_metadata_store[chapter_id]

        # Update fields that are provided
        if metadata.title is not None:
            existing_metadata.title = metadata.title
        if metadata.week_number is not None:
            existing_metadata.week_number = metadata.week_number
        if metadata.module is not None:
            existing_metadata.module = metadata.module
        if metadata.learning_objectives is not None:
            existing_metadata.learning_objectives = metadata.learning_objectives
        if metadata.prerequisites is not None:
            existing_metadata.prerequisites = metadata.prerequisites
        if metadata.estimated_reading_time is not None:
            existing_metadata.estimated_reading_time = metadata.estimated_reading_time
        if metadata.tags is not None:
            existing_metadata.tags = metadata.tags

        existing_metadata.updated_at = datetime.now()

        logger.info(f"Updated metadata for chapter {chapter_id}")

        return existing_metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chapter metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating chapter metadata: {str(e)}")

@router.delete("/chapters/{chapter_id}")
async def delete_chapter_metadata(chapter_id: str):
    """
    Delete metadata for a textbook chapter
    """
    try:
        if chapter_id not in chapter_metadata_store:
            raise HTTPException(status_code=404, detail=f"Chapter with ID {chapter_id} not found")

        del chapter_metadata_store[chapter_id]

        logger.info(f"Deleted metadata for chapter {chapter_id}")

        return {"message": f"Chapter metadata {chapter_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chapter metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting chapter metadata: {str(e)}")

class GetAllChaptersResponse(BaseModel):
    chapters: List[ChapterMetadata]
    total_count: int

@router.get("/chapters", response_model=GetAllChaptersResponse)
async def get_all_chapters(
    week_number: Optional[int] = None,
    module: Optional[str] = None,
    tag: Optional[str] = None
):
    """
    Get all chapters with optional filtering
    """
    try:
        chapters = list(chapter_metadata_store.values())

        # Apply filters if provided
        if week_number is not None:
            chapters = [c for c in chapters if c.week_number == week_number]

        if module is not None:
            chapters = [c for c in chapters if c.module == module]

        if tag is not None:
            chapters = [c for c in chapters if tag in c.tags]

        logger.info(f"Retrieved {len(chapters)} chapters")

        return GetAllChaptersResponse(
            chapters=chapters,
            total_count=len(chapters)
        )

    except Exception as e:
        logger.error(f"Error retrieving chapters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chapters: {str(e)}")

class ContentChunkMetadata(BaseModel):
    id: str
    chapter_id: str
    title: str
    content_summary: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime

class ContentChunkMetadataCreate(BaseModel):
    chapter_id: str
    title: str
    content_summary: str
    tags: List[str] = []

@router.post("/chunks", response_model=ContentChunkMetadata)
async def create_content_chunk_metadata(metadata: ContentChunkMetadataCreate):
    """
    Create metadata for a content chunk (for RAG system)
    """
    try:
        # Verify that the chapter exists
        if metadata.chapter_id not in chapter_metadata_store:
            raise HTTPException(status_code=404, detail=f"Chapter with ID {metadata.chapter_id} not found")

        chunk_id = str(uuid.uuid4())

        chunk_metadata = ContentChunkMetadata(
            id=chunk_id,
            chapter_id=metadata.chapter_id,
            title=metadata.title,
            content_summary=metadata.content_summary,
            tags=metadata.tags,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # In a full implementation, this would be stored in a database
        # For now, we'll just log the creation
        logger.info(f"Created content chunk metadata for '{metadata.title}' in chapter {metadata.chapter_id}")

        return chunk_metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating content chunk metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating content chunk metadata: {str(e)}")