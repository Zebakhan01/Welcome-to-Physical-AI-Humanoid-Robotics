from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ContentChunkBase(BaseModel):
    title: str
    content: str
    chapter_id: str
    metadata: Optional[Dict[str, Any]] = None
    embedding_vector: Optional[list] = None

class ContentChunkCreate(ContentChunkBase):
    pass

class ContentChunkUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ContentChunk(ContentChunkBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True