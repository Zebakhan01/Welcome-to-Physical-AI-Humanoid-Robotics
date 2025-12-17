from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class MessageBase(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime

class MessageCreate(MessageBase):
    pass

class Message(MessageBase):
    id: str

    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    title: str
    user_id: str

class ConversationCreate(ConversationBase):
    pass

class ConversationUpdate(BaseModel):
    title: Optional[str] = None

class Conversation(ConversationBase):
    id: str
    created_at: datetime
    updated_at: datetime
    messages: List[Message] = []

    class Config:
        from_attributes = True