from typing import Any, Dict, List
import re
from pydantic import BaseModel, validator

class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None
    user_id: str = None

    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 10000:  # Limit message length
            raise ValueError('Message too long')
        return v

class ContentIndexRequest(BaseModel):
    content: str
    title: str
    chapter_id: str
    metadata: Dict[str, Any] = {}

    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v

    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Title cannot be empty')
        return v

class UserLoginRequest(BaseModel):
    email: str
    password: str

    @validator('email')
    def validate_email(cls, v):
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v.lower()

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_textbook_content(content: str) -> bool:
    """Validate that content follows textbook standards"""
    if not content or len(content.strip()) == 0:
        return False

    # Check for basic structure (minimum requirements)
    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    if len(non_empty_lines) < 3:  # Minimum content requirement
        return False

    return True