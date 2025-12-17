from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

class UserBase(BaseModel):
    email: str
    name: Optional[str] = None
    preferred_language: str = "en"

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    preferred_language: Optional[str] = None

class User(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True