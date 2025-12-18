from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import List, Optional
from uuid import UUID
import uuid
from ..models.sqlalchemy_models import User, Conversation, Message
from ..models.user import UserCreate, UserUpdate
from ..models.conversation import ConversationCreate, ConversationUpdate, MessageCreate


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email"""
    return db.query(User).filter(User.email == email).first()


def create_user(db: Session, user_data: UserCreate) -> User:
    """Create a new user"""
    db_user = User(
        email=user_data.email,
        name=user_data.name,
        preferred_language=user_data.preferred_language
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: UUID, user_update: UserUpdate) -> Optional[User]:
    """Update user information"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        if user_update.name is not None:
            db_user.name = user_update.name
        if user_update.preferred_language is not None:
            db_user.preferred_language = user_update.preferred_language
        db.commit()
        db.refresh(db_user)
    return db_user


def create_conversation(db: Session, user_id: UUID, title: str) -> Conversation:
    """Create a new conversation"""
    db_conversation = Conversation(
        user_id=user_id,
        title=title
    )
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation


def get_conversation_by_id(db: Session, conversation_id: UUID) -> Optional[Conversation]:
    """Get a conversation by ID"""
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def get_user_conversations(db: Session, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Conversation]:
    """Get all conversations for a user"""
    return db.query(Conversation).filter(Conversation.user_id == user_id).order_by(desc(Conversation.created_at)).offset(offset).limit(limit).all()


def create_message(db: Session, conversation_id: UUID, role: str, content: str, sources: Optional[list] = None) -> Message:
    """Create a new message in a conversation"""
    db_message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        sources=sources
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_conversation_messages(db: Session, conversation_id: UUID, limit: int = 100, offset: int = 0) -> List[Message]:
    """Get all messages in a conversation"""
    return db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp).offset(offset).limit(limit).all()


def update_conversation_title(db: Session, conversation_id: UUID, title: str) -> Optional[Conversation]:
    """Update conversation title"""
    db_conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if db_conversation:
        db_conversation.title = title
        db.commit()
        db.refresh(db_conversation)
    return db_conversation


def delete_conversation(db: Session, conversation_id: UUID) -> bool:
    """Delete a conversation and all its messages"""
    # First delete all messages in the conversation
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    # Then delete the conversation
    result = db.query(Conversation).filter(Conversation.id == conversation_id).delete()
    db.commit()
    return result > 0