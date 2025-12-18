from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from ...utils.logger import logger
from ...utils.validators import ChatRequest  # Assuming ChatRequest is defined in validators
from ..rag.retrieval_service import query_content, QueryRequest
from ..rag.embedding_service import generate_embeddings
from ..rag.llm_service import generate_answer_with_context, generate_simple_answer_with_context
from sentence_transformers import SentenceTransformer
import asyncio
from ...database import get_db
from ...services.database_service import (
    create_conversation, get_conversation_by_id, get_user_conversations,
    create_message, get_conversation_messages, update_conversation_title
)
from sqlalchemy.orm import Session

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    timestamp: datetime

@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Process a chat message using RAG to retrieve relevant textbook content and generate response
    """
    try:
        # Use the provided conversation_id or create a new one
        conversation_id = request.conversation_id
        if not conversation_id:
            # Create a new conversation if none provided
            conversation = create_conversation(
                db=db,
                user_id=request.user_id or str(uuid.uuid4()),  # Use a default user ID if not provided
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message
            )
            conversation_id = conversation.id
        else:
            # Validate that the conversation exists
            conversation = get_conversation_by_id(db, uuid.UUID(conversation_id))
            if not conversation:
                # Create a new conversation if the provided ID doesn't exist
                conversation = create_conversation(
                    db=db,
                    user_id=request.user_id or str(uuid.uuid4()),
                    title=request.message[:50] + "..." if len(request.message) > 50 else request.message
                )
                conversation_id = conversation.id

        logger.info(f"Chat message received for conversation {conversation_id}: {request.message}")

        # Store the user's message in the database
        user_message = create_message(
            db=db,
            conversation_id=uuid.UUID(conversation_id),
            role="user",
            content=request.message
        )

        # Retrieve relevant content from the textbook using RAG
        query_request = QueryRequest(query=request.message, top_k=3)
        retrieval_result = await query_content(query_request)

        # Prepare context for response generation
        context_items = []
        sources = []

        for result in retrieval_result.results:
            content = result.get("content", "")
            title = result.get("title", "Unknown")
            score = result.get("score", 0)

            if content.strip():  # Only add non-empty content
                context_items.append({
                    "content": content,
                    "title": title,
                    "score": score
                })
                sources.append({
                    "title": title,
                    "score": score,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

        # Generate response based on retrieved content
        if context_items:
            # Use LLM service to generate response with context
            try:
                response_obj = await generate_answer_with_context(
                    query=request.message,
                    context=context_items
                )
                response_text = response_obj.answer
                # Update sources with the ones from the LLM response
                sources = response_obj.sources
            except Exception as e:
                logger.warning(f"LLM generation failed, using fallback: {e}")
                # Use fallback implementation if LLM is not available
                response_obj = generate_simple_answer_with_context(
                    query=request.message,
                    context=context_items
                )
                response_text = response_obj.answer
                sources = response_obj.sources
        else:
            # If no relevant content found, provide a helpful response
            response_text = f"I couldn't find specific information about '{request.message}' in the textbook. Please try rephrasing your question or check other sections of the Physical AI & Humanoid Robotics textbook."
            sources = []

        # Store the assistant's response in the database
        assistant_message = create_message(
            db=db,
            conversation_id=uuid.UUID(conversation_id),
            role="assistant",
            content=response_text,
            sources=sources
        )

        return ChatResponse(
            response=response_text,
            conversation_id=str(conversation_id),
            sources=sources,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in chat message processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

class ConversationHistoryRequest(BaseModel):
    conversation_id: str
    limit: int = 10

class ConversationHistoryResponse(BaseModel):
    messages: List[Message]
    conversation_id: str

@router.post("/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    request: ConversationHistoryRequest,
    db: Session = Depends(get_db)
):
    """
    Retrieve conversation history from the database
    """
    try:
        logger.info(f"Conversation history requested for: {request.conversation_id}")

        # Retrieve messages from the database
        conversation_uuid = uuid.UUID(request.conversation_id)
        db_messages = get_conversation_messages(
            db=db,
            conversation_id=conversation_uuid,
            limit=request.limit
        )

        # Convert database messages to the expected format
        history_messages = []
        for db_msg in db_messages:
            history_messages.append({
                "role": db_msg.role,
                "content": db_msg.content,
                "timestamp": db_msg.timestamp
            })

        # If no messages found, return a welcome message
        if not history_messages:
            history_messages = [
                {
                    "role": "assistant",
                    "content": "Welcome to the Physical AI & Humanoid Robotics textbook chat! I can answer questions based on the textbook content. What would you like to know?",
                    "timestamp": datetime.now()
                }
            ]

        return ConversationHistoryResponse(
            messages=history_messages,
            conversation_id=request.conversation_id
        )
    except Exception as e:
        logger.error(f"Error in conversation history retrieval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation history retrieval failed: {str(e)}")

@router.post("/reset")
async def reset_conversation(
    db: Session = Depends(get_db)
):
    """
    Reset conversation context by creating a new conversation
    """
    try:
        new_conversation_id = str(uuid.uuid4())

        logger.info(f"Conversation reset requested, new ID: {new_conversation_id}")

        return {
            "conversation_id": new_conversation_id,
            "message": "Conversation has been reset. Starting a new session."
        }
    except Exception as e:
        logger.error(f"Error in conversation reset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation reset failed: {str(e)}")