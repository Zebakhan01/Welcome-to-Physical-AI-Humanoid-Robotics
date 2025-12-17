from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from ...utils.logger import logger

router = APIRouter()

class MessageProcessingRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class MessageProcessingResponse(BaseModel):
    processed_message: str
    is_valid: bool
    categories: List[str]
    warnings: List[str]
    processed_at: datetime

class MessageValidationRequest(BaseModel):
    message: str
    max_length: int = 10000
    allowed_categories: List[str] = ["academic", "technical", "general"]

@router.post("/process", response_model=MessageProcessingResponse)
async def process_message(request: MessageProcessingRequest):
    """
    Process a user message for the chatbot
    """
    try:
        processed_message = request.message.strip()
        is_valid = True
        categories = []
        warnings = []

        # Validate message length
        if len(processed_message) > 10000:
            warnings.append("Message is very long and may be truncated")
            processed_message = processed_message[:10000]

        # Check for empty message
        if not processed_message:
            is_valid = False
            warnings.append("Message is empty")

        # Categorize the message
        categories = categorize_message(processed_message)

        # Check for potentially inappropriate content
        if contains_inappropriate_content(processed_message):
            is_valid = False
            warnings.append("Message contains potentially inappropriate content")

        # Check if message is related to the textbook topic
        if not is_textbook_related(processed_message):
            warnings.append("Message may not be related to Physical AI & Humanoid Robotics")

        logger.info(f"Processed message for user {request.user_id}")

        return MessageProcessingResponse(
            processed_message=processed_message,
            is_valid=is_valid,
            categories=categories,
            warnings=warnings,
            processed_at=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

def categorize_message(message: str) -> List[str]:
    """
    Categorize the message based on content
    """
    categories = []
    message_lower = message.lower()

    # Academic/educational content
    academic_keywords = [
        "learn", "study", "course", "chapter", "topic", "concept", "understand",
        "explain", "teach", "education", "student", "professor", "class"
    ]
    if any(keyword in message_lower for keyword in academic_keywords):
        categories.append("academic")

    # Technical content
    technical_keywords = [
        "robot", "ai", "physical ai", "humanoid", "ros", "sensor", "actuator",
        "kinematics", "dynamics", "control", "algorithm", "code", "simulation"
    ]
    if any(keyword in message_lower for keyword in technical_keywords):
        categories.append("technical")

    # General content
    general_keywords = [
        "hello", "hi", "help", "question", "what", "how", "why", "when", "where"
    ]
    if any(keyword in message_lower for keyword in general_keywords) and not categories:
        categories.append("general")

    # Specific textbook content
    textbook_keywords = [
        "textbook", "book", "week", "chapter", "module", "glossary", "appendix"
    ]
    if any(keyword in message_lower for keyword in textbook_keywords):
        categories.append("textbook")

    return categories if categories else ["general"]

def contains_inappropriate_content(message: str) -> bool:
    """
    Check if message contains inappropriate content
    """
    # This is a basic implementation - in production, use a more sophisticated approach
    inappropriate_patterns = [
        r'\b(hate|kill|destroy|attack|violence)\b',
        r'\b(cursing|profanity|offensive)\b',
        # Add more patterns as needed
    ]

    message_lower = message.lower()
    for pattern in inappropriate_patterns:
        if re.search(pattern, message_lower):
            return True

    return False

def is_textbook_related(message: str) -> bool:
    """
    Check if message is related to the textbook topic
    """
    textbook_related_keywords = [
        "physical ai", "humanoid", "robotics", "robot", "ai", "textbook",
        "course", "learn", "study", "week", "chapter", "module"
    ]

    message_lower = message.lower()
    return any(keyword in message_lower for keyword in textbook_related_keywords)

class SanitizeRequest(BaseModel):
    text: str
    remove_special_chars: bool = False
    max_length: Optional[int] = None

class SanitizeResponse(BaseModel):
    sanitized_text: str
    original_length: int
    sanitized_length: int

@router.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_text(request: SanitizeRequest):
    """
    Sanitize text input for security
    """
    try:
        sanitized = request.text

        # Remove or escape potentially dangerous characters
        if request.remove_special_chars:
            # Remove special characters but keep alphanumeric, spaces, and basic punctuation
            sanitized = re.sub(r'[^\w\s\-\.,!?;:\'\"()]+', '', sanitized)

        # Truncate if max_length is specified
        if request.max_length and len(sanitized) > request.max_length:
            sanitized = sanitized[:request.max_length]

        logger.info(f"Sanitized text of length {len(request.text)} to {len(sanitized)}")

        return SanitizeResponse(
            sanitized_text=sanitized,
            original_length=len(request.text),
            sanitized_length=len(sanitized)
        )

    except Exception as e:
        logger.error(f"Error sanitizing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sanitizing text: {str(e)}")

class ExtractEntitiesRequest(BaseModel):
    text: str
    entity_types: List[str] = ["chapter", "week", "topic", "concept"]

class ExtractEntitiesResponse(BaseModel):
    entities: Dict[str, List[str]]
    extracted_at: datetime

@router.post("/extract-entities", response_model=ExtractEntitiesResponse)
async def extract_entities(request: ExtractEntitiesRequest):
    """
    Extract entities from text
    """
    try:
        entities = {}

        if "chapter" in request.entity_types:
            # Extract chapter references like "Chapter 1", "chapter 1", "Week 1", etc.
            chapter_pattern = r'\b(chapter|week)\s+(\d+)\b'
            chapters = re.findall(chapter_pattern, request.text, re.IGNORECASE)
            entities["chapters"] = [f"{item[0]} {item[1]}" for item in chapters]

        if "topic" in request.entity_types:
            # Extract common robotics/AI topics
            topic_pattern = r'\b(physical ai|humanoid|robot|ros|sensor|actuator|kinematics|dynamics|control|algorithm|simulation|gazebo|unity|isaac|vla|vision-language-action)\b'
            topics = re.findall(topic_pattern, request.text, re.IGNORECASE)
            entities["topics"] = list(set(topics))  # Remove duplicates

        logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities from text")

        return ExtractEntitiesResponse(
            entities=entities,
            extracted_at=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")