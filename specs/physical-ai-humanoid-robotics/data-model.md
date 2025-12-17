# Data Model: Physical AI & Humanoid Robotics Textbook

## User Model
**Purpose**: Store user information and preferences for personalized learning experience

```python
class User:
    id: UUID
    email: str
    name: str
    created_at: datetime
    preferences: dict  # Language, personalization settings
    learning_progress: dict  # Track completed chapters/modules
    background: str  # Software/Hardware background for personalization
```

## Chapter Model
**Purpose**: Represent textbook chapters with metadata and learning objectives

```python
class Chapter:
    id: UUID
    title: str
    slug: str  # URL-friendly identifier
    content_path: str  # File path to markdown content
    category: str  # intro, weeks, modules, capstone, hardware, appendix
    week_number: Optional[int]  # For week-based chapters
    module_name: Optional[str]  # For module-based chapters
    learning_objectives: List[str]
    prerequisites: List[str]
    estimated_reading_time: int  # in minutes
    created_at: datetime
    updated_at: datetime
```

## ContentChunk Model
**Purpose**: Store segmented textbook content for RAG system

```python
class ContentChunk:
    id: UUID
    chapter_id: UUID  # Reference to parent chapter
    content: str  # The actual text content
    embedding: List[float]  # Vector embedding for similarity search
    metadata: dict  # Additional info like section title, page numbers
    chunk_order: int  # Order within the chapter
    token_count: int  # For tracking content length
    created_at: datetime
```

## Conversation Model
**Purpose**: Store chat history and conversation context

```python
class Conversation:
    id: UUID
    user_id: UUID
    title: str  # Generated from first question
    created_at: datetime
    updated_at: datetime
    messages: List[Message]  # List of messages in the conversation
```

## Message Model
**Purpose**: Individual messages within a conversation

```python
class Message:
    id: UUID
    conversation_id: UUID
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    context_chunks: List[UUID]  # References to content chunks used
    sources: List[dict]  # Chapter/section references for fact-checking
```

## ContentReference Model
**Purpose**: Link content chunks to their original source locations

```python
class ContentReference:
    id: UUID
    chapter_id: UUID
    chunk_id: UUID
    section_title: str
    page_reference: Optional[str]
    url_path: str  # Relative path for linking back to content
    created_at: datetime
```

## PersonalizationProfile Model
**Purpose**: Store user background and preferences for content adaptation

```python
class PersonalizationProfile:
    id: UUID
    user_id: UUID
    background: str  # "software", "hardware", "mixed", "beginner"
    preferred_language: str  # Default "en", support for "ur" (Urdu)
    learning_pace: str  # "fast", "moderate", "slow"
    interests: List[str]  # Specific robotics areas of interest
    challenge_level: str  # "beginner", "intermediate", "advanced"
    created_at: datetime
    updated_at: datetime
```

## Translation Model
**Purpose**: Store translated content for multilingual support

```python
class Translation:
    id: UUID
    content_chunk_id: UUID
    language_code: str  # ISO 639-1 code (e.g., "en", "ur")
    translated_content: str
    status: str  # "pending", "completed", "reviewed"
    translator: Optional[str]  # For attribution
    created_at: datetime
    updated_at: datetime
```

## Database Relationships

### User Relationships
- One-to-Many: User → Conversations
- One-to-One: User → PersonalizationProfile

### Chapter Relationships
- One-to-Many: Chapter → ContentChunks
- One-to-Many: Chapter → ContentReferences

### ContentChunk Relationships
- One-to-Many: ContentChunk → ContentReferences
- Many-to-Many: ContentChunk ↔ Conversations (through context_chunks)

### Conversation Relationships
- One-to-Many: Conversation → Messages
- Many-to-One: Conversation → User