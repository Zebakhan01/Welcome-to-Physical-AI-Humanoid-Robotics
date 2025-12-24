# Data Model: RAG Chatbot Backend

## Entity: Document
**Purpose**: Represents a textbook chapter/section with metadata including source file path, chapter, and section information

**Fields**:
- `id` (UUID): Unique identifier for the document
- `file_path` (String): Path to the source file in the Docusaurus structure
- `title` (String): Title of the document/chapter
- `chapter` (String): Chapter identifier or name
- `section` (String): Section identifier or name
- `content_hash` (String): Hash of the content for change detection
- `created_at` (DateTime): Timestamp of document creation
- `updated_at` (DateTime): Timestamp of last update
- `index_status` (String): Current status of indexing (pending, processing, completed, failed)

**Relationships**:
- One-to-Many: Document has many ContentChunks
- IndexStatus is embedded in the document record

**Validation Rules**:
- `file_path` must be a valid path within the textbook directory
- `title` is required
- `index_status` must be one of: "pending", "processing", "completed", "failed"

## Entity: ContentChunk
**Purpose**: A processed segment of textbook content with embedded vector representation and source reference

**Fields**:
- `id` (UUID): Unique identifier for the chunk
- `document_id` (UUID): Reference to the parent Document
- `content` (Text): The actual text content of the chunk
- `chunk_index` (Integer): Position of this chunk within the document
- `vector_id` (String): ID of the vector in Qdrant Cloud
- `source_metadata` (JSON): Additional metadata from the source (headings, etc.)
- `created_at` (DateTime): Timestamp of chunk creation
- `updated_at` (DateTime): Timestamp of last update

**Relationships**:
- Many-to-One: ContentChunk belongs to Document
- The vector_id links to the corresponding vector in Qdrant Cloud

**Validation Rules**:
- `content` must not be empty
- `document_id` must reference an existing Document
- `chunk_index` must be non-negative
- `vector_id` must be valid for Qdrant

## Entity: IndexStatus
**Purpose**: Tracks the indexing state of documents (processed, pending, failed)
**Note**: This is primarily stored as a field in the Document entity but may be tracked separately for complex workflows

**Fields**:
- `status` (String): Current status ("pending", "processing", "completed", "failed")
- `started_at` (DateTime): When indexing started
- `completed_at` (DateTime): When indexing completed (if applicable)
- `error_message` (String): Error details if status is "failed"

**Relationships**:
- Embedded in Document entity

**Validation Rules**:
- `status` must be one of: "pending", "processing", "completed", "failed"
- `completed_at` must be after `started_at` if both are set

## Entity: APIRequestLog (Optional for future use)
**Purpose**: For monitoring and analytics (future-ready as per architecture requirements)

**Fields**:
- `id` (UUID): Unique identifier
- `query` (String): The original question
- `response` (String): The generated answer
- `source_citations` (JSON): List of sources used in the answer
- `query_type` (String): "general" or "selected_text"
- `created_at` (DateTime): Timestamp of the request

**Relationships**:
- No direct relationships to core RAG entities
- For analytics purposes only

## State Transitions

### Document Index Status Transitions:
1. `pending` → `processing` when indexing begins
2. `processing` → `completed` when indexing finishes successfully
3. `processing` → `failed` when indexing encounters an error

## Data Relationships

```
Document (1) ─── (n) ContentChunk
    │
    │─── contains
    │
    └─── IndexStatus (embedded)
```

The Document entity serves as the parent for multiple ContentChunk entities, each representing a segment of the original textbook content. Each ContentChunk contains a reference to its vector representation in Qdrant Cloud via the `vector_id` field.

## Database Constraints

1. **Foreign Key Constraints**:
   - ContentChunk.document_id references Document.id
   - Enforced at the database level

2. **Uniqueness Constraints**:
   - Document.file_path should be unique to prevent duplicate indexing
   - ContentChunk.vector_id should be unique across all chunks

3. **Check Constraints**:
   - ContentChunk.chunk_index >= 0
   - Document.index_status in ("pending", "processing", "completed", "failed")

## Future-Ready Considerations

As per the architecture requirements, the data model is designed to support future extensions:

1. **Personalization**: Additional user-specific tables could be added without affecting core RAG entities
2. **Multilingual Support**: Content and metadata fields could be extended to support multiple languages
3. **Chapter-Level Customization**: Document entity already includes chapter information for scoped operations
4. **Per-Module Isolation**: The document_id foreign key in ContentChunk enables module-level filtering