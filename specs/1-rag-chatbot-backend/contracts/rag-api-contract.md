# API Contract: RAG Chatbot Backend

## Overview
REST API for the RAG (Retrieval-Augmented Generation) chatbot backend that allows students to ask questions about the Physical AI & Humanoid Robotics textbook content.

## Base URL
`/api/v1`

## Content Types
- Request: `application/json`
- Response: `application/json`

## Authentication
No authentication required for Phase 2 (as specified in requirements)

---

## Endpoints

### 1. General Question Answering
**POST** `/rag/ask`

Ask a question about the textbook content and receive an answer based on the available content.

#### Request
```json
{
  "question": "string (required)",
  "include_citations": "boolean (optional, default: true)"
}
```

#### Response (Success 200)
```json
{
  "answer": "string",
  "citations": [
    {
      "source_file": "string",
      "chapter": "string",
      "section": "string",
      "text": "string (relevant text excerpt)",
      "relevance_score": "number (0-1)"
    }
  ],
  "query_type": "general",
  "timestamp": "ISO 8601 datetime string"
}
```

#### Response (Not Found 200)
```json
{
  "answer": "This information is not available in the textbook.",
  "citations": [],
  "query_type": "general",
  "timestamp": "ISO 8601 datetime string"
}
```

#### Response (Error 400)
```json
{
  "error": "string",
  "message": "string"
}
```

---

### 2. Selected-Text Question Answering
**POST** `/rag/ask-selected`

Ask a question about specific selected text from the textbook and receive an answer based only on that selected content.

#### Request
```json
{
  "question": "string (required)",
  "selected_text": "string (required)",
  "include_citations": "boolean (optional, default: true)"
}
```

#### Response (Success 200)
```json
{
  "answer": "string",
  "citations": [
    {
      "source_file": "string",
      "chapter": "string",
      "section": "string",
      "text": "string (relevant text excerpt from selected text)",
      "relevance_score": "number (0-1)"
    }
  ],
  "query_type": "selected_text",
  "timestamp": "ISO 8601 datetime string"
}
```

#### Response (Not Found 200)
```json
{
  "answer": "This information is not available in the selected text.",
  "citations": [],
  "query_type": "selected_text",
  "timestamp": "ISO 8601 datetime string"
}
```

#### Response (Error 400)
```json
{
  "error": "string",
  "message": "string"
}
```

---

### 3. Content Indexing Status
**GET** `/content/status`

Get the indexing status of textbook content.

#### Response (Success 200)
```json
{
  "total_documents": "integer",
  "indexed_documents": "integer",
  "pending_documents": "integer",
  "failed_documents": "integer",
  "indexing_status": "string (overall status: pending, processing, completed, failed)",
  "last_indexed_at": "ISO 8601 datetime string (null if never indexed)",
  "documents": [
    {
      "file_path": "string",
      "title": "string",
      "status": "string (pending, processing, completed, failed)",
      "indexed_at": "ISO 8601 datetime string (null if not completed)"
    }
  ]
}
```

---

### 4. Re-index Content
**POST** `/content/reindex`

Trigger re-indexing of textbook content.

#### Request
```json
{
  "force_reindex": "boolean (optional, default: false)",
  "file_path": "string (optional, to reindex specific file)"
}
```

#### Response (Success 200)
```json
{
  "message": "string",
  "status": "string (processing, completed)",
  "documents_to_index": "integer"
}
```

---

### 5. Health Check
**GET** `/health`

Check the health status of the API.

#### Response (Success 200)
```json
{
  "status": "healthy",
  "timestamp": "ISO 8601 datetime string",
  "services": {
    "database": "string (connected/disconnected)",
    "vector_store": "string (connected/disconnected)",
    "embedding_service": "string (available/unavailable)"
  }
}
```

---

## Common Error Responses

### 400 Bad Request
```json
{
  "error": "validation_error",
  "message": "string describing the validation issue"
}
```

### 404 Not Found
```json
{
  "error": "not_found",
  "message": "string describing what was not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "internal_error",
  "message": "An unexpected error occurred"
}
```

---

## Query Parameters
All endpoints support the following common query parameters:
- `timeout`: Request timeout in seconds (default: 30)
- `include_citations`: Whether to include source citations (default: true)

## Headers
- `Content-Type`: `application/json` (for requests)
- `Accept`: `application/json` (for responses)
- `X-Request-ID`: Optional client-provided request identifier for tracing