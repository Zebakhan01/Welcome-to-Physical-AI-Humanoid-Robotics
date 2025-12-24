# Implementation Plan: RAG Chatbot Backend

**Branch**: `1-rag-chatbot-backend` | **Date**: 2025-12-23 | **Spec**: [specs/1-rag-chatbot-backend/spec.md](./spec.md)
**Input**: Feature specification from `/specs/1-rag-chatbot-backend/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG (Retrieval-Augmented Generation) chatbot backend for the Physical AI & Humanoid Robotics textbook. The system will accept natural language questions, retrieve relevant content from textbook materials using Cohere embeddings and Qdrant vector storage, and generate answers based solely on the provided content without external knowledge. The backend will support both general textbook queries and selected-text restricted retrieval modes, with all responses including proper source citations.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, Cohere API, Qdrant Cloud, Neon Serverless PostgreSQL, Pydantic
**Storage**: Neon Serverless PostgreSQL for metadata, Qdrant Cloud for vector storage
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: web (backend-only)
**Performance Goals**: Response time under 5 seconds for 95% of queries
**Constraints**: Deterministic RAG (textbook-only answers), no external knowledge, environment variable configuration
**Scale/Scope**: Single textbook content, supporting concurrent student users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Backend-only development: All work confined to /backend directory, no frontend modifications
- [x] Deterministic RAG: Responses only from textbook content, no external knowledge
- [x] Environment variables: No hardcoded secrets, all config via env vars
- [x] Cohere embeddings: Use Cohere API exclusively (not OpenAI)
- [x] Qdrant vector store: Use Qdrant Cloud for vector storage
- [x] Neon Serverless PostgreSQL: Use for all persistent data storage

## Project Structure

### Documentation (this feature)
```text
specs/1-rag-chatbot-backend/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)
```text
backend/
├── src/
│   ├── models/
│   │   ├── document.py
│   │   ├── content_chunk.py
│   │   └── index_status.py
│   ├── services/
│   │   ├── content_loader.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── retrieval_service.py
│   │   └── answer_generation.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   │   ├── rag.py
│   │   │   └── content.py
│   │   └── dependencies.py
│   └── config/
│       └── settings.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── requirements.txt
```

**Structure Decision**: Backend-only structure selected to comply with constitution requirement of backend-only development. All implementation will be within the /backend directory to ensure no frontend modifications occur.

## Phase 1: Cleanup and Environment Setup

### Step 1.1: Repository Cleanup
- [ ] Delete all existing backend code as required by constitution
- [ ] Create new /backend directory structure
- [ ] Set up Python virtual environment
- [ ] Initialize requirements.txt with dependencies (FastAPI, Cohere, Qdrant, SQLAlchemy, etc.)

### Step 1.2: Configuration Setup
- [ ] Create configuration module to handle environment variables
- [ ] Define required environment variables for Cohere API, Qdrant Cloud, and Neon PostgreSQL
- [ ] Implement configuration validation
- [ ] Create .env.example file with documentation

**Validation Checkpoint**: Environment variables are properly configured and accessible
**STOP**: Confirm all hardcoded secrets would be prevented by design

## Phase 2: Data Models and Database Setup

### Step 2.1: Define Data Models
- [ ] Create Document model with chapter, section, file path metadata
- [ ] Create ContentChunk model with vector embeddings and source references
- [ ] Create IndexStatus model to track processing state
- [ ] Implement Pydantic schemas for API serialization

### Step 2.2: Database Integration
- [ ] Set up SQLAlchemy models for Neon PostgreSQL
- [ ] Create database connection utilities
- [ ] Implement database migration framework
- [ ] Define database session management

**Validation Checkpoint**: Data models correctly represent the entities from the specification
**STOP**: Confirm database schema aligns with requirements

## Phase 3: Content Processing and Loading

### Step 3.1: Docusaurus Content Parser
- [ ] Implement markdown parser for Docusaurus content
- [ ] Create logic to ignore MDX/JSX/React code blocks
- [ ] Extract clean text content while preserving structure
- [ ] Maintain source metadata (chapter, section, file path)

### Step 3.2: Content Loading Service
- [ ] Create service to load content from Docusaurus directory
- [ ] Implement content chunking logic for optimal retrieval
- [ ] Store document metadata in PostgreSQL
- [ ] Track indexing status for each document

**Validation Checkpoint**: Content is properly parsed, chunked, and stored with metadata
**STOP**: Confirm content processing meets requirements before proceeding

## Phase 4: Embedding Service Implementation

### Step 4.1: Cohere API Integration
- [ ] Create Cohere client wrapper
- [ ] Implement embedding generation function using Cohere models
- [ ] Ensure deterministic embeddings for reproducible results
- [ ] Handle API rate limiting and errors

### Step 4.2: Embedding Generation Pipeline
- [ ] Create service to generate embeddings for content chunks
- [ ] Implement batch processing for efficiency
- [ ] Store embeddings with content chunks
- [ ] Validate embedding quality and consistency

**Validation Checkpoint**: Embeddings are generated correctly and deterministically
**STOP**: Confirm embeddings meet quality requirements

## Phase 5: Vector Storage Setup

### Step 5.1: Qdrant Cloud Integration
- [ ] Set up Qdrant client for cloud connection
- [ ] Create collection for storing content embeddings
- [ ] Implement vector storage utilities
- [ ] Define vector schema matching content chunks

### Step 5.2: Vector Indexing
- [ ] Create service to store embeddings in Qdrant
- [ ] Link vectors to PostgreSQL metadata
- [ ] Implement vector search capabilities
- [ ] Add metadata filtering for retrieval

**Validation Checkpoint**: Vectors are stored and searchable in Qdrant
**STOP**: Confirm vector storage and retrieval functionality

## Phase 6: Retrieval Service

### Step 6.1: Basic Retrieval Implementation
- [ ] Create service to retrieve relevant chunks using vector similarity
- [ ] Implement similarity search with Qdrant
- [ ] Return chunks with source metadata for citations
- [ ] Add result ranking and filtering

### Step 6.2: Selected-Text Retrieval
- [ ] Implement mode for restricting retrieval to specific text
- [ ] Add filtering to limit results to selected content
- [ ] Ensure proper scope isolation for selected-text queries
- [ ] Validate restricted retrieval works as expected

**Validation Checkpoint**: Both general and selected-text retrieval modes work correctly
**STOP**: Confirm retrieval accuracy and scope restrictions

## Phase 7: Answer Generation Service

### Step 7.1: Context Injection
- [ ] Create service to format retrieved context for LLM
- [ ] Implement proper context formatting with citations
- [ ] Ensure context includes source metadata
- [ ] Validate context quality and relevance

### Step 7.2: Answer Generation
- [ ] Implement answer generation using provided context only
- [ ] Create logic to detect when answer is not in context
- [ ] Return "This information is not available in the textbook" when appropriate
- [ ] Ensure no external knowledge is used in responses

**Validation Checkpoint**: Answers are generated from context only with proper citations
**STOP**: Confirm deterministic RAG behavior

## Phase 8: API Development

### Step 8.1: Core API Endpoints
- [ ] Create FastAPI application structure
- [ ] Implement question answering endpoint
- [ ] Add selected-text query endpoint
- [ ] Create content indexing endpoint

### Step 8.2: API Validation and Error Handling
- [ ] Add request/response validation
- [ ] Implement proper error handling
- [ ] Add API documentation with OpenAPI
- [ ] Create health check endpoints

**Validation Checkpoint**: All API endpoints function correctly and meet specification
**STOP**: Confirm API meets all functional requirements

## Phase 9: Integration and Testing

### Step 9.1: End-to-End Testing
- [ ] Create integration tests for complete RAG pipeline
- [ ] Test both query modes (general and selected-text)
- [ ] Validate response accuracy and citations
- [ ] Test error handling and edge cases

### Step 9.2: Performance Validation
- [ ] Test response times meet performance goals
- [ ] Validate system handles expected load
- [ ] Confirm content indexing completes within time limits
- [ ] Verify all success criteria from specification

**Validation Checkpoint**: Complete system meets all functional and performance requirements
**STOP**: Final validation before tasks breakdown

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | All requirements comply with constitution | N/A |