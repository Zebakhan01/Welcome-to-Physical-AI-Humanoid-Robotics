---
id: 1
title: rag-chatbot-backend-spec
stage: spec
date: 2025-12-23
surface: agent
model: Claude Sonnet 4.5
feature: 1-rag-chatbot-backend
branch: 1-rag-chatbot-backend
user: user
command: /sp.specify
labels: [rag, chatbot, backend, specification]
links:
  spec: specs/1-rag-chatbot-backend/spec.md
  ticket: null
  adr: null
  pr: null
files:
 - specs/1-rag-chatbot-backend/spec.md
 - specs/1-rag-chatbot-backend/checklists/requirements.md
tests:
 - null
---

## Prompt

$ARGUMENTS

PROJECT: Physical AI & Humanoid Robotics — Integrated RAG Chatbot (Phase 2)

SCOPE OF THIS SPECIFICATION:
Define ONLY Phase 2 — Integrated RAG Chatbot backend behavior.
No frontend scope.
No deployment scope.
No implementation yet.

--------------------------------------------------
FUNCTIONAL REQUIREMENTS
--------------------------------------------------

1. RAG CHATBOT CORE
- Accept natural language questions from users
- Retrieve answers strictly from textbook content
- Never use external or pretrained knowledge
- No hallucinations allowed

2. CONTENT SOURCE
- Source content comes from Docusaurus textbook markdown
- Content is pre-processed into clean text
- MDX / JSX / React code must be ignored
- Each chunk must retain source metadata (chapter, section, file)

3. EMBEDDINGS
- Use Cohere embedding models only
- Embeddings generated server-side
- Embeddings must be reproducible and deterministic

4. VECTOR STORAGE
- Store embeddings in Qdrant Cloud
- Each vector linked to content metadata
- Support similarity search and filtered retrieval

5. DATABASE (NEON POSTGRES)
- Store:
  - Document metadata
  - Chunk references
  - Indexing status
- No users, no auth in Phase 2

6. QUERY MODES
- Mode A: Normal question → retrieve relevant chunks
- Mode B: Selected-text question → restrict retrieval to selected text only

7. ANSWER GENERATION
- Inject retrieved context into LLM prompt
- LLM must answer ONLY from provided context
- If answer not found → respond with:
  "This information is not available in the textbook."

8. API DESIGN
- REST API using FastAPI
- Clear separation of services:
  - content loader
  - embedding service
  - vector store
  - retrieval service
  - answer generation service

--------------------------------------------------
ARCHITECTURE REQUIREMENTS (FUTURE-READY)
--------------------------------------------------

The backend architecture MUST be designed to support future extensions without refactoring core RAG logic:

- Future personalization (per-user preferences, history, profiles)
- Urdu translation and multilingual responses
- Chapter-level customization and scoped retrieval
- Per-chapter or per-module vector isolation
- Language-aware embedding and retrieval pipelines

NOTE:
These features are NOT implemented in Phase 2.
Architecture must only remain compatible with them.

--------------------------------------------------
NON-FUNCTIONAL REQUIREMENTS
--------------------------------------------------

- Backend must be stateless
- All secrets via environment variables
- Logs must not expose sensitive data
- Code must be modular and testable

--------------------------------------------------
OUT OF SCOPE (EXPLICIT)
--------------------------------------------------

- Frontend UI
- Chat widgets
- Authentication
- Personalization (implementation)
- Analytics
- Fine-tuning models

--------------------------------------------------
DELIVERABLE OF THIS STEP
--------------------------------------------------

- A complete Phase-2 functional specification
- No code
- No plans
- No tasks yet

--------------------------------------------------
STOP RULE
--------------------------------------------------

After producing the specification, STOP and wait.

## Response snapshot

# Feature Specification: RAG Chatbot Backend

**Feature Branch**: `1-rag-chatbot-backend`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "Physical AI & Humanoid Robotics — Integrated RAG Chatbot (Phase 2)"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Textbook Question Answering (Priority: P1)

A student studying Physical AI & Humanoid Robotics asks natural language questions about the textbook content. The system retrieves relevant information from the textbook and generates accurate answers based solely on the provided content, never using external knowledge or hallucinating information.

**Why this priority**: This is the core functionality that provides immediate value to students seeking answers from their textbook content.

**Independent Test**: Can be fully tested by submitting questions to the system and verifying that responses are based only on textbook content with proper citations. Delivers direct value of textbook information retrieval.

**Acceptance Scenarios**:
1. **Given** a user has access to the RAG system, **When** they ask a question about textbook content, **Then** the system returns an answer based only on the textbook with proper source citations
2. **Given** a user asks a question not covered in the textbook, **When** the system processes the query, **Then** it responds with "This information is not available in the textbook."

---

### User Story 2 - Selected Text Querying (Priority: P2)

A student selects specific text from the textbook and asks questions about that selected content. The system restricts its retrieval to only the selected text, providing focused answers based on the limited scope.

**Why this priority**: Enables students to ask detailed questions about specific sections they're studying, providing more targeted assistance.

**Independent Test**: Can be tested by selecting specific text and asking questions, verifying that responses are restricted to the selected content only. Delivers value of focused, contextual answers.

**Acceptance Scenarios**:
1. **Given** a user has selected specific text from the textbook, **When** they ask a question about that text, **Then** the system returns answers based only on the selected text
2. **Given** a user has selected specific text, **When** they ask a question outside the selected scope, **Then** the system responds appropriately based on the restricted context

---

### User Story 3 - Content Management and Indexing (Priority: P3)

An administrator manages the textbook content by ensuring it's properly indexed for retrieval. The system processes Docusaurus markdown content, extracts clean text while ignoring code blocks, and maintains metadata linking to source locations.

**Why this priority**: Essential infrastructure that enables the core functionality to work properly with the textbook content.

**Independent Test**: Can be tested by verifying that content is properly indexed and retrievable. Delivers the foundational capability for all other features.

**Acceptance Scenarios**:
1. **Given** new textbook content exists in Docusaurus markdown format, **When** the indexing process runs, **Then** the content is properly stored with metadata and ready for retrieval
2. **Given** textbook content with MDX/JSX/React code, **When** the content is processed, **Then** the code is ignored and only clean text is indexed with proper source metadata

---

### Edge Cases

- What happens when the question is ambiguous and could match multiple textbook sections?
- How does the system handle very long or very short questions?
- What occurs when the vector store is temporarily unavailable?
- How does the system respond to questions that require complex multi-step reasoning from the text?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST retrieve answers exclusively from textbook content using RAG (no external knowledge)
- **FR-002**: System MUST use Cohere API for text embeddings (not OpenAI)
- **FR-003**: System MUST store vectors in Qdrant Cloud vector database
- **FR-004**: System MUST store persistent data in Neon Serverless PostgreSQL
- **FR-005**: System MUST load all configuration from environment variables (no hardcoded values)
- **FR-006**: System MUST support user-selected text queries from the textbook
- **FR-007**: System MUST provide citations to source material when responding to queries
- **FR-008**: System MUST process Docusaurus markdown content and extract clean text while ignoring MDX/JSX/React code
- **FR-009**: System MUST maintain source metadata (chapter, section, file) for each indexed content chunk
- **FR-010**: System MUST provide REST API endpoints for question submission and answer retrieval
- **FR-011**: System MUST separate services for content loading, embedding, vector storage, retrieval, and answer generation
- **FR-012**: System MUST answer ONLY from provided context when generating responses
- **FR-013**: System MUST respond with "This information is not available in the textbook" when content is not found
- **FR-014**: System MUST generate reproducible and deterministic embeddings

### Key Entities *(include if feature involves data)*

- **Document**: Represents a textbook chapter/section with metadata including source file path, chapter, and section information
- **ContentChunk**: A processed segment of textbook content with embedded vector representation and source reference
- **IndexStatus**: Tracks the indexing state of documents (processed, pending, failed)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Students can ask questions about textbook content and receive accurate answers within 5 seconds
- **SC-002**: 95% of valid textbook-related questions return answers with proper source citations
- **SC-003**: System successfully processes 100% of Docusaurus markdown content without errors
- **SC-004**: 90% of questions not covered in the textbook properly return "This information is not available in the textbook" message
- **SC-005**: Content indexing completes within 10 minutes for a typical textbook size (up to 500 pages of content)
- **SC-006**: Selected text query mode returns results that are 100% restricted to the specified text scope
- **SC-007**: System maintains 99% uptime during active usage periods
- **SC-008**: All configuration parameters are loaded from environment variables with no hardcoded values in the codebase

## Outcome

- ✅ Impact: Created complete functional specification for RAG Chatbot Backend with user scenarios, functional requirements, and success criteria
- 🧪 Tests: No tests created yet (specification phase)
- 📁 Files: Created spec.md and requirements checklist
- 🔁 Next prompts: Ready for planning phase (/sp.plan)
- 🧠 Reflection: Specification addresses all requirements from user input with clear user stories, functional requirements, and measurable success criteria

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): Standard specification generation
- Next experiment (smallest change to try): Proceed to planning phase