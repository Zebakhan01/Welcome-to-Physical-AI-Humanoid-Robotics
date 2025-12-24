<!--
SYNC IMPACT REPORT
Version change: 1.0.0 → 2.0.0
Modified principles:
- I. Spec-First Development → I. Backend-Only Development (NON-NEGOTIABLE)
- II. Educational Clarity → II. Deterministic RAG Answers (NON-NEGOTIABLE)
- III. Modularity → III. Environment Variables for Configuration (NON-NEGOTIABLE)
- IV. AI-Native Integration → IV. Spec-First Development
- V. Hardware Accuracy → V. Cohere Embeddings and Qdrant Vector Store
- VI. Course Alignment → VI. Neon Serverless PostgreSQL for Data Persistence

Added sections: None
Removed sections: None
Templates requiring updates:
- ✅ plan-template.md - updated to reflect backend-only focus
- ✅ spec-template.md - updated to reflect RAG requirements
- ✅ tasks-template.md - updated to reflect backend structure
- ✅ phr-template.prompt.md - no changes needed

Follow-up TODOs: None
-->
# Physical AI & Humanoid Robotics — Integrated RAG Chatbot Constitution

## Core Principles

### I. Backend-Only Development (NON-NEGOTIABLE)
All development work is strictly confined to the backend components within the /backend directory. No frontend code, UI elements, or client-side modifications are permitted. The existing frontend must remain completely untouched and preserved as-is. This ensures the frontend continues to function without disruption while backend capabilities are enhanced.

### II. Deterministic RAG Answers (NON-NEGOTIABLE)
All chatbot responses must derive exclusively from the textbook content and user-selected text. No external knowledge, internet calls, or hallucinated information is permitted. Answers must be traceable to specific content within the book. The system must provide citations to source material when responding to queries.

### III. Environment Variables for Configuration (NON-NEGOTIABLE)
No hardcoded secrets, API keys, or configuration values are allowed in the codebase. All sensitive information and configuration parameters must be loaded from environment variables only. This ensures secure deployment across different environments and prevents accidental exposure of credentials.

### IV. Spec-First Development
All features and implementations begin with detailed specifications before any code is written. Every API endpoint, data model, and system component must have clear requirements and acceptance criteria documented before implementation begins. Specifications serve as the single source of truth for all development activities.

### V. Cohere Embeddings and Qdrant Vector Store
The system exclusively uses Cohere for text embeddings (not OpenAI) and Qdrant Cloud for vector storage. This technology stack ensures consistent, high-quality semantic search capabilities optimized for textbook content retrieval.

### VI. Neon Serverless PostgreSQL for Data Persistence
All persistent data must be stored in Neon Serverless PostgreSQL database. This provides scalable, reliable storage for metadata, user sessions, and any other required data while maintaining cost efficiency.

## Technology Stack Requirements

### Primary Technologies
- Backend: Python 3.11 + FastAPI framework
- Embeddings: Cohere API (exclusively, no OpenAI)
- Vector Store: Qdrant Cloud
- Database: Neon Serverless PostgreSQL
- Chat Framework: OpenAI Agents / ChatKit compatible
- No training, retrieval-only RAG system

### Content Standards
- Textbook content serves as the ONLY source of truth for RAG responses
- All answers must be verifiable against provided textbook materials
- Support for user-selected text queries from the textbook
- No external knowledge integration or internet access

## Development Workflow

### Backend-First Approach
- All development activities focus exclusively on backend services in /backend directory
- API endpoints designed for frontend compatibility (existing frontend remains unchanged)
- Services built to support RAG functionality with textbook content
- Regular testing to ensure frontend-backend integration remains intact

### Quality Assurance
- RAG responses validated against textbook content accuracy
- API endpoints tested for performance and reliability
- Security review ensures no hardcoded secrets or configuration
- Performance testing for vector search and retrieval operations

### Collaboration Protocols
- All changes confined to backend components only
- Frontend compatibility maintained at all times
- Regular verification that existing frontend functionality remains intact
- Clear documentation of API contracts for frontend integration

## Governance

This constitution governs all aspects of the "Physical AI & Humanoid Robotics — Integrated RAG Chatbot" backend development. All contributors must understand and abide by these principles. Changes to this constitution require explicit approval from the project leadership team with clear justification tied to educational or technical objectives.

All pull requests and code reviews must verify compliance with constitutional principles. Backend-only work, no hardcoded secrets, and deterministic RAG answers are mandatory requirements. Use this document as the primary guidance for all development decisions.

**Version**: 2.0.0 | **Ratified**: 2025-12-14 | **Last Amended**: 2025-12-23
