# Research: RAG Chatbot Backend Implementation

## Decision: Technology Stack Selection
**Rationale**: Selected Python 3.11 with FastAPI based on project requirements for a REST backend that needs to integrate with Cohere, Qdrant Cloud, and Neon PostgreSQL. FastAPI provides excellent async support, automatic API documentation, and strong typing capabilities needed for the RAG system.

**Alternatives considered**:
- Flask: Simpler but lacks async support and automatic documentation
- Node.js/Express: Could work but Python has better ML/NLP libraries for RAG
- Go: Good performance but less ecosystem support for the specific AI tools required

## Decision: Cohere Embedding Models
**Rationale**: Cohere's embed-multilingual-v3.0 model selected as it supports multiple languages and provides high-quality embeddings suitable for textbook content. The model is specifically designed for retrieval tasks and integrates well with RAG systems.

**Alternatives considered**:
- OpenAI embeddings: Prohibited by constitution requirement to use Cohere only
- Sentence Transformers: Self-hosted option but requires more infrastructure
- Other commercial APIs: Limited by requirement to use Cohere

## Decision: Qdrant Vector Database
**Rationale**: Qdrant Cloud selected as it provides managed vector storage with similarity search capabilities needed for RAG. It offers good performance, scalability, and integrates well with Python applications.

**Alternatives considered**:
- Pinecone: Commercial alternative but Qdrant has better open-source roots
- Weaviate: Good alternative but Cohere integration is more straightforward with Qdrant
- Self-hosted solutions: More control but requires more infrastructure management

## Decision: Content Chunking Strategy
**Rationale**: Using recursive character text splitter with 1000-character chunks and 200-character overlap. This size provides a good balance between context retention and retrieval precision for textbook content.

**Alternatives considered**:
- Sentence-based splitting: Could work but might create uneven chunks
- Paragraph-based splitting: Might create chunks too large for context windows
- Custom splitting: Would require more development time without clear benefits

## Decision: Answer Generation Approach
**Rationale**: Using a structured prompt with explicit instructions to only use provided context. This ensures deterministic RAG behavior as required by the specification. The system will be prompted to explicitly state when information is not available in the provided context.

**Alternatives considered**:
- More complex reasoning chains: Would risk hallucination
- Fine-tuning: Prohibited by constitution (retrieval-only RAG system)
- External knowledge integration: Prohibited by requirements

## Decision: API Architecture
**Rationale**: REST API with FastAPI provides clear separation of concerns with dedicated endpoints for different query types. This aligns with the requirement to support both general and selected-text query modes.

**Alternatives considered**:
- GraphQL: More flexible but unnecessary complexity for this use case
- gRPC: Better for internal services but REST is more appropriate for this context
- Serverless functions: Could work but FastAPI provides better state management for RAG

## Key Findings

1. **Environment Variables**: All configuration will be managed through environment variables using Pydantic's BaseSettings, ensuring no hardcoded values per constitution requirements.

2. **Textbook Content Processing**: Docusaurus markdown can be processed using the `markdown` library with custom extensions to ignore MDX/JSX/React code blocks while preserving content structure.

3. **Metadata Tracking**: Source file, chapter, and section information will be preserved in both PostgreSQL (for structured metadata) and Qdrant (for retrieval context).

4. **Performance Considerations**: Caching strategies will be implemented for frequently accessed embeddings to meet the 5-second response time requirement.

5. **Error Handling**: Comprehensive error handling will be implemented to ensure the system responds appropriately when the vector store is unavailable or content is not found.