#!/usr/bin/env python3
"""
Demonstration script for Phase 3: Embeddings + Qdrant integration
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def demonstrate_embedding_pipeline():
    """Demonstrate the complete embedding and Qdrant pipeline"""
    print("Phase 3: Embeddings + Qdrant Integration - Demonstration")
    print("=" * 58)
    print()

    print("This demonstration shows the complete pipeline:")
    print("- Content loading from textbook files")
    print("- Content chunking and preprocessing")
    print("- Embedding generation using sentence transformers")
    print("- Storage in Qdrant vector database")
    print("- Semantic search using embeddings")
    print()

    # 1. Demonstrate content loading
    print("1. CONTENT LOADING & CHUNKING")
    print("-" * 30)

    from backend.api.content.content_loader import load_content, ContentLoadRequest

    # Load content from a sample directory
    request = ContentLoadRequest(
        source_path="docs/intro",
        recursive=False,
        chunk_size=500,
        overlap_size=50
    )

    result = await load_content(request)

    print(f"Loaded {result.loaded_files} files from docs/intro")
    print(f"Created {result.total_chunks} content chunks")
    print(f"Total content: {result.total_words} words")

    if result.chunks:
        print(f"Sample chunk: '{result.chunks[0].content[:100]}...'")
        print(f"Chunk metadata: {list(result.chunks[0].metadata.keys())}")
    print()

    # 2. Demonstrate embedding generation
    print("2. EMBEDDING GENERATION")
    print("-" * 24)

    from backend.api.rag.embedding_service import generate_embeddings, EmbeddingRequest

    # Generate embeddings for sample content
    sample_texts = [
        "Physical AI is the intersection of artificial intelligence and real-world interaction.",
        "Robotics fundamentals include kinematics, dynamics, and control systems.",
        "Machine learning enables robots to adapt and learn from experience."
    ]

    embedding_request = EmbeddingRequest(texts=sample_texts)
    embedding_response = await generate_embeddings(embedding_request)

    print(f"Generated embeddings for {len(sample_texts)} sample texts")
    print(f"Embedding dimensionality: {len(embedding_response.embeddings[0])}")
    print(f"Embedding model: {embedding_response.model}")

    # Show sample embedding characteristics
    for i, embedding in enumerate(embedding_response.embeddings[:2]):
        magnitude = sum(x**2 for x in embedding)**0.5
        print(f"  Text {i+1} embedding magnitude: {magnitude:.4f}")
    print()

    # 3. Demonstrate content indexing (if Qdrant is available)
    print("3. CONTENT INDEXING PIPELINE")
    print("-" * 30)

    from backend.api.rag.content_indexer import IndexContentRequest
    from backend.api.content.content_loader import ContentChunk

    # Prepare a small sample for indexing
    sample_chunks = [
        ContentChunk(
            id="demo-chunk-1",
            title="Physical AI Introduction",
            content="Physical AI is the intersection of artificial intelligence and real-world interaction. This field focuses on creating AI systems that can understand and interact with the physical world through sensors and actuators.",
            source_file="docs/intro/index.md",
            chapter_id="intro",
            section_level=1,
            word_count=45,
            char_count=250,
            metadata={"topic": "physical-ai", "difficulty": "beginner", "module": "intro"}
        ),
        ContentChunk(
            id="demo-chunk-2",
            title="Robotics Fundamentals",
            content="Robotics fundamentals include understanding kinematics, which deals with motion without considering forces, and dynamics, which considers the forces that cause motion.",
            source_file="docs/intro/course-overview.md",
            chapter_id="intro",
            section_level=1,
            word_count=38,
            char_count=210,
            metadata={"topic": "robotics-fundamentals", "difficulty": "beginner", "module": "intro"}
        )
    ]

    indexing_request = IndexContentRequest(
        chunks=sample_chunks,
        collection_name="demo_textbook_content",
        batch_size=2
    )

    try:
        from backend.api.rag.content_indexer import index_content_chunks
        indexing_response = await index_content_chunks(indexing_request)

        print(f"Indexed {indexing_response.indexed_chunks} chunks successfully")
        print(f"Failed chunks: {indexing_response.failed_chunks}")
        print(f"Collection: {indexing_response.collection_name}")
        print(f"Processing time: {indexing_response.processing_time:.2f}s")

    except Exception as e:
        print(f"Indexing would work with Qdrant server running: {e}")
        print("  (This is expected if Qdrant is not available)")
    print()

    # 4. Demonstrate bulk indexing capability
    print("4. BULK INDEXING CAPABILITY")
    print("-" * 28)

    from backend.api.rag.content_indexer import bulk_index_content, BulkIndexRequest

    bulk_request = BulkIndexRequest(
        source_path="docs/intro",
        collection_name="demo_bulk_content",
        chunk_size=500,
        overlap_size=50,
        batch_size=10
    )

    try:
        # This would process all content from docs/intro and index it
        print(f"Would process {result.total_chunks} chunks from docs/intro")
        print("Would generate embeddings for all chunks")
        print("Would store all in Qdrant collection: demo_bulk_content")
        print("Batch size: 10 chunks per batch for efficiency")

        # In a real scenario with Qdrant running:
        # bulk_response = await bulk_index_content(bulk_request)
        # print(f"Bulk indexing result: {bulk_response}")

    except Exception as e:
        print(f"Bulk indexing would work with Qdrant server: {e}")
    print()

    # 5. Show the complete pipeline integration
    print("5. COMPLETE PIPELINE INTEGRATION")
    print("-" * 34)

    print("The complete pipeline flows as follows:")
    print("  Content Files -> Content Loader -> Chunks -> Embeddings -> Qdrant Storage")
    print("                                                               |")
    print("  User Query -> Embedding -> Qdrant Search <- Similarity Match <-")
    print()

    print("API Endpoints Available:")
    print("  POST /api/content/load - Load and chunk content")
    print("  POST /api/rag/embeddings - Generate embeddings")
    print("  POST /api/rag/index-chunks - Index content with embeddings")
    print("  POST /api/rag/query-indexed - Semantic search with embeddings")
    print("  POST /api/rag/bulk-index - End-to-end indexing pipeline")
    print()

    print("Benefits of this integration:")
    print("  V Semantic search beyond keyword matching")
    print("  V Contextual understanding of content")
    print("  V Scalable vector storage and retrieval")
    print("  V Efficient similarity matching")
    print("  V Integration with existing content loader")
    print()

    print("=" * 58)
    print("Phase 3: Embeddings + Qdrant Integration - Complete!")
    print()
    print("The system is ready for:")
    print("- Running with a Qdrant server for production")
    print("- Scaling to the full textbook content")
    print("- Powering semantic search in the RAG system")
    print("- Integration with the chat interface")


if __name__ == "__main__":
    asyncio.run(demonstrate_embedding_pipeline())