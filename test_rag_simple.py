#!/usr/bin/env python3
"""
Simple test script to verify the RAG system components
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def test_embedding_service():
    """Test the embedding service"""
    print("Testing embedding service...")

    try:
        from backend.api.rag.embedding_service import generate_embeddings, EmbeddingRequest

        # Test embedding generation
        request = EmbeddingRequest(texts=["Hello, this is a test sentence."])
        response = await generate_embeddings(request)

        print(f"V Embedding service working - Generated {len(response.embeddings)} embeddings")
        print(f"V First embedding length: {len(response.embeddings[0])}")

        return True
    except Exception as e:
        print(f"X Error in embedding service: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vector_store_connection():
    """Test the vector store connection"""
    print("Testing vector store connection...")

    try:
        from backend.api.rag.vector_store import vector_store_health

        # Test health check
        response = await vector_store_health()

        print(f"V Vector store health check completed")
        print(f"V Connected: {response.connected}")
        print(f"V Status: {response.status}")

        return True
    except Exception as e:
        print(f"X Error in vector store connection: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_content_parsing():
    """Test the content parsing service"""
    print("Testing content parsing...")

    try:
        from backend.api.content.content_parser import parse_content, ContentParseRequest

        # Test content parsing
        test_content = """# Introduction to Physical AI

Physical AI is the intersection of artificial intelligence and real-world interaction.
This field focuses on creating AI systems that can understand and interact with the physical world.

## Key Concepts

- Embodied AI
- Robot learning
- Sensorimotor intelligence
"""

        request = ContentParseRequest(
            content=test_content,
            title="Test Chapter",
            chapter_id="intro",
            format="markdown"
        )

        response = await parse_content(request)

        print(f"V Content parsing working - Found {len(response.sections)} sections")
        print(f"V Word count: {response.word_count}")
        print(f"V Reading time: {response.reading_time} minutes")

        return True
    except Exception as e:
        print(f"X Error in content parsing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("Starting RAG System Component Tests")
    print("=" * 50)

    # Test embedding service
    embedding_success = await test_embedding_service()
    print()

    # Test vector store connection
    vector_success = await test_vector_store_connection()
    print()

    # Test content parsing
    parsing_success = await test_content_parsing()
    print()

    print("=" * 50)
    if embedding_success and vector_success and parsing_success:
        print("All component tests passed! RAG system components are working correctly.")
        return True
    else:
        print("Some component tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)