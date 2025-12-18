#!/usr/bin/env python3
"""
Test script to verify the RAG system with textbook content
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from backend.api.content.content_parser import index_content, ContentIndexRequest

async def test_content_indexing():
    """Test the content indexing pipeline"""
    print("Testing content indexing pipeline...")

    # Create a request to index all textbook content
    request = ContentIndexRequest(
        source_path="docs",
        recursive=True
    )

    try:
        result = await index_content(request)
        print(f"✅ Successfully indexed {result.indexed_files} files")
        print(f"✅ Processed {len(result.processed_content)} content sections")

        if result.processed_content:
            print(f"✅ First few indexed items:")
            for i, item in enumerate(result.processed_content[:3]):
                print(f"   - File: {item['file']}")
                print(f"   - Section: {item['section']}")
                print(f"   - Vector ID: {item['vector_id']}")
                print()

        return True
    except Exception as e:
        print(f"X Error during content indexing: {e}")
        return False

async def test_rag_functionality():
    """Test the RAG functionality by simulating a chat query"""
    print("Testing RAG functionality...")

    try:
        # Import the chat and retrieval modules
        from backend.api.chat.chat_routes import chat_message
        from backend.utils.validators import ChatRequest

        # Create a sample chat request
        chat_request = ChatRequest(
            message="What is Physical AI?",
            conversation_id="test-conversation-123"
        )

        # Process the chat message using RAG
        response = await chat_message(chat_request)

        print(f"✅ Chat response generated successfully")
        print(f"✅ Response preview: {response.response[:200]}...")
        print(f"✅ Sources found: {len(response.sources)}")

        if response.sources:
            print("✅ Top source:")
            top_source = response.sources[0]
            print(f"   - Title: {top_source['title']}")
            print(f"   - Score: {top_source['score']}")
            print(f"   - Preview: {top_source['content_preview'][:100]}...")

        return True
    except Exception as e:
        print(f"X Error during RAG functionality test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("Starting RAG System Tests")
    print("=" * 50)

    # Test content indexing
    indexing_success = await test_content_indexing()
    print()

    # Test RAG functionality
    rag_success = await test_rag_functionality()
    print()

    print("=" * 50)
    if indexing_success and rag_success:
        print("All tests passed! RAG system is working correctly.")
        return True
    else:
        print("Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)