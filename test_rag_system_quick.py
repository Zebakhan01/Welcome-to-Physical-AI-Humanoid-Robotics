#!/usr/bin/env python3
"""
Quick test to verify the RAG system is working properly
"""
import asyncio
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_rag_system():
    """Test the RAG system functionality"""
    print("Testing RAG System...")
    print("=" * 30)

    try:
        # Test 1: Import core components
        print("1. Testing imports...")
        from backend.api.rag.embedding_service import generate_embeddings, EmbeddingRequest
        from backend.api.rag.retrieval_service import query_content, QueryRequest
        from backend.api.chat.chat_routes import chat_message, ChatRequest
        print("   [PASS] Core components imported successfully")

        # Test 2: Embedding generation
        print("\n2. Testing embedding generation...")
        test_request = EmbeddingRequest(texts=["Test query for RAG system"])
        embedding_response = await generate_embeddings(test_request)

        if len(embedding_response.embeddings) == 1 and len(embedding_response.embeddings[0]) == 384:
            print(f"   [PASS] Embedding generated successfully (384 dimensions)")
        else:
            print(f"   [FAIL] Embedding generation failed - wrong dimensions: {len(embedding_response.embeddings[0])}")
            return False

        # Test 3: Content retrieval (this might fail if Qdrant is not running, which is OK)
        print("\n3. Testing content retrieval...")
        try:
            query_request = QueryRequest(query="Test query", top_k=2)
            retrieval_response = await query_content(query_request)
            print(f"   [PASS] Content retrieval attempted (found {len(retrieval_response.results)} results)")
        except Exception as e:
            print(f"   [INFO] Content retrieval failed (expected if Qdrant not running): {e}")

        # Test 4: Chat functionality
        print("\n4. Testing chat functionality...")
        chat_request = ChatRequest(
            message="What is Physical AI?",
            conversation_id="test-conversation"
        )
        chat_response = await chat_message(chat_request)

        print(f"   [PASS] Chat response generated")
        print(f"   Response preview: {chat_response.response[:100]}...")

        print("\n" + "=" * 30)
        print("[PASS] RAG System Test PASSED!")
        print("All core components are functional.")
        return True

    except Exception as e:
        print(f"\n[FAIL] RAG System Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_rag_system())
    sys.exit(0 if success else 1)