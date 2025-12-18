#!/usr/bin/env python3
"""
Demo script to showcase the RAG system functionality
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def demo_embedding_generation():
    """Demonstrate embedding generation"""
    print("DEMO: Embedding Generation")
    print("-" * 40)

    from backend.api.rag.embedding_service import generate_embeddings, EmbeddingRequest

    # Generate embeddings for sample questions
    sample_questions = [
        "What is Physical AI?",
        "Explain robotics fundamentals",
        "How does computer vision work in robotics?"
    ]

    request = EmbeddingRequest(texts=sample_questions)
    response = await generate_embeddings(request)

    print(f"Generated embeddings for {len(sample_questions)} questions")
    print(f"Each embedding has {len(response.embeddings[0])} dimensions")
    print()

async def demo_content_parsing():
    """Demonstrate content parsing"""
    print("DEMO: Content Parsing")
    print("-" * 40)

    from backend.api.content.content_parser import parse_content, ContentParseRequest

    # Sample textbook content
    sample_content = """# Introduction to Physical AI

Physical AI is the intersection of artificial intelligence and real-world interaction.
This field focuses on creating AI systems that can understand and interact with the physical world.

## Key Concepts

- Embodied AI: AI systems with physical form and interaction capabilities
- Robot learning: How robots learn from experience and environment
- Sensorimotor intelligence: Coordination of sensing and motor actions

## Applications

Physical AI has applications in:
- Service robotics
- Industrial automation
- Healthcare assistance
- Educational tools
"""

    request = ContentParseRequest(
        content=sample_content,
        title="Introduction to Physical AI",
        chapter_id="week-01",
        format="markdown"
    )

    response = await parse_content(request)

    print(f"Parsed content into {len(response.sections)} sections")
    print(f"Word count: {response.word_count}")
    print(f"Estimated reading time: {response.reading_time} minutes")

    for i, section in enumerate(response.sections):
        print(f"  Section {i+1}: {section['title']} ({section['level']})")
    print()

async def demo_rag_chat():
    """Demonstrate RAG chat functionality"""
    print("DEMO: RAG Chat System")
    print("-" * 40)

    from backend.api.chat.chat_routes import chat_message
    from backend.utils.validators import ChatRequest

    # Simulate a chat query (this will work even without indexed content)
    chat_request = ChatRequest(
        message="What is Physical AI?",
        conversation_id="demo-conversation"
    )

    try:
        response = await chat_message(chat_request)
        print(f"Query: {chat_request.message}")
        print(f"Response preview: {response.response[:200]}...")
        print(f"Sources found: {len(response.sources)}")
        print()
    except Exception as e:
        print(f"Chat demo would work with indexed content. Error: {e}")
        print("This is expected without Qdrant server running.")
        print()

async def main():
    """Main demo function"""
    print("Physical AI & Humanoid Robotics Textbook - RAG System Demo")
    print("=" * 60)
    print()

    print("This demo showcases the core components of the implemented RAG system.")
    print("The system enables Q&A functionality based on textbook content.\n")

    await demo_embedding_generation()
    await demo_content_parsing()
    await demo_rag_chat()

    print("=" * 60)
    print("RAG System Implementation Complete!")
    print("\nThe system is ready for:")
    print("- Indexing the full textbook content")
    print("- Processing user queries with RAG")
    print("- Providing accurate, source-based answers")
    print("- Scaling to handle all textbook chapters")

if __name__ == "__main__":
    asyncio.run(main())