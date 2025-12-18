#!/usr/bin/env python3
"""
Demonstration script for the content loader functionality (Phase 2)
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def demonstrate_content_loader():
    """Demonstrate the content loader functionality"""
    print("Content Loader (Phase 2) - Demonstration")
    print("=" * 50)
    print()

    print("This demonstration shows the content loader capabilities:")
    print("- Loading textbook content without embeddings")
    print("- Splitting content into manageable chunks")
    print("- Providing content search functionality")
    print("- Analyzing content structure")
    print()

    # 1. Demonstrate content loading
    print("1. CONTENT LOADING")
    print("-" * 20)

    from backend.api.content.content_loader import load_content, ContentLoadRequest

    request = ContentLoadRequest(
        source_path="docs",
        recursive=True,
        chunk_size=500,  # Smaller chunks for demonstration
        overlap_size=50
    )

    result = await load_content(request)

    print(f"Loaded {result.loaded_files} files from the textbook")
    print(f"Created {result.total_chunks} content chunks")
    print(f"Total content: {result.total_words} words, {result.processing_stats['total_chars']} characters")
    print(f"Average chunk size: {result.processing_stats['avg_chunk_size']:.1f} characters")
    print()

    # Show sample chunks
    print("Sample chunks created:")
    for i, chunk in enumerate(result.chunks[:3]):
        print(f"  Chunk {i+1}:")
        print(f"    ID: {chunk.id}")
        print(f"    Title: {chunk.title}")
        print(f"    Source: {chunk.source_file}")
        print(f"    Chapter: {chunk.chapter_id}")
        print(f"    Words: {chunk.word_count}, Chars: {chunk.char_count}")
        print(f"    Preview: {chunk.content[:100]}...")
        print()

    print()

    # 2. Demonstrate content search
    print("2. CONTENT SEARCH")
    print("-" * 20)

    from backend.api.content.content_loader import search_content, ContentSearchRequest

    search_request = ContentSearchRequest(
        query="robotics fundamentals",
        source_path="docs",
        limit=3
    )

    search_result = await search_content(search_request)

    print(f"Searched for: '{search_result.search_query}'")
    print(f"Found {search_result.total_results} results")
    print()

    print("Top search results:")
    for i, result in enumerate(search_result.results):
        chunk = result.chunk
        print(f"  Result {i+1} (Score: {result.relevance_score:.3f}):")
        print(f"    Title: {chunk.title}")
        print(f"    Source: {chunk.source_file}")
        print(f"    Matched terms: {result.matched_terms}")
        print(f"    Preview: {chunk.content[:150]}...")
        print()

    print()

    # 3. Demonstrate content structure analysis
    print("3. CONTENT STRUCTURE ANALYSIS")
    print("-" * 30)

    from backend.api.content.content_loader import get_content_structure, ContentStructureRequest

    structure_request = ContentStructureRequest(
        source_path="docs",
        recursive=True
    )

    structure_result = await get_content_structure(structure_request)

    print(f"Textbook structure analysis:")
    print(f"  Total files: {structure_result.total_files}")
    print(f"  Total directories: {structure_result.total_directories}")
    print(f"  Root directory: {structure_result.structure.name}")
    print()

    print("Directory structure (first few levels):")
    def print_structure(node, depth=0):
        indent = "  " * depth
        node_type = "[DIR]" if node.type == "directory" else "[FILE]"
        print(f"{indent}{node_type} {node.name}")

        # Print first few children
        for child in node.children[:3]:  # Limit to first 3 children
            print_structure(child, depth + 1)

        if len(node.children) > 3:
            print(f"{indent}  ... and {len(node.children) - 3} more")

    print_structure(structure_result.structure)
    print()

    print("=" * 50)
    print("Content Loader Demonstration Complete!")
    print()
    print("Key Features Demonstrated:")
    print("V Content loading without embeddings")
    print("V Smart text chunking with overlap")
    print("V Keyword-based content search")
    print("V Directory structure analysis")
    print("V Rich metadata extraction")
    print()
    print("The content loader is ready for Phase 3 integration with embeddings!")


if __name__ == "__main__":
    asyncio.run(demonstrate_content_loader())