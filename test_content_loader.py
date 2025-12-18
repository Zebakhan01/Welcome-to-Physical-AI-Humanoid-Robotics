#!/usr/bin/env python3
"""
Test script for the content loader functionality (Phase 2)
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path so we can import backend modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

async def test_content_loading():
    """Test the content loading functionality"""
    print("Testing Content Loading (Phase 2)...")
    print("-" * 40)

    try:
        from backend.api.content.content_loader import load_content, ContentLoadRequest

        # Create a request to load content from the docs directory
        request = ContentLoadRequest(
            source_path="docs",
            recursive=True,
            chunk_size=1000,
            overlap_size=100
        )

        result = await load_content(request)

        print(f"V Loaded {result.loaded_files} files")
        print(f"V Created {result.total_chunks} content chunks")
        print(f"V Total words: {result.total_words}")
        print(f"V Average chunk size: {result.processing_stats['avg_chunk_size']:.2f} characters")
        print(f"V Total characters: {result.processing_stats['total_chars']}")

        # Show some sample chunks
        if result.chunks:
            print(f"\nSample chunks:")
            for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
                print(f"  Chunk {i+1}:")
                print(f"    ID: {chunk.id}")
                print(f"    Title: {chunk.title}")
                print(f"    Source: {chunk.source_file}")
                print(f"    Chapter: {chunk.chapter_id}")
                print(f"    Content preview: {chunk.content[:100]}...")
                print(f"    Words: {chunk.word_count}, Chars: {chunk.char_count}")
                print()

        return True

    except Exception as e:
        print(f"X Error in content loading: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_content_search():
    """Test the content search functionality"""
    print("Testing Content Search...")
    print("-" * 40)

    try:
        from backend.api.content.content_loader import search_content, ContentSearchRequest

        # Create a search request
        request = ContentSearchRequest(
            query="Physical AI",
            source_path="docs",
            limit=5
        )

        result = await search_content(request)

        print(f"V Found {result.total_results} results for query '{result.search_query}'")

        if result.results:
            print(f"\nTop results:")
            for i, search_result in enumerate(result.results[:3]):  # Show top 3
                chunk = search_result.chunk
                print(f"  Result {i+1}:")
                print(f"    Score: {search_result.relevance_score:.3f}")
                print(f"    Title: {chunk.title}")
                print(f"    Source: {chunk.source_file}")
                print(f"    Preview: {chunk.content[:150]}...")
                print(f"    Matched terms: {search_result.matched_terms}")
                print()

        return True

    except Exception as e:
        print(f"X Error in content search: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_content_structure():
    """Test the content structure functionality"""
    print("Testing Content Structure...")
    print("-" * 40)

    try:
        from backend.api.content.content_loader import get_content_structure, ContentStructureRequest

        # Create a structure request
        request = ContentStructureRequest(
            source_path="docs",
            recursive=True
        )

        result = await get_content_structure(request)

        print(f"V Total files: {result.total_files}")
        print(f"V Total directories: {result.total_directories}")
        print(f"V Root directory: {result.structure.name}")
        print(f"V Root path: {result.structure.path}")

        # Show the directory structure (first few levels)
        def print_structure(node, depth=0):
            indent = "  " * depth
            node_type = "[DIR]" if node.type == "directory" else "[FILE]"
            print(f"{indent}{node_type} {node.name} ({node.path})")

            # Only print first few children to avoid too much output
            for child in node.children[:5]:  # Limit to first 5 children
                print_structure(child, depth + 1)

            if len(node.children) > 5:
                print(f"{indent}  ... and {len(node.children) - 5} more items")

        print(f"\nContent structure (first few items):")
        print_structure(result.structure)

        return True

    except Exception as e:
        print(f"X Error in content structure: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    print("Content Loader (Phase 2) - Testing")
    print("=" * 50)
    print()

    # Test content loading
    loading_success = await test_content_loading()
    print()

    # Test content search
    search_success = await test_content_search()
    print()

    # Test content structure
    structure_success = await test_content_structure()
    print()

    print("=" * 50)
    if loading_success and search_success and structure_success:
        print("All content loader tests passed!")
        print("\nThe content loader is working correctly with features:")
        print("- Content loading without embeddings")
        print("- Text chunking with overlap")
        print("- Content search using keyword matching")
        print("- Directory structure analysis")
        return True
    else:
        print("Some tests failed. Please check the error messages above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)