
"""
Content Loader for Phase 2: Loading textbook content without embeddings
This module provides functionality to load, parse, and structure textbook content
for the RAG system without generating embeddings.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ...utils.logger import logger
import os
import re
from pathlib import Path
import asyncio
import hashlib


router = APIRouter()


class ContentChunk(BaseModel):
    """Represents a chunk of content with metadata"""
    id: str
    title: str
    content: str
    source_file: str
    chapter_id: str
    section_level: int
    word_count: int
    char_count: int
    metadata: Dict[str, Any] = {}


class ContentLoadRequest(BaseModel):
    """Request to load content from a source"""
    source_path: str
    recursive: bool = True
    format_filter: List[str] = [".md"]
    chunk_size: int = 1000  # Maximum characters per chunk
    overlap_size: int = 100  # Overlap between chunks


class ContentLoadResponse(BaseModel):
    """Response with loaded content"""
    loaded_files: int
    total_chunks: int
    total_words: int
    chunks: List[ContentChunk]
    processing_stats: Dict[str, Any]


def calculate_chunk_id(content: str, source_file: str, position: int) -> str:
    """Generate a unique ID for a content chunk"""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
    return f"{file_hash}_{position}_{content_hash}"


def chunk_text(text: str, chunk_size: int, overlap_size: int) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [{"content": text, "start_pos": 0, "end_pos": len(text)}]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            search_start = max(start, end - 200)  # Look back up to 200 chars
            sentence_break = -1

            for punct in ['.', '!', '?', '\n']:
                last_punct = text.rfind(punct, search_start, end)
                if last_punct > sentence_break:
                    sentence_break = last_punct

            # If we found a good break point, use it
            if sentence_break > start + chunk_size // 2:  # Don't break too early
                end = sentence_break + 1

        chunk_content = text[start:end]
        chunks.append({
            "content": chunk_content,
            "start_pos": start,
            "end_pos": end
        })

        # Move start position with overlap
        start = end - overlap_size if overlap_size < end else end

        # Safety check to prevent infinite loop
        if start >= len(text):
            break

    return chunks


def split_markdown_by_headers(content: str) -> List[Dict[str, str]]:
    """Split markdown content by headers to preserve document structure"""
    sections = []

    # Split by H1, H2, H3 headers
    header_pattern = r'^(#{1,3})\s+(.+)$'
    lines = content.split('\n')

    current_section = {"header": "Unstructured Content", "content": "", "level": 3}
    current_header_level = 3

    for line in lines:
        header_match = re.match(header_pattern, line.strip())

        if header_match:
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append({
                    "title": current_section["header"],
                    "content": current_section["content"].strip(),
                    "level": current_header_level
                })

            # Start new section
            hashes = header_match.group(1)
            header_text = header_match.group(2)
            header_level = len(hashes)

            current_section = {
                "header": header_text,
                "content": "",
                "level": header_level
            }
            current_header_level = header_level
        else:
            current_section["content"] += line + "\n"

    # Add the last section
    if current_section["content"].strip():
        sections.append({
            "title": current_section["header"],
            "content": current_section["content"].strip(),
            "level": current_header_level
        })

    return sections


@router.post("/load", response_model=ContentLoadResponse)
async def load_content(request: ContentLoadRequest):
    """
    Load textbook content without generating embeddings
    """
    try:
        logger.info(f"Loading content from: {request.source_path}")

        source_path = Path(request.source_path)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source path does not exist: {request.source_path}")

        loaded_files = 0
        total_chunks = 0
        total_words = 0
        all_chunks = []

        # Get all markdown files
        files_to_process = []
        if request.recursive:
            for ext in request.format_filter:
                files_to_process.extend(list(source_path.rglob(f"*{ext}")))
        else:
            for ext in request.format_filter:
                files_to_process.extend(list(source_path.glob(f"*{ext}")))

        processing_stats = {
            "files_by_type": {},
            "avg_chunk_size": 0,
            "total_chars": 0
        }

        for file_path in files_to_process:
            try:
                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Determine chapter ID from file path
                relative_path = str(file_path.relative_to(source_path))
                chapter_id = relative_path.split('/')[0] if '/' in relative_path else 'general'

                # Split content by headers to preserve structure
                header_sections = split_markdown_by_headers(content)

                for section in header_sections:
                    section_content = section["content"]
                    section_title = section["title"]

                    if section_content.strip():
                        # Chunk the section content
                        text_chunks = chunk_text(
                            section_content,
                            request.chunk_size,
                            request.overlap_size
                        )

                        for i, chunk_data in enumerate(text_chunks):
                            chunk_content = chunk_data["content"]

                            if chunk_content.strip():  # Only process non-empty chunks
                                word_count = len(chunk_content.split())
                                char_count = len(chunk_content)

                                chunk = ContentChunk(
                                    id=calculate_chunk_id(chunk_content, str(file_path), len(all_chunks)),
                                    title=f"{section_title} - Chunk {i+1}",
                                    content=chunk_content,
                                    source_file=str(file_path),
                                    chapter_id=chapter_id,
                                    section_level=section["level"],
                                    word_count=word_count,
                                    char_count=char_count,
                                    metadata={
                                        "file_path": str(file_path),
                                        "relative_path": relative_path,
                                        "section_title": section_title,
                                        "chunk_index": i,
                                        "total_chunks_in_section": len(text_chunks),
                                        "start_pos": chunk_data["start_pos"],
                                        "end_pos": chunk_data["end_pos"]
                                    }
                                )

                                all_chunks.append(chunk)
                                total_chunks += 1
                                total_words += word_count

                loaded_files += 1

                # Update stats
                ext = file_path.suffix
                if ext in processing_stats["files_by_type"]:
                    processing_stats["files_by_type"][ext] += 1
                else:
                    processing_stats["files_by_type"][ext] = 1

                logger.info(f"Loaded content from: {file_path}")

            except Exception as file_error:
                logger.error(f"Error processing file {file_path}: {str(file_error)}")
                continue  # Continue with other files even if one fails

        # Calculate processing stats
        if all_chunks:
            processing_stats["avg_chunk_size"] = sum(c.char_count for c in all_chunks) / len(all_chunks)
        processing_stats["total_chars"] = sum(c.char_count for c in all_chunks)

        logger.info(f"Content loading completed. Loaded {loaded_files} files, {total_chunks} chunks, {total_words} words.")

        return ContentLoadResponse(
            loaded_files=loaded_files,
            total_chunks=total_chunks,
            total_words=total_words,
            chunks=all_chunks,
            processing_stats=processing_stats
        )

    except Exception as e:
        logger.error(f"Error in content loading: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content loading failed: {str(e)}")


class ContentSearchRequest(BaseModel):
    """Request to search loaded content"""
    query: str
    source_path: Optional[str] = None
    chapter_filter: Optional[str] = None
    limit: int = 10


class ContentSearchResult(BaseModel):
    """Result of content search"""
    chunk: ContentChunk
    relevance_score: float
    matched_terms: List[str]


class ContentSearchResponse(BaseModel):
    """Response for content search"""
    results: List[ContentSearchResult]
    total_results: int
    search_query: str


@router.post("/search", response_model=ContentSearchResponse)
async def search_content(request: ContentSearchRequest):
    """
    Search through loaded content using simple keyword matching
    (This is a basic implementation without embeddings)
    """
    try:
        logger.info(f"Searching content for query: {request.query}")

        # For this basic implementation, we'll simulate search by loading content
        # In a real implementation, this would search through already-loaded content in memory or DB
        load_request = ContentLoadRequest(
            source_path=request.source_path or "docs",
            recursive=True,
            chunk_size=1000,
            overlap_size=100
        )

        # Load content to search through
        content_response = await load_content(load_request)

        # Filter chunks based on chapter if specified
        chunks_to_search = content_response.chunks
        if request.chapter_filter:
            chunks_to_search = [c for c in chunks_to_search if request.chapter_filter.lower() in c.chapter_id.lower()]

        # Simple keyword matching (in a real system, this would use more sophisticated methods)
        query_terms = request.query.lower().split()
        results = []

        for chunk in chunks_to_search:
            content_lower = chunk.content.lower()
            score = 0
            matched_terms = []

            for term in query_terms:
                if term in content_lower:
                    score += 1
                    matched_terms.append(term)

            if score > 0:
                # Calculate a more sophisticated score based on term frequency and position
                term_freq_score = score / len(query_terms)  # How many query terms were matched
                content_length_score = min(1.0, len(chunk.content) / 1000)  # Prefer reasonably sized chunks
                position_score = 1.0  # In this simple version, all positions are equal

                final_score = term_freq_score * content_length_score * position_score
                results.append(ContentSearchResult(
                    chunk=chunk,
                    relevance_score=final_score,
                    matched_terms=matched_terms
                ))

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        results = results[:request.limit]

        logger.info(f"Search completed. Found {len(results)} results for query: {request.query}")

        return ContentSearchResponse(
            results=results,
            total_results=len(results),
            search_query=request.query
        )

    except Exception as e:
        logger.error(f"Error in content search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content search failed: {str(e)}")


class ContentStructureRequest(BaseModel):
    """Request to get the structure of loaded content"""
    source_path: str
    recursive: bool = True


class ContentDirectory(BaseModel):
    """Represents a directory in the content structure"""
    name: str
    path: str
    type: str  # "directory" or "file"
    children: List['ContentDirectory'] = []
    metadata: Optional[Dict[str, Any]] = None


class ContentStructureResponse(BaseModel):
    """Response with content structure"""
    structure: ContentDirectory
    total_files: int
    total_directories: int


@router.post("/structure", response_model=ContentStructureResponse)
async def get_content_structure(request: ContentStructureRequest):
    """
    Get the directory structure of the content
    """
    try:
        logger.info(f"Getting content structure from: {request.source_path}")

        source_path = Path(request.source_path)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source path does not exist: {request.source_path}")

        def build_structure(current_path: Path) -> ContentDirectory:
            """Recursively build the content structure"""
            is_file = current_path.is_file()
            item_type = "file" if is_file else "directory"

            # Get the name (last part of path)
            name = current_path.name
            path_str = str(current_path.relative_to(source_path.parent if current_path == source_path and current_path.is_file() else source_path))

            # For files, include metadata
            metadata = None
            if is_file:
                metadata = {
                    "size": current_path.stat().st_size,
                    "extension": current_path.suffix,
                    "modified": current_path.stat().st_mtime
                }

            node = ContentDirectory(
                name=name,
                path=path_str,
                type=item_type,
                metadata=metadata
            )

            # If it's a directory, process its contents
            if current_path.is_dir() and request.recursive:
                for child in sorted(current_path.iterdir()):
                    # Skip hidden files and directories
                    if child.name.startswith('.'):
                        continue
                    node.children.append(build_structure(child))

            return node

        structure = build_structure(source_path)

        # Count files and directories
        def count_items(node: ContentDirectory) -> tuple[int, int]:
            """Count files and directories recursively"""
            files = 0
            dirs = 0

            if node.type == "file":
                files = 1
            else:
                dirs = 1

            for child in node.children:
                child_files, child_dirs = count_items(child)
                files += child_files
                dirs += child_dirs

            return files, dirs

        total_files, total_directories = count_items(structure)

        logger.info(f"Content structure retrieved. {total_files} files, {total_directories} directories.")

        return ContentStructureResponse(
            structure=structure,
            total_files=total_files,
            total_directories=total_directories - 1  # Subtract 1 to not count the root
        )

    except Exception as e:
        logger.error(f"Error getting content structure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content structure retrieval failed: {str(e)}")


# Update the type annotation for the recursive model
ContentDirectory.update_forward_refs()