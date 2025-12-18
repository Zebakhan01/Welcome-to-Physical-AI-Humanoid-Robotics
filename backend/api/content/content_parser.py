from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ...utils.logger import logger
import os
import re
from pathlib import Path
import asyncio

router = APIRouter()

class ContentParseRequest(BaseModel):
    content: str
    title: str
    chapter_id: str
    format: str = "markdown"  # markdown, html, text

class ContentParseResponse(BaseModel):
    title: str
    chapter_id: str
    sections: List[Dict[str, Any]]
    word_count: int
    reading_time: int  # in minutes

def split_markdown_content(content: str) -> List[Dict[str, Any]]:
    """
    Split markdown content into sections based on headers
    """
    # Split content by markdown headers (##, ###, etc.)
    sections = []

    # Split by H2 headers (##)
    h2_parts = re.split(r'\n(?=##\s)', content)

    for part in h2_parts:
        if part.strip():
            # Extract header title
            header_match = re.match(r'##\s+(.+?)(?:\n|$)', part.strip())
            if header_match:
                title = header_match.group(1).strip()
                # Remove the header line from content
                content_part = re.sub(r'^##\s+.+\n?', '', part, count=1).strip()
            else:
                # If no H2 header, treat as main content
                title = "Main Content"
                content_part = part.strip()

            if content_part:
                sections.append({
                    "title": title,
                    "content": content_part,
                    "level": 2,
                    "type": "section"
                })

    # If no sections were created, create one with the full content
    if not sections:
        sections = [{
            "title": "Full Content",
            "content": content,
            "level": 1,
            "type": "section"
        }]

    return sections

@router.post("/parse", response_model=ContentParseResponse)
async def parse_content(request: ContentParseRequest):
    """
    Parse textbook content and extract structured information
    """
    try:
        logger.info(f"Content parsing requested for: {request.title}")

        # Split content into sections based on the format
        if request.format.lower() == "markdown":
            sections = split_markdown_content(request.content)
        else:
            # For other formats, create a single section
            sections = [{
                "title": request.title,
                "content": request.content,
                "level": 1,
                "type": "section"
            }]

        # Calculate word count and reading time
        words = len(request.content.split())
        reading_time = max(1, words // 200)  # Assuming 200 words per minute

        return ContentParseResponse(
            title=request.title,
            chapter_id=request.chapter_id,
            sections=sections,
            word_count=words,
            reading_time=reading_time
        )
    except Exception as e:
        logger.error(f"Error in content parsing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content parsing failed: {str(e)}")

class ContentIndexRequest(BaseModel):
    source_path: str
    recursive: bool = True

class ContentIndexResponse(BaseModel):
    indexed_files: int
    processed_content: List[Dict[str, Any]]

@router.post("/index", response_model=ContentIndexResponse)
async def index_content(request: ContentIndexRequest):
    """
    Index textbook content from a directory
    """
    try:
        logger.info(f"Content indexing requested for: {request.source_path}")

        # Import the RAG services to store content
        from ..rag.retrieval_service import store_content, VectorStoreRequest

        source_path = Path(request.source_path)
        if not source_path.exists():
            raise HTTPException(status_code=404, detail=f"Source path does not exist: {request.source_path}")

        indexed_files = 0
        processed_content = []

        # Get all markdown files
        if request.recursive:
            md_files = list(source_path.rglob("*.md"))
        else:
            md_files = list(source_path.glob("*.md"))

        for md_file in md_files:
            try:
                # Read the content of the markdown file
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Create a title from the file path
                title = str(md_file.relative_to(source_path)).replace('\\', '/')

                # Parse the content into sections
                parse_request = ContentParseRequest(
                    content=content,
                    title=title,
                    chapter_id=title.split('/')[0] if '/' in title else 'general',
                    format="markdown"
                )

                parse_response = await parse_content(parse_request)

                # Store each section in the vector store
                for section in parse_response.sections:
                    section_content = section['content']
                    section_title = f"{title} - {section['title']}"

                    if section_content.strip():  # Only store non-empty content
                        store_request = VectorStoreRequest(
                            content=section_content,
                            title=section_title,
                            chapter_id=parse_request.chapter_id,
                            metadata={
                                "source_file": str(md_file),
                                "section_title": section['title'],
                                "format": "markdown",
                                "word_count": len(section_content.split())
                            }
                        )

                        # Store the content in the vector database
                        store_result = await store_content(store_request)

                        processed_item = {
                            "file": str(md_file),
                            "section": section['title'],
                            "vector_id": store_result["vector_id"],
                            "status": store_result["status"]
                        }
                        processed_content.append(processed_item)

                indexed_files += 1
                logger.info(f"Indexed file: {md_file}")

            except Exception as file_error:
                logger.error(f"Error processing file {md_file}: {str(file_error)}")
                continue  # Continue with other files even if one fails

        logger.info(f"Content indexing completed. Indexed {indexed_files} files.")

        return ContentIndexResponse(
            indexed_files=indexed_files,
            processed_content=processed_content
        )
    except Exception as e:
        logger.error(f"Error in content indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Content indexing failed: {str(e)}")