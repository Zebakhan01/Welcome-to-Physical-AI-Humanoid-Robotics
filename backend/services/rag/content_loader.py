"""
Content Loader Service for Phase 2
Loads Docusaurus markdown files, strips MDX/JSX/React syntax, normalizes text,
chunks content deterministically with metadata
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
from pydantic import BaseModel
from ...utils.logger import logger


class ContentChunk(BaseModel):
    """Represents a chunk of content with metadata"""
    id: str
    title: str
    content: str
    source_file: str
    chapter: str
    section: str
    source_file_path: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = {}


class ContentLoaderConfig(BaseModel):
    """Configuration for content loading"""
    source_path: str
    recursive: bool = True
    format_filter: List[str] = [".md", ".mdx"]
    chunk_size: int = 1000
    overlap_size: int = 100
    strip_mdx: bool = True
    strip_jsx: bool = True


class ContentLoaderService:
    """
    Service class for loading and preprocessing textbook content
    """

    def __init__(self):
        """Initialize the content loader service"""
        logger.info("✅ Content Loader Service initialized")

    def strip_mdx_jsx_syntax(self, content: str) -> str:
        """
        Remove MDX and JSX syntax from markdown content

        Args:
            content: Raw markdown content with MDX/JSX

        Returns:
            Clean markdown content without MDX/JSX syntax
        """
        # Remove JSX-style imports and exports
        content = re.sub(r'import\s+.*?from\s+[\'\"].*?[\'\"];?', '', content)
        content = re.sub(r'export\s+{.*?};?', '', content)
        content = re.sub(r'export\s+default\s+.*?;', '', content)

        # Remove JSX components and tags
        content = re.sub(r'<\s*[^>]*\s*>', '', content)

        # Remove JSX expressions in curly braces (be careful not to remove regular markdown)
        # This pattern looks for {some_content} but tries to avoid removing things like {#id} or regular markdown
        content = re.sub(r'\{[^{}]*\}', '', content)

        # Remove MDX-specific syntax like import/export statements and component usage
        content = re.sub(r'```jsx\n.*?\n```', '', content, flags=re.DOTALL)
        content = re.sub(r'```js\n.*?\n```', '', content, flags=re.DOTALL)

        # Remove JSX comments
        content = re.sub(r'{/\*.*?\*/}', '', content, flags=re.DOTALL)

        # Clean up multiple newlines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

        return content.strip()

    def extract_chapter_section(self, file_path: str) -> tuple[str, str]:
        """
        Extract chapter and section from file path

        Args:
            file_path: Path to the markdown file

        Returns:
            Tuple of (chapter, section)
        """
        path_obj = Path(file_path)
        parent_dir = path_obj.parent.name
        file_name = path_obj.stem

        # Use parent directory as chapter and file name as section
        chapter = parent_dir if parent_dir != '.' else 'general'
        section = file_name

        return chapter, section

    def chunk_text(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """
        Split text into overlapping chunks with respect to sentence boundaries

        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap_size: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

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
                    if last_punct > sentence_break and last_punct > start + chunk_size // 2:
                        sentence_break = last_punct + 1  # Include the punctuation

                # If we found a good break point, use it
                if sentence_break > start + chunk_size // 2:
                    end = sentence_break

            chunk_content = text[start:end].strip()
            if chunk_content:  # Only add non-empty chunks
                chunks.append(chunk_content)

            # Move start position with overlap
            start = end - overlap_size if overlap_size < end else end

            # Safety check to prevent infinite loop
            if start >= len(text):
                break

        return chunks

    def split_by_headers(self, content: str) -> List[Dict[str, str]]:
        """
        Split content by markdown headers to preserve document structure

        Args:
            content: Markdown content to split

        Returns:
            List of sections with title and content
        """
        sections = []

        # Split by H1, H2, H3 headers
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = content.split('\n')

        current_section = {"title": "General Content", "content": "", "level": 3}
        current_header_level = 3

        for line in lines:
            header_match = re.match(header_pattern, line.strip())

            if header_match:
                # Save previous section if it has content
                if current_section["content"].strip():
                    sections.append({
                        "title": current_section["title"],
                        "content": current_section["content"].strip(),
                        "level": current_header_level
                    })

                # Start new section
                hashes = header_match.group(1)
                header_text = header_match.group(2)
                header_level = len(hashes)

                current_section = {
                    "title": header_text,
                    "content": "",
                    "level": header_level
                }
                current_header_level = header_level
            else:
                current_section["content"] += line + "\n"

        # Add the last section
        if current_section["content"].strip():
            sections.append({
                "title": current_section["title"],
                "content": current_section["content"].strip(),
                "level": current_header_level
            })

        return sections

    async def load_content(self, config: ContentLoaderConfig) -> List[ContentChunk]:
        """
        Load and process content from the specified source path

        Args:
            config: Content loading configuration

        Returns:
            List of content chunks with metadata
        """
        try:
            source_path = Path(config.source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {config.source_path}")

            all_chunks = []
            chunk_index = 0

            # Get all markdown files
            files_to_process = []
            if config.recursive:
                for ext in config.format_filter:
                    files_to_process.extend(list(source_path.rglob(f"*{ext}")))
            else:
                for ext in config.format_filter:
                    files_to_process.extend(list(source_path.glob(f"*{ext}")))

            logger.info(f"Found {len(files_to_process)} files to process")

            for file_path in files_to_process:
                try:
                    # Read the content of the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Strip MDX/JSX syntax if configured
                    if config.strip_mdx or config.strip_jsx:
                        content = self.strip_mdx_jsx_syntax(content)

                    # Extract chapter and section from file path
                    chapter, section = self.extract_chapter_section(str(file_path))

                    # Split content by headers to preserve structure
                    header_sections = self.split_by_headers(content)

                    for section_idx, section_data in enumerate(header_sections):
                        section_content = section_data["content"]
                        section_title = section_data["title"]

                        if section_content.strip():
                            # Chunk the section content
                            text_chunks = self.chunk_text(
                                section_content,
                                config.chunk_size,
                                config.overlap_size
                            )

                            for chunk_idx, chunk_content in enumerate(text_chunks):
                                if chunk_content.strip():  # Only process non-empty chunks
                                    # Generate unique ID for the chunk
                                    content_hash = hashlib.md5(chunk_content.encode()).hexdigest()[:8]
                                    file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
                                    chunk_id = f"{file_hash}_{chunk_idx}_{content_hash}"

                                    chunk = ContentChunk(
                                        id=chunk_id,
                                        title=f"{section_title}",
                                        content=chunk_content,
                                        source_file=file_path.name,
                                        chapter=chapter,
                                        section=section,
                                        source_file_path=str(file_path),
                                        chunk_index=chunk_idx,
                                        total_chunks=len(text_chunks),
                                        metadata={
                                            "file_path": str(file_path),
                                            "relative_path": str(file_path.relative_to(source_path)),
                                            "section_title": section_title,
                                            "section_level": section_data["level"],
                                            "chunk_index_in_section": chunk_idx,
                                            "total_chunks_in_section": len(text_chunks),
                                            "original_word_count": len(chunk_content.split()),
                                            "original_char_count": len(chunk_content),
                                        }
                                    )

                                    all_chunks.append(chunk)
                                    chunk_index += 1

                    logger.info(f"Processed content from: {file_path}")

                except Exception as file_error:
                    logger.error(f"Error processing file {file_path}: {str(file_error)}")
                    continue  # Continue with other files even if one fails

            logger.info(f"Content loading completed. Generated {len(all_chunks)} chunks from {len(files_to_process)} files.")
            return all_chunks

        except Exception as e:
            logger.error(f"Error in content loading: {str(e)}")
            raise

    async def get_content_structure(self, source_path: str, recursive: bool = True) -> Dict[str, Any]:
        """
        Get the directory structure of the content

        Args:
            source_path: Path to the content directory
            recursive: Whether to scan recursively

        Returns:
            Dictionary representing the content structure
        """
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {source_path}")

            def build_structure(current_path: Path) -> Dict[str, Any]:
                """Recursively build the content structure"""
                is_file = current_path.is_file()
                item_type = "file" if is_file else "directory"

                # Get the name (last part of path)
                name = current_path.name
                path_str = str(current_path.relative_to(source_path))

                # For files, include metadata
                metadata = None
                if is_file:
                    metadata = {
                        "size": current_path.stat().st_size,
                        "extension": current_path.suffix,
                        "modified": current_path.stat().st_mtime
                    }

                node = {
                    "name": name,
                    "path": path_str,
                    "type": item_type,
                    "metadata": metadata
                }

                # If it's a directory and recursive, process its contents
                if current_path.is_dir() and recursive:
                    children = []
                    for child in sorted(current_path.iterdir()):
                        # Skip hidden files and directories
                        if child.name.startswith('.'):
                            continue
                        children.append(build_structure(child))
                    node["children"] = children

                return node

            structure = build_structure(source_path)
            logger.info(f"Content structure retrieved from: {source_path}")
            return structure

        except Exception as e:
            logger.error(f"Error getting content structure: {str(e)}")
            raise