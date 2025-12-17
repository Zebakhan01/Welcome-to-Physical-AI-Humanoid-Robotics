"""
RAG Document Loader Module
This module handles document loading and preprocessing
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from .chunker import ContentChunker
from .preparer import prepare_chunks_for_embeddings

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, book_content_path: str = "../shared/book_content", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.book_content_path = Path(book_content_path)
        self.chunker = ContentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_process_and_prepare_documents(self) -> List[Dict[str, Any]]:
        """
        Load all markdown documents, split them into chunks, and prepare for embeddings
        Returns a list of prepared document chunks ready for embeddings
        """
        documents = self.load_all_documents()
        chunks = self.chunker.chunk_documents(documents)
        prepared_chunks = prepare_chunks_for_embeddings(chunks)
        logger.info(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks, and prepared {len(prepared_chunks)} chunks for embeddings")
        return prepared_chunks

    def load_and_chunk_documents(self) -> List[Dict[str, Any]]:
        """
        Load all markdown documents and split them into chunks
        Returns a list of document chunks with metadata
        """
        documents = self.load_all_documents()
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks")
        return chunks

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        Load all markdown documents from the book_content directory
        Returns a list of documents with metadata
        """
        documents = []

        if not self.book_content_path.exists():
            logger.warning(f"Book content path does not exist: {self.book_content_path}")
            return documents

        # Walk through all markdown files
        for md_file in self.book_content_path.rglob("*.md"):
            try:
                doc = self._load_single_document(md_file)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading document {md_file}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _load_single_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a single markdown document, removing any JSX-like content safely
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove any JSX-like content (React components, etc.)
        # This regex removes content between curly braces that looks like JSX
        content = self._remove_jsx_like_content(content)

        # Extract title from the first H1 if available
        title = self._extract_title(content)

        # Clean up the content
        clean_content = self._clean_content(content)

        return {
            "id": str(file_path.relative_to(self.book_content_path)),
            "title": title,
            "content": clean_content,
            "source": str(file_path),
            "metadata": {
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.book_content_path)),
                "file_size": file_path.stat().st_size
            }
        }

    def _remove_jsx_like_content(self, content: str) -> str:
        """
        Remove JSX-like content from markdown files safely
        """
        # Remove any content that looks like JSX components
        # This removes content like {<Component />} or {variable}
        # but preserves regular text in curly braces like {some text}

        # First, handle code blocks to preserve them
        code_blocks = []
        def replace_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks)-1}__"

        # Extract code blocks to preserve them during JSX removal
        code_pattern = r'```.*?```|`[^`]*`'
        content = re.sub(code_pattern, replace_code_block, content, flags=re.DOTALL)

        # Remove JSX-like content (content between braces that looks like components)
        jsx_pattern = r'\{[^}]*<[a-zA-Z][^>]*>[^}]*\}'
        content = re.sub(jsx_pattern, '', content)

        # Replace code blocks back
        for i, code_block in enumerate(code_blocks):
            content = content.replace(f"__CODE_BLOCK_{i}__", code_block)

        return content

    def _extract_title(self, content: str) -> str:
        """
        Extract the first H1 title from markdown content
        """
        # Look for the first H1 title
        h1_pattern = r'^#\s+(.+)$'
        match = re.search(h1_pattern, content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled Document"

    def _clean_content(self, content: str) -> str:
        """
        Clean up markdown content by removing metadata frontmatter and extra whitespace
        """
        # Remove YAML frontmatter if present
        if content.startswith('---'):
            end_frontmatter = content.find('---', 3)
            if end_frontmatter != -1:
                content = content[end_frontmatter + 3:].strip()

        # Remove extra whitespace and normalize line breaks
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()

        return content

def load_process_and_prepare_documents() -> List[Dict[str, Any]]:
    """
    Convenience function to load, process, and prepare all documents for embeddings
    """
    loader = DocumentLoader()
    return loader.load_process_and_prepare_documents()

if __name__ == "__main__":
    # Test the loader, chunker, and preparer
    loader = DocumentLoader()
    prepared_chunks = loader.load_process_and_prepare_documents()
    print(f"Loaded, processed, and prepared {len(prepared_chunks)} document chunks for embeddings")
    if prepared_chunks:
        print(f"First chunk ID: {prepared_chunks[0]['id']}")
        print(f"First chunk source: {prepared_chunks[0]['metadata']['source_document_title']}")
        print(f"First chunk preview: {prepared_chunks[0]['text'][:100]}...")