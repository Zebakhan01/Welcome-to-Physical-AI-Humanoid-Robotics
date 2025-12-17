"""
RAG Content Chunker Module
This module handles splitting loaded documents into chunks for embedding
"""
from typing import List, Dict, Any
import re

class ContentChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a single document into chunks
        """
        content = document["content"]
        chunks = []

        # Split content into chunks
        sentences = self._split_into_sentences(content)

        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(current_chunk, document)
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Get the end of current chunk for overlap
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size + 1  # +1 for space

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = self._create_chunk(current_chunk, document)
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        """
        # Split on sentence endings followed by whitespace and capital letter
        sentences = re.split(r'[.!?]+\s+', text)
        # Remove empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _create_chunk(self, content: str, original_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a chunk with metadata from the original document
        """
        return {
            "id": f"{original_document['id']}_chunk_{len(content[:50])}",  # Simple ID
            "content": content,
            "metadata": {
                **original_document["metadata"],
                "source_document_id": original_document["id"],
                "source_document_title": original_document["title"],
                "chunk_size": len(content),
                "original_doc_path": original_document["metadata"]["relative_path"]
            }
        }

def chunk_content(documents: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk documents
    """
    chunker = ContentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_documents(documents)

if __name__ == "__main__":
    # Example usage
    sample_docs = [
        {
            "id": "test_doc_1",
            "title": "Test Document",
            "content": "This is a sample document. It has multiple sentences. Each sentence should be processed properly. The content will be split into chunks based on the specified size.",
            "metadata": {"source": "test", "file_path": "test.md"}
        }
    ]

    chunker = ContentChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk_documents(sample_docs)

    print(f"Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk['content'])} chars - {chunk['content'][:50]}...")