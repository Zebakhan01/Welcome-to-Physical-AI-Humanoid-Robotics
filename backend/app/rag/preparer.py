"""
RAG Content Preparer Module
This module prepares chunks for embeddings (without actually embedding)
"""
from typing import List, Dict, Any
import hashlib

class ContentPreparer:
    def __init__(self):
        pass

    def prepare_for_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare chunks for embedding by adding additional metadata and formatting
        """
        prepared_chunks = []

        for i, chunk in enumerate(chunks):
            # Create a unique ID for the chunk
            chunk_id = self._generate_chunk_id(chunk, i)

            prepared_chunk = {
                "id": chunk_id,
                "text": chunk["content"],
                "metadata": {
                    **chunk["metadata"],
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }

            prepared_chunks.append(prepared_chunk)

        return prepared_chunks

    def _generate_chunk_id(self, chunk: Dict[str, Any], index: int) -> str:
        """
        Generate a unique ID for a chunk
        """
        # Create a unique identifier based on content and source
        content_prefix = chunk["content"][:50].replace(" ", "_").replace("\n", "_")
        source_path = chunk["metadata"]["relative_path"].replace("/", "_").replace("\\", "_")

        # Use hash to ensure uniqueness
        content_hash = hashlib.md5(chunk["content"].encode('utf-8')).hexdigest()[:8]

        return f"{source_path}_{content_hash}_{index}"

def prepare_chunks_for_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to prepare chunks for embeddings
    """
    preparer = ContentPreparer()
    return preparer.prepare_for_embeddings(chunks)

if __name__ == "__main__":
    # Example usage
    sample_chunks = [
        {
            "content": "This is a sample chunk of content that will be prepared for embeddings.",
            "metadata": {
                "source_document_title": "Sample Document",
                "relative_path": "sample.md",
                "chunk_size": 65
            }
        }
    ]

    prepared = prepare_chunks_for_embeddings(sample_chunks)
    print(f"Prepared {len(prepared)} chunks for embeddings")
    if prepared:
        print(f"Prepared chunk ID: {prepared[0]['id']}")
        print(f"Prepared chunk text: {prepared[0]['text']}")
        print(f"Prepared metadata: {prepared[0]['metadata']}")