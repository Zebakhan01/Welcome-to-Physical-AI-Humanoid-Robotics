"""
Neon PostgreSQL Database Service for Phase 2
Handles document metadata and chunk references using Neon Serverless PostgreSQL
"""
import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from ...utils.logger import logger


Base = declarative_base()


class DocumentModel(Base):
    """SQLAlchemy model for document metadata"""
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    source_file_path = Column(String, nullable=False)
    chapter = Column(String, nullable=True)
    section = Column(String, nullable=True)
    total_chunks = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    status = Column(String, default="indexed")  # indexed, processing, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default={})


class ChunkModel(Base):
    """SQLAlchemy model for chunk metadata"""
    __tablename__ = "chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    total_chunks = Column(Integer, nullable=False)
    vector_id = Column(String, nullable=True)  # Reference to vector store ID
    word_count = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_json = Column(JSON, default={})


class DocumentMetadata(BaseModel):
    """Pydantic model for document metadata"""
    id: str
    title: str
    source_file_path: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    total_chunks: int
    word_count: int
    char_count: int
    status: str = "indexed"
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}


class ChunkMetadata(BaseModel):
    """Pydantic model for chunk metadata"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    total_chunks: int
    vector_id: Optional[str] = None
    word_count: int
    char_count: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = {}


class NeonDatabaseService:
    """
    Service class for managing document metadata using Neon PostgreSQL
    """

    def __init__(self):
        """Initialize the Neon database service"""
        database_url = os.getenv("NEON_DATABASE_URL")
        if not database_url:
            raise ValueError("NEON_DATABASE_URL environment variable is required")

        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False  # Set to True for debugging SQL queries
        )

        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("✅ Neon Database Service initialized")

    def get_db_session(self):
        """Get a database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    async def store_document_metadata(self, doc_metadata: DocumentMetadata) -> bool:
        """
        Store document metadata in the database

        Args:
            doc_metadata: Document metadata to store

        Returns:
            True if storage was successful
        """
        try:
            # Convert Pydantic model to SQLAlchemy model
            db_doc = DocumentModel(
                id=doc_metadata.id,
                title=doc_metadata.title,
                source_file_path=doc_metadata.source_file_path,
                chapter=doc_metadata.chapter,
                section=doc_metadata.section,
                total_chunks=doc_metadata.total_chunks,
                word_count=doc_metadata.word_count,
                char_count=doc_metadata.char_count,
                status=doc_metadata.status,
                metadata_json=doc_metadata.metadata
            )

            # Use asyncio to run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_store_document_metadata,
                db_doc
            )

            logger.info(f"Stored document metadata: {doc_metadata.id}")
            return True

        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            raise

    def _sync_store_document_metadata(self, db_doc: DocumentModel):
        """Synchronous helper for storing document metadata"""
        db = next(self.get_db_session())
        try:
            # Check if document already exists
            existing_doc = db.query(DocumentModel).filter(DocumentModel.id == db_doc.id).first()
            if existing_doc:
                # Update existing document
                existing_doc.title = db_doc.title
                existing_doc.source_file_path = db_doc.source_file_path
                existing_doc.chapter = db_doc.chapter
                existing_doc.section = db_doc.section
                existing_doc.total_chunks = db_doc.total_chunks
                existing_doc.word_count = db_doc.word_count
                existing_doc.char_count = db_doc.char_count
                existing_doc.status = db_doc.status
                existing_doc.metadata_json = db_doc.metadata_json
                existing_doc.updated_at = datetime.utcnow()
            else:
                # Add new document
                db.add(db_doc)
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    async def store_chunk_metadata(self, chunk_metadata: ChunkMetadata) -> bool:
        """
        Store chunk metadata in the database

        Args:
            chunk_metadata: Chunk metadata to store

        Returns:
            True if storage was successful
        """
        try:
            # Convert Pydantic model to SQLAlchemy model
            db_chunk = ChunkModel(
                id=chunk_metadata.id,
                document_id=chunk_metadata.document_id,
                content=chunk_metadata.content,
                chunk_index=chunk_metadata.chunk_index,
                total_chunks=chunk_metadata.total_chunks,
                vector_id=chunk_metadata.vector_id,
                word_count=chunk_metadata.word_count,
                char_count=chunk_metadata.char_count,
                metadata_json=chunk_metadata.metadata
            )

            # Use asyncio to run the database operation in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_store_chunk_metadata,
                db_chunk
            )

            logger.info(f"Stored chunk metadata: {chunk_metadata.id}")
            return True

        except Exception as e:
            logger.error(f"Error storing chunk metadata: {str(e)}")
            raise

    def _sync_store_chunk_metadata(self, db_chunk: ChunkModel):
        """Synchronous helper for storing chunk metadata"""
        db = next(self.get_db_session())
        try:
            # Check if chunk already exists
            existing_chunk = db.query(ChunkModel).filter(ChunkModel.id == db_chunk.id).first()
            if existing_chunk:
                # Update existing chunk
                existing_chunk.document_id = db_chunk.document_id
                existing_chunk.content = db_chunk.content
                existing_chunk.chunk_index = db_chunk.chunk_index
                existing_chunk.total_chunks = db_chunk.total_chunks
                existing_chunk.vector_id = db_chunk.vector_id
                existing_chunk.word_count = db_chunk.word_count
                existing_chunk.char_count = db_chunk.char_count
                existing_chunk.metadata_json = db_chunk.metadata_json
                existing_chunk.updated_at = datetime.utcnow()
            else:
                # Add new chunk
                db.add(db_chunk)
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    async def store_document_chunks(self, doc_metadata: DocumentMetadata, chunks: List[ChunkMetadata]) -> bool:
        """
        Store document metadata and all its chunks in the database

        Args:
            doc_metadata: Document metadata
            chunks: List of chunk metadata

        Returns:
            True if storage was successful
        """
        try:
            # Store document first
            await self.store_document_metadata(doc_metadata)

            # Store all chunks
            for chunk in chunks:
                await self.store_chunk_metadata(chunk)

            logger.info(f"Stored document {doc_metadata.id} with {len(chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error storing document and chunks: {str(e)}")
            raise

    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID

        Args:
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._sync_get_document_by_id,
                doc_id
            )

            if result:
                return DocumentMetadata(
                    id=result.id,
                    title=result.title,
                    source_file_path=result.source_file_path,
                    chapter=result.chapter,
                    section=result.section,
                    total_chunks=result.total_chunks,
                    word_count=result.word_count,
                    char_count=result.char_count,
                    status=result.status,
                    created_at=result.created_at,
                    updated_at=result.updated_at,
                    metadata=result.metadata_json
                )
            return None

        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            raise

    def _sync_get_document_by_id(self, doc_id: str) -> Optional[DocumentModel]:
        """Synchronous helper for getting document by ID"""
        db = next(self.get_db_session())
        try:
            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            return doc
        finally:
            db.close()

    async def get_chunks_by_document_id(self, doc_id: str) -> List[ChunkMetadata]:
        """
        Retrieve all chunks for a document by document ID

        Args:
            doc_id: Document ID

        Returns:
            List of chunk metadata
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._sync_get_chunks_by_document_id,
                doc_id
            )

            chunks = []
            for result in results:
                chunks.append(ChunkMetadata(
                    id=result.id,
                    document_id=result.document_id,
                    content=result.content,
                    chunk_index=result.chunk_index,
                    total_chunks=result.total_chunks,
                    vector_id=result.vector_id,
                    word_count=result.word_count,
                    char_count=result.char_count,
                    created_at=result.created_at,
                    updated_at=result.updated_at,
                    metadata=result.metadata_json
                ))

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks by document ID: {str(e)}")
            raise

    def _sync_get_chunks_by_document_id(self, doc_id: str) -> List[ChunkModel]:
        """Synchronous helper for getting chunks by document ID"""
        db = next(self.get_db_session())
        try:
            chunks = db.query(ChunkModel).filter(ChunkModel.document_id == doc_id).all()
            return chunks
        finally:
            db.close()

    async def get_document_status(self, doc_id: str) -> Optional[str]:
        """
        Get the indexing status of a document

        Args:
            doc_id: Document ID

        Returns:
            Status string or None if document not found
        """
        try:
            doc = await self.get_document_by_id(doc_id)
            return doc.status if doc else None
        except Exception as e:
            logger.error(f"Error getting document status: {str(e)}")
            raise

    async def update_document_status(self, doc_id: str, status: str) -> bool:
        """
        Update the indexing status of a document

        Args:
            doc_id: Document ID
            status: New status

        Returns:
            True if update was successful
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_update_document_status,
                doc_id,
                status
            )

            logger.info(f"Updated document {doc_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise

    def _sync_update_document_status(self, doc_id: str, status: str):
        """Synchronous helper for updating document status"""
        db = next(self.get_db_session())
        try:
            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if doc:
                doc.status = status
                doc.updated_at = datetime.utcnow()
                db.commit()
            else:
                raise ValueError(f"Document with ID {doc_id} not found")
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    async def get_all_documents(self, status_filter: Optional[str] = None) -> List[DocumentMetadata]:
        """
        Get all documents, optionally filtered by status

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of document metadata
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._sync_get_all_documents,
                status_filter
            )

            docs = []
            for result in results:
                docs.append(DocumentMetadata(
                    id=result.id,
                    title=result.title,
                    source_file_path=result.source_file_path,
                    chapter=result.chapter,
                    section=result.section,
                    total_chunks=result.total_chunks,
                    word_count=result.word_count,
                    char_count=result.char_count,
                    status=result.status,
                    created_at=result.created_at,
                    updated_at=result.updated_at,
                    metadata=result.metadata_json
                ))

            return docs

        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise

    def _sync_get_all_documents(self, status_filter: Optional[str] = None) -> List[DocumentModel]:
        """Synchronous helper for getting all documents"""
        db = next(self.get_db_session())
        try:
            query = db.query(DocumentModel)
            if status_filter:
                query = query.filter(DocumentModel.status == status_filter)
            return query.all()
        finally:
            db.close()

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks from the database

        Args:
            doc_id: Document ID

        Returns:
            True if deletion was successful
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_delete_document,
                doc_id
            )

            logger.info(f"Deleted document {doc_id} and its chunks")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def _sync_delete_document(self, doc_id: str):
        """Synchronous helper for deleting document and chunks"""
        db = next(self.get_db_session())
        try:
            # Delete chunks first (due to foreign key constraint)
            db.query(ChunkModel).filter(ChunkModel.document_id == doc_id).delete()
            # Then delete the document
            db.query(DocumentModel).filter(DocumentModel.id == doc_id).delete()
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()