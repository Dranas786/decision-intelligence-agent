from __future__ import annotations

from dataclasses import dataclass

from app.rag.chunker import chunk_document
from app.rag.embedder import Embedder
from app.rag.schemas import SourceDocument
from app.rag.vector_store import QdrantVectorStore


@dataclass
class IngestionResult:
    """
    Summary of a document ingestion run.
    """

    document_id: str
    chunk_count: int


@dataclass
class DocumentIngestor:
    """
    Handles document chunking, embedding, and vector-store insertion.
    """

    embedder: Embedder
    vector_store: QdrantVectorStore

    def ingest_document(
        self,
        document: SourceDocument,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> IngestionResult:
        chunks = chunk_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        chunk_embeddings = [
            (chunk, self.embedder.embed_text(chunk.text))
            for chunk in chunks
        ]

        self.vector_store.upsert_chunks(chunk_embeddings)

        return IngestionResult(
            document_id=document.document_id,
            chunk_count=len(chunks),
        )