from __future__ import annotations

from dataclasses import dataclass

from app.rag.ingestion import DocumentIngestor, IngestionResult
from app.rag.loaders import load_document_from_file
from app.rag.prompt_builder import build_rag_prompt
from app.rag.retriever import Retriever
from app.rag.schemas import RetrievedChunk


@dataclass
class RagService:
    """
    High-level service for document ingestion and retrieval-augmented prompt creation.
    """

    ingestor: DocumentIngestor
    retriever: Retriever

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> IngestionResult:
        document = load_document_from_file(file_path)

        return self.ingestor.ingest_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def retrieve_chunks(
        self,
        question: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query=question, limit=limit)

    def build_prompt_for_question(
        self,
        question: str,
        limit: int = 5,
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> str:
        chunks = retrieved_chunks if retrieved_chunks is not None else self.retrieve_chunks(question=question, limit=limit)
        return build_rag_prompt(
            question=question,
            retrieved_chunks=chunks,
        )
