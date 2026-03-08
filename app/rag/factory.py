from __future__ import annotations

from app.rag.embedder import SentenceTransformerEmbedder
from app.rag.ingestion import DocumentIngestor
from app.rag.retriever import Retriever
from app.rag.service import RagService
from app.rag.vector_store import QdrantVectorStore


_rag_service: RagService | None = None


def get_rag_service() -> RagService:
    global _rag_service

    if _rag_service is None:
        embedder = SentenceTransformerEmbedder()
        vector_size = embedder.model.get_sentence_embedding_dimension()

        vector_store = QdrantVectorStore(
            collection_name="knowledge_base",
            vector_size=vector_size,
        )

        ingestor = DocumentIngestor(
            embedder=embedder,
            vector_store=vector_store,
        )

        retriever = Retriever(
            embedder=embedder,
            vector_store=vector_store,
        )

        _rag_service = RagService(
            ingestor=ingestor,
            retriever=retriever,
        )

    return _rag_service