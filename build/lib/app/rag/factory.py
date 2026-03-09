from __future__ import annotations

import os
from pathlib import Path

from app.rag.embedder import get_embedder
from app.rag.ingestion import DocumentIngestor
from app.rag.retriever import Retriever
from app.rag.service import RagService
from app.rag.vector_store import QdrantVectorStore


_rag_service: RagService | None = None


def _collection_name() -> str:
    return os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")


def _build_vector_store(vector_size: int) -> QdrantVectorStore:
    qdrant_url = os.getenv("QDRANT_URL", "").strip() or None
    qdrant_host = os.getenv("QDRANT_HOST", "").strip() or None
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name = _collection_name()

    if qdrant_url:
        return QdrantVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

    if qdrant_host:
        return QdrantVectorStore(
            collection_name=collection_name,
            vector_size=vector_size,
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key,
        )

    local_path = Path(os.getenv("QDRANT_PATH", "data/qdrant")).resolve()
    return QdrantVectorStore(
        collection_name=collection_name,
        vector_size=vector_size,
        path=str(local_path),
    )


def get_rag_service() -> RagService:
    global _rag_service

    if _rag_service is None:
        embedder = get_embedder()
        vector_store = _build_vector_store(embedder.embedding_dimension())

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
