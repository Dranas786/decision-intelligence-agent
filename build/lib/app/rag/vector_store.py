from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.rag.schemas import DocumentChunk, RetrievedChunk


@dataclass
class QdrantVectorStore:
    """
    Stores and retrieves embedded document chunks using Qdrant.
    """

    collection_name: str
    vector_size: int
    host: str | None = None
    port: int | None = 6333
    url: str | None = None
    api_key: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.path:
            storage_path = Path(self.path)
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))
        elif self.url:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        elif self.host:
            self.client = QdrantClient(host=self.host, port=self.port or 6333, api_key=self.api_key)
        else:
            raise ValueError("QdrantVectorStore requires either path, url, or host configuration.")
        self._ensure_collection()

    def _point_id(self, chunk_id: str) -> str:
        """
        Convert an application chunk_id into a stable UUID string for Qdrant.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunk(self, chunk: DocumentChunk, embedding: list[float]) -> None:
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "metadata": chunk.metadata,
        }

        point = PointStruct(
            id=self._point_id(chunk.chunk_id),
            vector=embedding,
            payload=payload,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def upsert_chunks(
        self,
        chunk_embeddings: list[tuple[DocumentChunk, list[float]]],
    ) -> None:
        points: list[PointStruct] = []

        for chunk, embedding in chunk_embeddings:
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
        )

        points = search_result.points
        retrieved: list[RetrievedChunk] = []

        for point in points:
            payload: dict[str, Any] = point.payload or {}

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    document_id=str(payload.get("document_id", "")),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    metadata=payload.get("metadata", {}) or {},
                )
            )

        return retrieved
