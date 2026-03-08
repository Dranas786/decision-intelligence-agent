from __future__ import annotations

from dataclasses import dataclass

from app.rag.embedder import Embedder
from app.rag.schemas import RetrievedChunk
from app.rag.vector_store import QdrantVectorStore


@dataclass
class Retriever:
    """
    Retrieves the most relevant document chunks for a user query.
    """

    embedder: Embedder
    vector_store: QdrantVectorStore

    def retrieve(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        query_embedding = self.embedder.embed_text(cleaned_query)
        return self.vector_store.search(query_embedding=query_embedding, limit=limit)