from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    """
    Contract for embedding text into numeric vectors.
    """

    def embed_text(self, text: str) -> list[float]:
        ...


@dataclass
class SentenceTransformerEmbedder:
    """
    Real semantic embedder using a local sentence-transformers model.
    """

    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            embedding_size = self.model.get_sentence_embedding_dimension()
            return [0.0] * embedding_size

        vector = self.model.encode(cleaned, normalize_embeddings=True)
        return vector.tolist()
