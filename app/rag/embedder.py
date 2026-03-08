from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from app.config import profile_default


class Embedder(Protocol):
    """
    Contract for embedding text into numeric vectors.
    """

    def embed_text(self, text: str) -> list[float]:
        ...

    def embedding_dimension(self) -> int:
        ...


@dataclass
class HashingTextEmbedder:
    """
    Lightweight deterministic embedder suitable for low-memory deployments.
    """

    vector_size: int = int(
        os.getenv(
            "EMBEDDING_VECTOR_SIZE",
            profile_default(hosted_free="384", local_full="384"),
        )
    )
    _vectorizer: HashingVectorizer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=self.vector_size,
            alternate_sign=False,
            norm="l2",
        )

    def embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * self.vector_size

        matrix = self._vectorizer.transform([cleaned])
        vector = matrix.toarray()[0].astype(np.float32, copy=False)
        return vector.tolist()

    def embedding_dimension(self) -> int:
        return self.vector_size


@dataclass
class SentenceTransformerEmbedder:
    """
    Semantic embedder using a local sentence-transformers model.
    """

    model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Install the 'rag-local' extra or switch EMBEDDING_PROVIDER=hash."
            ) from exc

        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * self.embedding_dimension()

        vector = self.model.encode(cleaned, normalize_embeddings=True)
        return vector.tolist()

    def embedding_dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def get_embedder() -> Embedder:
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        profile_default(hosted_free="hash", local_full="sentence-transformer"),
    ).strip().lower()
    if provider in {"sentence-transformer", "sentence_transformer", "sentence-transformers", "st"}:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            return HashingTextEmbedder()
    return HashingTextEmbedder()
