from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    """
    Represents one raw source document before chunking.
    """

    document_id: str
    title: str
    content: str
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """
    Represents one chunk extracted from a source document.
    """

    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """
    Represents one chunk returned from retrieval, including similarity score.
    """

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)