from __future__ import annotations

from app.rag.schemas import SourceDocument, DocumentChunk


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Split a source document into overlapping text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk = DocumentChunk(
                chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                document_id=document.document_id,
                text=chunk_text,
                chunk_index=chunk_index,
                metadata={
                    "title": document.title,
                    "source_path": document.source_path,
                    **document.metadata,
                },
            )
            chunks.append(chunk)
            chunk_index += 1

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks