from __future__ import annotations

from pathlib import Path

from app.rag.schemas import SourceDocument


def load_document_from_file(file_path: str) -> SourceDocument:
    """
    Load a supported text-based document from disk and convert it into a SourceDocument.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise ValueError("Unsupported document type. Use .txt or .md for now.")

    content = path.read_text(encoding="utf-8").strip()

    return SourceDocument(
        document_id=path.stem,
        title=path.stem.replace("_", " ").replace("-", " ").title(),
        content=content,
        source_path=str(path),
        metadata={
            "file_type": suffix,
        },
    )