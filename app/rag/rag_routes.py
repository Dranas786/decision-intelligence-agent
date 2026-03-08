from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.factory import get_rag_service


router = APIRouter(prefix="/v1/rag", tags=["rag"])


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to ingest.")
    chunk_size: int = Field(default=500, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)


class IngestFileResponse(BaseModel):
    document_id: str
    chunk_count: int


class AskRagRequest(BaseModel):
    question: str = Field(..., description="User question for retrieval.")
    limit: int = Field(default=5, ge=1, le=20)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class AskRagResponse(BaseModel):
    question: str
    prompt: str
    retrieved_chunks: list[RetrievedChunkResponse]


@router.post("/ingest-file", response_model=IngestFileResponse)
def ingest_file(request: IngestFileRequest):
    try:
        rag_service = get_rag_service()
        result = rag_service.ingest_file(
            file_path=request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return IngestFileResponse(
            document_id=result.document_id,
            chunk_count=result.chunk_count,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ask", response_model=AskRagResponse)
def ask_rag(request: AskRagRequest):
    try:
        rag_service = get_rag_service()
        retrieved_chunks = rag_service.retrieve_chunks(
            question=request.question,
            limit=request.limit,
        )
        prompt = rag_service.build_prompt_for_question(
            question=request.question,
            limit=request.limit,
        )

        return AskRagResponse(
            question=request.question,
            prompt=prompt,
            retrieved_chunks=[
                RetrievedChunkResponse(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
                for chunk in retrieved_chunks
            ],
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))