from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.orchestrator import run_agent


router = APIRouter(prefix="/v1", tags=["agent"])


class AnalyzeRequest(BaseModel):
    dataset_path: str
    question: str
    table_name: str | None = "data"
    domain: Literal["general", "finance", "healthcare", "pipeline"] | None = "general"
    semantic_config: dict[str, Any] = Field(default_factory=dict)
    tool_whitelist: list[str] | None = None
    analysis_params: dict[str, Any] = Field(default_factory=dict)
    use_rag: bool = False
    rag_limit: int = 3


class ExecutedTool(BaseModel):
    tool: str
    status: str
    args: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    plan: list[str]
    executed_tools: list[ExecutedTool]
    insights: list[str]
    insight_objects: list[dict[str, Any]]
    diagnostics: list[str]
    charts: list[dict[str, Any]] = Field(default_factory=list)
    rag_used: bool = False
    retrieved_chunks: list[RetrievedChunkResponse] = Field(default_factory=list)
    rag_prompt: str | None = None
    combined_context: dict[str, Any] = Field(default_factory=dict)
    grounded_answer_input: str | None = None
    final_answer: str | None = None


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        result = run_agent(
            dataset_path=request.dataset_path,
            question=request.question,
            table_name=request.table_name,
            domain=request.domain,
            semantic_config=request.semantic_config,
            tool_whitelist=request.tool_whitelist,
            analysis_params=request.analysis_params,
            use_rag=request.use_rag,
            rag_limit=request.rag_limit,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))