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


class ExecutedTool(BaseModel):
    tool: str
    status: str
    args: dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    plan: list[str]
    executed_tools: list[ExecutedTool]
    insights: list[str]
    insight_objects: list[dict[str, Any]]
    diagnostics: list[str]
    charts: list[dict[str, Any]] = Field(default_factory=list)


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
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
