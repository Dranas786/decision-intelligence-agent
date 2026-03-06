from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.agent.orchestrator import run_agent


router = APIRouter(prefix="/v1", tags=["agent"])


class AnalyzeRequest(BaseModel):
    dataset_path: str
    question: str
    table_name: Optional[str] = "data"


class AnalyzeResponse(BaseModel):
    plan: list
    insights: list
    charts: Optional[list] = []


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):

    try:
        result = run_agent(
            dataset_path=request.dataset_path,
            question=request.question,
            table_name=request.table_name
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))