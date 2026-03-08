from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agent.orchestrator import SUPPORTED_DOMAINS, run_agent
from app.rag.factory import get_rag_service


router = APIRouter(prefix="/v1/demo", tags=["demo"])

UPLOAD_ROOT = Path("data/demo_uploads")
PIPELINE_DATASET_EXTENSIONS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
TABULAR_DATASET_EXTENSIONS = {".csv"}
CONTEXT_FILE_EXTENSIONS = {".txt", ".md"}


def _safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def _dataset_extensions_for_domain(domain: str) -> set[str]:
    if domain == "pipeline":
        return PIPELINE_DATASET_EXTENSIONS
    return TABULAR_DATASET_EXTENSIONS


@router.post("/analyze-upload")
async def analyze_upload(
    question: str = Form(...),
    domain: str = Form("general"),
    table_name: str = Form("data"),
    use_rag: bool = Form(True),
    rag_limit: int = Form(3),
    semantic_config_json: str = Form("{}"),
    analysis_params_json: str = Form("{}"),
    dataset_file: UploadFile | None = File(default=None),
    csv_file: UploadFile | None = File(default=None),
    context_files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    try:
        semantic_config = json.loads(semantic_config_json or "{}")
        analysis_params = json.loads(analysis_params_json or "{}")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in semantic_config_json or analysis_params_json: {exc}")

    if domain not in SUPPORTED_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported domain '{domain}'. Use one of: {sorted(SUPPORTED_DOMAINS)}.")

    uploaded_dataset = dataset_file or csv_file
    if uploaded_dataset is None or not uploaded_dataset.filename:
        raise HTTPException(status_code=400, detail="A dataset file is required.")

    dataset_name = _safe_filename(uploaded_dataset.filename)
    dataset_suffix = Path(dataset_name).suffix.lower()
    allowed_extensions = _dataset_extensions_for_domain(domain)

    if dataset_suffix not in allowed_extensions:
        expected = ", ".join(sorted(allowed_extensions))
        raise HTTPException(status_code=400, detail=f"Uploaded dataset must match the selected domain. Expected one of: {expected}.")

    run_id = str(uuid.uuid4())[:8]
    run_dir = UPLOAD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = run_dir / dataset_name
    with dataset_path.open("wb") as file_obj:
        shutil.copyfileobj(uploaded_dataset.file, file_obj)

    ingested_files: list[dict[str, Any]] = []
    if context_files:
        rag_service = get_rag_service()

        for uploaded in context_files:
            if not uploaded.filename:
                continue

            safe_name = _safe_filename(uploaded.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in CONTEXT_FILE_EXTENSIONS:
                continue

            file_path = run_dir / safe_name
            with file_path.open("wb") as file_obj:
                shutil.copyfileobj(uploaded.file, file_obj)

            ingest_result = rag_service.ingest_file(str(file_path))
            ingested_files.append(
                {
                    "filename": safe_name,
                    "document_id": ingest_result.document_id,
                    "chunk_count": ingest_result.chunk_count,
                }
            )

    try:
        result = run_agent(
            dataset_path=str(dataset_path),
            question=question,
            table_name=table_name,
            domain=domain,
            semantic_config=semantic_config,
            analysis_params=analysis_params,
            use_rag=use_rag,
            rag_limit=rag_limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result["demo_run"] = {
        "run_id": run_id,
        "dataset_path": str(dataset_path),
        "dataset_kind": "point_cloud" if domain == "pipeline" else "table",
        "ingested_files": ingested_files,
    }
    return result
