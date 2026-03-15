from __future__ import annotations

import json
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.agent.orchestrator import SUPPORTED_DOMAINS, run_agent
from app.engineering.local_pipeline import build_local_report
from app.rag.factory import get_rag_service


router = APIRouter(prefix="/v1/demo", tags=["demo"])

UPLOAD_ROOT = Path("data/demo_uploads")
FILES_FOR_DEMO_ROOT = Path("files_for_demo")
PIPELINE_DATASET_EXTENSIONS = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
TABULAR_DATASET_EXTENSIONS = {".csv"}
CONTEXT_FILE_EXTENSIONS = {".txt", ".md"}
ANALYSIS_PAYLOAD_NAME = "analysis_payload.json"
REPORT_PAYLOAD_NAME = "report_payload.json"


class BuildReportRequest(BaseModel):
    dataset_path: str
    question: str
    domain: Literal["general", "finance", "healthcare", "pipeline"]
    semantic_config: dict[str, Any] = Field(default_factory=dict)
    analysis_params: dict[str, Any] = Field(default_factory=dict)
    analysis_result: dict[str, Any] = Field(default_factory=dict)
    visualization_request: str = ""


class AnalyzeSampleRequest(BaseModel):
    domain: Literal["general", "finance", "healthcare", "pipeline"]
    question: str | None = None
    table_name: str = "data"
    use_rag: bool = True
    rag_limit: int = 3
    semantic_config: dict[str, Any] = Field(default_factory=dict)
    analysis_params: dict[str, Any] = Field(default_factory=dict)



def _safe_filename(name: str) -> str:
    return Path(name).name.replace(" ", "_")



def _dataset_extensions_for_domain(domain: str) -> set[str]:
    if domain == "pipeline":
        return PIPELINE_DATASET_EXTENSIONS
    return TABULAR_DATASET_EXTENSIONS



def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")



def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))



def _sample_package_catalog() -> dict[str, dict[str, Any]]:
    return {
        "general": {
            "label": "Coffee quality governance pack",
            "dataset": FILES_FOR_DEMO_ROOT / "coffee_quality_demo.csv",
            "context_files": [FILES_FOR_DEMO_ROOT / "coffee_quality_context.txt"],
            "question": "Assess this dataset for quality and prepare it for analytics. Explain the quality issues found, governance risks, schema concerns, freshness concerns, and what requires human review.",
            "semantic_config": {"time_col": "last_update", "primary_metric": "visit_count", "dimensions": ["neighbourhood"]},
            "analysis_params": {},
            "visualization_request": "Build a local bronze/silver/gold pipeline and create a report that emphasizes data quality status, field-level issues, standardization opportunities, governance review items, and a business-facing metric rollup by neighbourhood.",
            "notes": "Retail-style governance demo with duplicate rows, missing fields, standardization issues, and sensitive-looking data.",
        },
        "finance": {
            "label": "Market monitoring pack",
            "dataset": FILES_FOR_DEMO_ROOT / "finance_market_demo.csv",
            "context_files": [FILES_FOR_DEMO_ROOT / "finance_market_context.txt"],
            "question": "Assess this market dataset for quality, then calculate returns, risk, drawdown, and unusual volume activity. Explain any governance risks before this would be used in reporting.",
            "semantic_config": {"time_col": "date", "entity_col": "ticker", "price_col": "close", "volume_col": "volume", "benchmark_col": "benchmark_close", "signal_col": "signal"},
            "analysis_params": {},
            "visualization_request": "Build a local bronze/silver/gold pipeline and create a report that emphasizes quality checks, returns, drawdown, risk summary, unusual volume, and a stakeholder-ready market monitoring view.",
            "notes": "Small market-surveillance example with time-series measures and reporting risk checks.",
        },
        "healthcare": {
            "label": "Admissions quality pack",
            "dataset": FILES_FOR_DEMO_ROOT / "healthcare_admissions_demo.csv",
            "context_files": [FILES_FOR_DEMO_ROOT / "healthcare_context.txt"],
            "question": "Assess this admissions dataset for quality, compare cohorts, identify readmission risk patterns, and explain governance concerns or fields that require human review before reporting.",
            "semantic_config": {"patient_id_col": "patient_id", "admission_date_col": "admission_date", "discharge_date_col": "discharge_date", "cohort_col": "cohort", "outcome_col": "outcome_score", "duration_col": "duration_days", "event_col": "event", "treatment_col": "treatment"},
            "analysis_params": {},
            "visualization_request": "Build a local bronze/silver/gold pipeline and create a report that emphasizes data quality, cohort comparison, readmission-related patterns, length of stay, and governance review items for healthcare reporting.",
            "notes": "Healthcare cohort example with governance-sensitive fields and reporting review points.",
        },
        "pipeline": {
            "label": "Multi-defect pipeline inspection pack",
            "dataset": FILES_FOR_DEMO_ROOT / "pipeline_multi_defect_demo.xyz",
            "context_files": [
                FILES_FOR_DEMO_ROOT / "pipeline_inspection_context.md",
                FILES_FOR_DEMO_ROOT / "pipeline_multi_defect_context.md",
            ],
            "question": "Find dents and deformation in this pipeline scan, compute the deviation map, measure ovality, assess fit quality, and explain which findings should be treated as engineering-review items.",
            "semantic_config": {"units": "m", "voxel_size": 0.03},
            "analysis_params": {"deviation_threshold": 0.05, "min_cluster_points": 10, "slice_spacing": 0.35},
            "visualization_request": "Build a local bronze/silver/gold pipeline and create a report with a 3D scan view, cross-section view, unwrapped defect map, axial defect profile, ovality profile, dent summary, dent risk matrix, and a clear engineering review queue.",
            "notes": "Richer synthetic scan with multiple dents, broad ovality, and more realistic surface variation for the local demo.",
        },
    }



def _get_sample_package(domain: str) -> dict[str, Any]:
    package = _sample_package_catalog().get(domain)
    if package is None:
        raise HTTPException(status_code=404, detail=f"No sample package is configured for domain '{domain}'.")
    dataset = package["dataset"]
    if not dataset.exists():
        raise HTTPException(status_code=500, detail=f"Configured sample dataset is missing: {dataset}")
    for path in package.get("context_files", []):
        if not Path(path).exists():
            raise HTTPException(status_code=500, detail=f"Configured sample context file is missing: {path}")
    return deepcopy(package)



def _package_metadata(domain: str) -> dict[str, Any]:
    package = _get_sample_package(domain)
    return {
        "domain": domain,
        "label": package["label"],
        "dataset_name": package["dataset"].name,
        "context_files": [Path(path).name for path in package.get("context_files", [])],
        "question": package["question"],
        "semantic_config": package["semantic_config"],
        "analysis_params": package["analysis_params"],
        "visualization_request": package["visualization_request"],
        "notes": package["notes"],
    }



def _materialize_context_files(run_dir: Path, context_paths: list[Path], use_rag: bool) -> list[dict[str, Any]]:
    ingested_files: list[dict[str, Any]] = []
    rag_service = get_rag_service() if use_rag and context_paths else None
    for source_path in context_paths:
        safe_name = _safe_filename(source_path.name)
        destination = run_dir / safe_name
        shutil.copy2(source_path, destination)
        if rag_service is not None:
            ingest_result = rag_service.ingest_file(str(destination))
            ingested_files.append({"filename": safe_name, "document_id": ingest_result.document_id, "chunk_count": ingest_result.chunk_count})
        else:
            ingested_files.append({"filename": safe_name, "document_id": None, "chunk_count": 0})
    return ingested_files



def _run_analysis_and_persist(*, run_dir: Path, run_id: str, dataset_path: Path, question: str, domain: str, table_name: str, use_rag: bool, rag_limit: int, semantic_config: dict[str, Any], analysis_params: dict[str, Any], ingested_files: list[dict[str, Any]], sample_label: str | None = None) -> dict[str, Any]:
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
        "extra_info_url": f"/v1/demo/extra_info/{run_id}",
        "sample_label": sample_label,
    }

    _write_json(
        run_dir / ANALYSIS_PAYLOAD_NAME,
        {
            "request": {
                "question": question,
                "domain": domain,
                "table_name": table_name,
                "use_rag": use_rag,
                "rag_limit": rag_limit,
                "semantic_config": semantic_config,
                "analysis_params": analysis_params,
                "sample_label": sample_label,
            },
            "result": result,
        },
    )
    return result


@router.get("/sample-package/{domain}")
def sample_package(domain: str) -> dict[str, Any]:
    if domain not in SUPPORTED_DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unsupported domain '{domain}'.")
    return _package_metadata(domain)


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

    accepted_context_paths: list[Path] = []
    if context_files:
        for uploaded in context_files:
            if not uploaded.filename:
                continue
            safe_name = _safe_filename(uploaded.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in CONTEXT_FILE_EXTENSIONS:
                continue
            destination = run_dir / safe_name
            with destination.open("wb") as file_obj:
                shutil.copyfileobj(uploaded.file, file_obj)
            accepted_context_paths.append(destination)

    ingested_files = []
    if accepted_context_paths:
        rag_service = get_rag_service() if use_rag else None
        for file_path in accepted_context_paths:
            if rag_service is not None:
                ingest_result = rag_service.ingest_file(str(file_path))
                ingested_files.append({"filename": file_path.name, "document_id": ingest_result.document_id, "chunk_count": ingest_result.chunk_count})
            else:
                ingested_files.append({"filename": file_path.name, "document_id": None, "chunk_count": 0})

    return _run_analysis_and_persist(
        run_dir=run_dir,
        run_id=run_id,
        dataset_path=dataset_path,
        question=question,
        domain=domain,
        table_name=table_name,
        use_rag=use_rag,
        rag_limit=rag_limit,
        semantic_config=semantic_config,
        analysis_params=analysis_params,
        ingested_files=ingested_files,
    )


@router.post("/analyze-sample")
def analyze_sample(request: AnalyzeSampleRequest) -> dict[str, Any]:
    package = _get_sample_package(request.domain)
    run_id = str(uuid.uuid4())[:8]
    run_dir = UPLOAD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = package["dataset"]
    dataset_path = run_dir / _safe_filename(source_dataset.name)
    shutil.copy2(source_dataset, dataset_path)

    semantic_config = dict(package.get("semantic_config", {}))
    semantic_config.update(request.semantic_config or {})
    analysis_params = dict(package.get("analysis_params", {}))
    analysis_params.update(request.analysis_params or {})
    question = request.question.strip() if request.question else package["question"]

    ingested_files = _materialize_context_files(run_dir, [Path(path) for path in package.get("context_files", [])], request.use_rag)

    return _run_analysis_and_persist(
        run_dir=run_dir,
        run_id=run_id,
        dataset_path=dataset_path,
        question=question,
        domain=request.domain,
        table_name=request.table_name,
        use_rag=request.use_rag,
        rag_limit=request.rag_limit,
        semantic_config=semantic_config,
        analysis_params=analysis_params,
        ingested_files=ingested_files,
        sample_label=package["label"],
    )


@router.post("/build-report")
def build_report(request: BuildReportRequest) -> dict[str, Any]:
    dataset_path = Path(request.dataset_path)
    if not dataset_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset path does not exist: {request.dataset_path}")

    try:
        report = build_local_report(
            dataset_path=str(dataset_path),
            domain=request.domain,
            question=request.question,
            semantic_config=request.semantic_config,
            analysis_params=request.analysis_params,
            analysis_result=request.analysis_result,
            visualization_request=request.visualization_request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    run_id = request.analysis_result.get("demo_run", {}).get("run_id") or dataset_path.stem
    run_dir = UPLOAD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / REPORT_PAYLOAD_NAME,
        {
            "request": request.model_dump(),
            "result": report,
        },
    )
    report["extra_info_url"] = f"/v1/demo/extra_info/{run_id}"
    return report


@router.get("/extra_info/{run_id}")
def extra_info(run_id: str) -> dict[str, Any]:
    run_dir = UPLOAD_ROOT / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run id '{run_id}' was not found.")

    analysis_payload = _read_json(run_dir / ANALYSIS_PAYLOAD_NAME)
    report_payload = _read_json(run_dir / REPORT_PAYLOAD_NAME)
    if analysis_payload is None and report_payload is None:
        raise HTTPException(status_code=404, detail=f"No extra info is available yet for run '{run_id}'.")

    return {
        "run_id": run_id,
        "analysis": analysis_payload,
        "report": report_payload,
    }
