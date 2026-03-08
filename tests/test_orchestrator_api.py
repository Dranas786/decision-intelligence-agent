from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.agent.orchestrator import run_agent
from app.main import app
from app.rag.factory import get_rag_service


client = TestClient(app)


def write_xyz(path: Path, points: np.ndarray) -> None:
    np.savetxt(path, points, fmt="%.6f")


def make_pipe_points(radius: float = 1.0, length: float = 8.0, dent_depth: float = 0.0, ovality: float = 0.0) -> np.ndarray:
    axial_values = np.linspace(-length / 2, length / 2, 45)
    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    rows: list[list[float]] = []
    for axial in axial_values:
        for angle in angles:
            local_radius = radius * (1 + ovality * np.cos(2 * angle))
            if dent_depth > 0 and abs(axial) < 0.6 and abs(angle) < 0.35:
                local_radius -= dent_depth
            rows.append([axial, local_radius * np.cos(angle), local_radius * np.sin(angle)])
    return np.asarray(rows, dtype=float)


def test_orchestrator_general_flow(tmp_path: Path):
    dataset_path = tmp_path / "business.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "revenue": [100, 101, 102, 99, 98, 120, 121, 123],
            "marketing_spend": [10, 11, 10, 9, 8, 14, 15, 16],
            "orders": [5, 6, 5, 4, 4, 7, 8, 8],
            "channel": ["paid", "paid", "organic", "organic", "paid", "email", "email", "email"],
        }
    ).to_csv(dataset_path, index=False)

    result = run_agent(
        dataset_path=str(dataset_path),
        question="Find anomalies, drivers, and forecast future revenue",
        semantic_config={"time_col": "date", "primary_metric": "revenue", "dimensions": ["channel"]},
    )

    assert "detect_anomalies" in result["plan"]
    assert "forecast_metric" in result["plan"]
    assert result["executed_tools"]
    assert result["insight_objects"]


def test_orchestrator_with_rag_enabled(tmp_path: Path):
    dataset_path = tmp_path / "business.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "revenue": [100, 102, 101, 130, 129, 131],
            "channel": ["paid", "paid", "organic", "email", "email", "email"],
        }
    ).to_csv(dataset_path, index=False)

    doc_path = tmp_path / "inspection_manual.txt"
    doc_path.write_text(
        "Pipeline dent classification guide:\n"
        "Minor dents are below 2 percent.\n"
        "Moderate dents are between 2 percent and 5 percent.\n"
        "Severe dents are above 5 percent and require engineering review.\n",
        encoding="utf-8",
    )

    rag_service = get_rag_service()
    ingest_result = rag_service.ingest_file(str(doc_path))
    assert ingest_result.chunk_count > 0

    result = run_agent(
        dataset_path=str(dataset_path),
        question="What dents require engineering review?",
        domain="general",
        semantic_config={"time_col": "date", "primary_metric": "revenue", "dimensions": ["channel"]},
        use_rag=True,
        rag_limit=3,
    )

    assert result["rag_used"] is True
    assert "retrieved_chunks" in result
    assert "rag_prompt" in result
    assert result["rag_prompt"] is not None
    assert result["retrieved_chunks"]
    assert any("engineering review" in chunk["text"].lower() for chunk in result["retrieved_chunks"])


def test_orchestrator_finance_healthcare_and_pipeline_domains(tmp_path: Path):
    finance_path = tmp_path / "finance.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D").tolist() * 2,
            "ticker": ["AAA"] * 6 + ["BBB"] * 6,
            "close": [100, 102, 103, 104, 105, 106, 50, 51, 52, 53, 54, 55],
            "volume": [1000, 1100, 900, 4000, 1000, 950, 800, 820, 830, 840, 850, 860],
            "signal": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        }
    ).to_csv(finance_path, index=False)

    finance_result = run_agent(
        dataset_path=str(finance_path),
        question="Analyze downside risk, volume spikes, and backtest the signal",
        domain="finance",
        semantic_config={"time_col": "date", "entity_col": "ticker", "price_col": "close", "volume_col": "volume", "signal_col": "signal"},
    )
    assert "measure_drawdown" in finance_result["plan"]
    assert any(step["status"] == "executed" for step in finance_result["executed_tools"])

    healthcare_path = tmp_path / "healthcare.csv"
    pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3"],
            "admission_date": ["2026-01-01", "2026-01-20", "2026-01-05", "2026-01-07"],
            "discharge_date": ["2026-01-05", "2026-01-25", "2026-01-09", "2026-01-10"],
            "cohort": ["A", "A", "B", "B"],
            "outcome": [1.0, 0.8, 0.4, 0.5],
            "duration_days": [10, 8, 20, 15],
            "event": [1, 0, 1, 1],
            "treatment": [0, 1, 0, 1],
        }
    ).to_csv(healthcare_path, index=False)

    healthcare_result = run_agent(
        dataset_path=str(healthcare_path),
        question="Compare cohorts, readmissions, survival, and treatment effect",
        domain="healthcare",
        semantic_config={
            "patient_id_col": "patient_id",
            "admission_date_col": "admission_date",
            "discharge_date_col": "discharge_date",
            "cohort_col": "cohort",
            "outcome_col": "outcome",
            "duration_col": "duration_days",
            "event_col": "event",
            "treatment_col": "treatment",
        },
    )
    assert "compute_readmission_rate" in healthcare_result["plan"]
    assert healthcare_result["insight_objects"]

    pipeline_path = tmp_path / "pipe.xyz"
    write_xyz(pipeline_path, make_pipe_points(dent_depth=0.12))
    pipeline_result = run_agent(
        dataset_path=str(pipeline_path),
        question="Find dents and deformation in this pipeline scan",
        domain="pipeline",
        semantic_config={"units": "m", "voxel_size": 0.03},
        analysis_params={"deviation_threshold": 0.05, "min_cluster_points": 10},
    )
    assert pipeline_result["plan"][0] == "profile_point_cloud"
    assert "detect_pipe_dents" in pipeline_result["plan"]
    assert any(step["tool"] == "detect_pipe_dents" for step in pipeline_result["executed_tools"])


def test_api_contract_and_invalid_inputs(tmp_path: Path):
    dataset_path = tmp_path / "api.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "revenue": [10, 12, 11, 20],
            "channel": ["paid", "paid", "organic", "organic"],
        }
    ).to_csv(dataset_path, index=False)

    response = client.post(
        "/v1/analyze",
        json={
            "dataset_path": str(dataset_path),
            "question": "Find segment drivers",
            "domain": "general",
            "semantic_config": {"time_col": "date", "primary_metric": "revenue", "dimensions": ["channel"]},
            "tool_whitelist": ["segment_drivers", "profile_table", "validate_dataset"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "executed_tools" in payload
    assert all(item["tool"] in {"segment_drivers", "profile_table", "validate_dataset"} for item in payload["executed_tools"])

    invalid_response = client.post(
        "/v1/analyze",
        json={
            "dataset_path": str(dataset_path),
            "question": "Run analysis",
            "domain": "unknown",
        },
    )
    assert invalid_response.status_code == 422

    bad_pipeline_path = tmp_path / "pipe.bad"
    bad_pipeline_path.write_text("1 2 3\n")
    unsupported_response = client.post(
        "/v1/analyze",
        json={
            "dataset_path": str(bad_pipeline_path),
            "question": "Inspect pipeline",
            "domain": "pipeline",
        },
    )
    assert unsupported_response.status_code == 400

    pipeline_path = tmp_path / "pipe.xyz"
    write_xyz(pipeline_path, make_pipe_points(dent_depth=0.08))
    invalid_hint_response = client.post(
        "/v1/analyze",
        json={
            "dataset_path": str(pipeline_path),
            "question": "Find dents and ovality",
            "domain": "pipeline",
            "semantic_config": {"units": "m", "axis_hint": [1, 2]},
            "analysis_params": {"deviation_threshold": 0.04, "min_cluster_points": 10, "slice_spacing": 0.4},
        },
    )
    assert invalid_hint_response.status_code == 200
    invalid_payload = invalid_hint_response.json()
    assert any("stable axis" in item.lower() or "axis hint" in item.lower() for item in invalid_payload["diagnostics"])
    assert all(tool not in {"profile_table", "validate_dataset"} for tool in invalid_payload["plan"])


def test_api_analyze_with_rag_enabled(tmp_path: Path):
    dataset_path = tmp_path / "api_rag.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "revenue": [10, 12, 11, 20],
            "channel": ["paid", "paid", "organic", "organic"],
        }
    ).to_csv(dataset_path, index=False)

    doc_path = tmp_path / "policy.txt"
    doc_path.write_text(
        "Inspection policy manual:\n"
        "Severe dents are above 5 percent and require engineering review.\n",
        encoding="utf-8",
    )

    rag_service = get_rag_service()
    ingest_result = rag_service.ingest_file(str(doc_path))
    assert ingest_result.chunk_count > 0

    response = client.post(
        "/v1/analyze",
        json={
            "dataset_path": str(dataset_path),
            "question": "What dents require engineering review?",
            "domain": "general",
            "semantic_config": {"time_col": "date", "primary_metric": "revenue", "dimensions": ["channel"]},
            "use_rag": True,
            "rag_limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rag_used"] is True
    assert "retrieved_chunks" in payload
    assert "rag_prompt" in payload
    assert payload["rag_prompt"] is not None
    assert payload["retrieved_chunks"]
    assert any("engineering review" in chunk["text"].lower() for chunk in payload["retrieved_chunks"])