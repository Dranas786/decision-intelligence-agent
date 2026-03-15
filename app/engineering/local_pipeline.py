
from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.agent.tools import load_dataset
from app.analytics.common import find_time_column, first_numeric
from app.analytics.pipeline_3d import (
    DEFAULT_SEVERITY_BANDS,
    clean_point_cloud,
    compute_pipe_deviation_map,
    detect_pipe_dents,
    fit_pipe_cylinder,
    load_point_cloud,
    measure_pipe_ovality,
)


REPORT_ROOT = Path("data/demo_reports")


def _tool_evidence_map(analysis_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for item in analysis_result.get("insight_objects", []):
        tool = item.get("tool")
        if tool and tool not in evidence:
            evidence[tool] = item.get("evidence", {}) or {}
    return evidence


def build_local_report(
    *,
    dataset_path: str,
    domain: str,
    question: str,
    semantic_config: dict[str, Any] | None = None,
    analysis_params: dict[str, Any] | None = None,
    analysis_result: dict[str, Any] | None = None,
    visualization_request: str = "",
) -> dict[str, Any]:
    semantic_config = semantic_config or {}
    analysis_params = analysis_params or {}
    analysis_result = analysis_result or {}

    run_hint = analysis_result.get("demo_run", {}).get("run_id") or Path(dataset_path).stem or str(uuid.uuid4())[:8]
    report_dir = REPORT_ROOT / f"{run_hint}_report"
    bronze_dir = report_dir / "bronze"
    silver_dir = report_dir / "silver"
    gold_dir = report_dir / "gold"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    if domain == "pipeline":
        report = _build_pipeline_report(
            dataset_path=dataset_path,
            question=question,
            semantic_config=semantic_config,
            analysis_params=analysis_params,
            analysis_result=analysis_result,
            visualization_request=visualization_request,
            bronze_dir=bronze_dir,
            silver_dir=silver_dir,
            gold_dir=gold_dir,
        )
    else:
        report = _build_tabular_report(
            dataset_path=dataset_path,
            question=question,
            semantic_config=semantic_config,
            analysis_result=analysis_result,
            visualization_request=visualization_request,
            bronze_dir=bronze_dir,
            silver_dir=silver_dir,
            gold_dir=gold_dir,
        )

    report["artifact_root"] = str(report_dir.resolve())
    report["agent_story"] = [
        {"agent": "Business Shareholder", "role": "Sets the business question and decides whether to build a reporting pipeline."},
        {"agent": "Data Analyst", "role": "Runs deterministic profiling, validation, and domain analytics."},
        {"agent": "Data Engineer", "role": "Builds local bronze, silver, and gold outputs plus SQLite tables."},
        {"agent": "Visualization Studio", "role": "Transforms engineered outputs into stakeholder-facing views."},
    ]
    return report


def _build_tabular_report(
    *,
    dataset_path: str,
    question: str,
    semantic_config: dict[str, Any],
    analysis_result: dict[str, Any],
    visualization_request: str,
    bronze_dir: Path,
    silver_dir: Path,
    gold_dir: Path,
) -> dict[str, Any]:
    df = load_dataset(dataset_path).copy()
    explanation_layer = analysis_result.get("explanation_layer", {}) or {}
    tool_evidence = _tool_evidence_map(analysis_result)

    raw_copy_path = bronze_dir / Path(dataset_path).name
    shutil.copy2(dataset_path, raw_copy_path)

    time_col = semantic_config.get("time_col") or find_time_column(df)
    primary_metric = semantic_config.get("primary_metric") or first_numeric(df)
    dimensions = semantic_config.get("dimensions") or []
    dimension = None
    if dimensions:
        candidate = dimensions[0]
        if isinstance(candidate, dict):
            candidate = candidate.get("name")
        if candidate in df.columns:
            dimension = candidate
    if dimension is None:
        dimension = next((column for column in df.columns if column.lower() in {"segment", "channel", "category", "neighbourhood"}), None)

    silver = df.copy()
    silver.insert(0, "record_id", range(1, len(silver) + 1))
    silver["dq_duplicate_row"] = silver.duplicated(keep=False)

    if primary_metric and primary_metric in silver.columns:
        metric_series = pd.to_numeric(silver[primary_metric], errors="coerce")
        silver[f"dq_missing_{primary_metric}"] = metric_series.isna()
        silver[f"dq_negative_{primary_metric}"] = metric_series < 0
    else:
        metric_series = pd.Series([np.nan] * len(silver))

    if time_col and time_col in silver.columns:
        parsed_dates = pd.to_datetime(silver[time_col], errors="coerce")
        silver[f"{time_col}_parsed"] = parsed_dates.dt.strftime("%Y-%m-%d")
        silver[f"dq_{time_col}_parse_failed"] = silver[time_col].notna() & parsed_dates.isna()

    for column in silver.select_dtypes(include=["object"]).columns:
        if column.lower() in {"neighbourhood", "segment", "channel", "category", "store_name"}:
            silver[f"{column}_normalized"] = silver[column].fillna("").astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True).str.title()

    quality_flag_columns = [column for column in silver.columns if column.startswith("dq_")]
    silver["dq_review_required"] = silver[quality_flag_columns].any(axis=1) if quality_flag_columns else False
    silver["dq_status"] = np.where(silver["dq_review_required"], "review_required", "ready")

    silver_path = silver_dir / "silver_records.csv"
    silver.to_csv(silver_path, index=False)
    quality_events_path = silver_dir / "quality_events.csv"
    silver[["record_id", "dq_status", "dq_review_required", *quality_flag_columns]].to_csv(quality_events_path, index=False)

    field_profile = _build_field_profile(df, tool_evidence)
    quality_issue_facts = _build_quality_issue_facts(df, silver, tool_evidence)
    governance_queue = _build_governance_review_queue(explanation_layer)
    standardization_df = _flatten_standardization_candidates(tool_evidence)
    collision_df = _flatten_collision_candidates(tool_evidence)
    freshness_df = _build_freshness_table(tool_evidence)

    db_path = gold_dir / "local_reporting.db"
    conn = sqlite3.connect(db_path)
    try:
        dim_entity = _build_dim_entity(silver)
        fact_records = _build_fact_records(silver, dim_entity, time_col)
        dim_entity.to_sql("dim_entity", conn, if_exists="replace", index=False)
        fact_records.to_sql("fact_records", conn, if_exists="replace", index=False)

        if time_col and f"{time_col}_parsed" in silver.columns:
            dim_time = _build_dim_time(silver[f"{time_col}_parsed"])
            dim_time.to_sql("dim_time", conn, if_exists="replace", index=False)

        dq_summary = _build_tabular_quality_summary(df, silver, explanation_layer, primary_metric, tool_evidence)
        dq_summary.to_sql("dq_summary", conn, if_exists="replace", index=False)
        field_profile.to_sql("dim_field_profile", conn, if_exists="replace", index=False)
        quality_issue_facts.to_sql("fact_quality_issues", conn, if_exists="replace", index=False)
        governance_queue.to_sql("governance_review_queue", conn, if_exists="replace", index=False)
        if not standardization_df.empty:
            standardization_df.to_sql("standardization_candidates", conn, if_exists="replace", index=False)
        if not collision_df.empty:
            collision_df.to_sql("entity_collision_candidates", conn, if_exists="replace", index=False)
        if not freshness_df.empty:
            freshness_df.to_sql("freshness_summary", conn, if_exists="replace", index=False)

        if dimension and dimension in silver.columns and primary_metric and primary_metric in silver.columns:
            rollup = silver.assign(_metric=pd.to_numeric(silver[primary_metric], errors="coerce")).groupby(dimension, dropna=False)["_metric"].agg(["count", "mean", "sum"]).reset_index()
            rollup.to_sql("gold_metric_by_dimension", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    report_summary = {
        "question": question,
        "visualization_request": visualization_request,
        "quality_findings": explanation_layer.get("quality_findings", []),
        "governance_notes": explanation_layer.get("governance_notes", []),
        "human_review_required": explanation_layer.get("human_review_required", []),
        "schema_status": tool_evidence.get("audit_schema_contract", {}).get("schema_status"),
        "freshness_status": tool_evidence.get("assess_freshness", {}).get("freshness_status"),
        "artifact_paths": {
            "bronze": str(raw_copy_path.resolve()),
            "silver": str(silver_path.resolve()),
            "quality_events": str(quality_events_path.resolve()),
            "gold_database": str(db_path.resolve()),
        },
    }
    (gold_dir / "report_summary.json").write_text(json.dumps(report_summary, indent=2), encoding="utf-8")

    report_markdown = "\n".join(
        [
            "# Local Data Product",
            "",
            "## Bronze",
            f"- Raw dataset copied to `{raw_copy_path}`.",
            "",
            "## Silver",
            f"- Standardized record set written to `{silver_path}`.",
            f"- Quality-event table written to `{quality_events_path}`.",
            "",
            "## Gold",
            f"- SQLite reporting database written to `{db_path}`.",
            "- Gold tables include `dim_entity`, `fact_records`, `dq_summary`, and optional rollups.",
        ]
    )

    tables = ["dim_entity", "fact_records", "dq_summary", "dim_field_profile", "fact_quality_issues", "governance_review_queue"]
    if time_col and f"{time_col}_parsed" in silver.columns:
        tables.append("dim_time")
    if not standardization_df.empty:
        tables.append("standardization_candidates")
    if not collision_df.empty:
        tables.append("entity_collision_candidates")
    if not freshness_df.empty:
        tables.append("freshness_summary")
    if dimension and primary_metric:
        tables.append("gold_metric_by_dimension")

    return {
        "mode": "tabular",
        "report_markdown": report_markdown,
        "layers": {
            "bronze": {"path": str(raw_copy_path.resolve()), "description": "Immutable raw copy of the uploaded dataset."},
            "silver": {"path": str(silver_path.resolve()), "description": "Standardized dataset with row-level quality flags."},
            "gold": {"path": str(db_path.resolve()), "description": "SQLite reporting layer with fact and dimension tables."},
        },
        "tables": tables,
        "visualizations": _build_tabular_visualizations(df, silver, primary_metric, dimension, explanation_layer, visualization_request, tool_evidence, quality_issue_facts),
        "report_summary": report_summary,
    }


def _build_dim_entity(df: pd.DataFrame) -> pd.DataFrame:
    id_col = next((column for column in df.columns if column.lower().endswith("id")), None)
    name_col = next((column for column in df.columns if "name" in column.lower() and "email" not in column.lower()), None)
    dimension_col = next((column for column in df.columns if column.lower() in {"segment", "channel", "category", "neighbourhood"}), None)
    cols = [column for column in [id_col, name_col, dimension_col] if column]
    if not cols:
        return pd.DataFrame({"entity_key": [1], "entity_label": ["dataset"]})
    entity = df[cols].drop_duplicates().reset_index(drop=True)
    entity.insert(0, "entity_key", range(1, len(entity) + 1))
    return entity


def _build_fact_records(df: pd.DataFrame, dim_entity: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    fact = df.copy()
    merge_cols = [column for column in dim_entity.columns if column != "entity_key"]
    if merge_cols:
        fact = fact.merge(dim_entity, on=merge_cols, how="left")
    if time_col and f"{time_col}_parsed" in fact.columns:
        fact["date_key"] = fact[f"{time_col}_parsed"].str.replace("-", "", regex=False)
    return fact


def _build_dim_time(parsed_dates: pd.Series) -> pd.DataFrame:
    clean = parsed_dates.dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if clean.empty:
        return pd.DataFrame(columns=["date_key", "date", "year", "month", "day"])
    dt = pd.to_datetime(clean, errors="coerce")
    return pd.DataFrame({"date_key": clean.str.replace("-", "", regex=False), "date": clean, "year": dt.dt.year, "month": dt.dt.month, "day": dt.dt.day})


def _build_tabular_quality_summary(
    raw_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    explanation_layer: dict[str, Any],
    primary_metric: str | None,
    tool_evidence: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    validation = tool_evidence.get("validate_dataset", {})
    freshness = tool_evidence.get("assess_freshness", {})
    rows = [
        {"metric": "row_count", "value": float(len(raw_df)), "notes": "Raw row count."},
        {"metric": "column_count", "value": float(len(raw_df.columns)), "notes": "Raw column count."},
        {"metric": "duplicate_share", "value": float(raw_df.duplicated().mean()), "notes": "Exact duplicate-row share."},
        {"metric": "review_required_rows", "value": float(silver_df["dq_review_required"].sum()), "notes": "Rows flagged for human review."},
    ]
    if primary_metric and primary_metric in raw_df.columns:
        rows.append({"metric": f"missing_{primary_metric}", "value": float(pd.to_numeric(raw_df[primary_metric], errors="coerce").isna().mean()), "notes": "Missing share for the primary metric."})
    quality_dimensions = validation.get("quality_dimensions", {})
    if isinstance(quality_dimensions, dict):
        for key in ("completeness", "uniqueness"):
            if key in quality_dimensions:
                rows.append({"metric": key, "value": float(quality_dimensions[key]), "notes": f"Quality-dimension score for {key}."})
    if freshness:
        rows.append({"metric": "freshness_age_days", "value": float(freshness.get("freshness_age_days", 0.0)), "notes": f"Freshness status: {freshness.get('freshness_status')}"})
    for note in explanation_layer.get("human_review_required", [])[:5]:
        rows.append({"metric": "human_review_required", "value": 1.0, "notes": note})
    return pd.DataFrame(rows)



def _build_field_profile(df: pd.DataFrame, tool_evidence: dict[str, dict[str, Any]]) -> pd.DataFrame:
    profile_evidence = tool_evidence.get("profile_table", {})
    sensitive_fields = set(profile_evidence.get("sensitive_fields", []))
    ambiguous_fields = set(profile_evidence.get("ambiguous_columns", []))
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        rows.append(
            {
                "field_name": column,
                "dtype": str(df[column].dtype),
                "missing_share": round(float(df[column].isna().mean()), 4),
                "distinct_count": int(df[column].nunique(dropna=True)),
                "sensitive_flag": column in sensitive_fields,
                "ambiguous_flag": column in ambiguous_fields,
            }
        )
    return pd.DataFrame(rows)



def _build_quality_issue_facts(df: pd.DataFrame, silver: pd.DataFrame, tool_evidence: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        missing_share = float(df[column].isna().mean())
        if missing_share > 0:
            rows.append({"field_name": column, "issue_type": "missingness", "issue_value": round(missing_share, 4)})

    for column in [column for column in silver.columns if column.startswith("dq_")]:
        issue_count = int(silver[column].fillna(False).astype(bool).sum())
        if issue_count > 0:
            field_name = column.replace("dq_", "").replace("_parse_failed", "")
            rows.append({"field_name": field_name, "issue_type": column, "issue_value": issue_count})

    for candidate in tool_evidence.get("audit_standardization", {}).get("standardization_candidates", []):
        rows.append({"field_name": candidate.get("column"), "issue_type": "standardization_variants", "issue_value": len(candidate.get("variant_groups", []))})

    for column in tool_evidence.get("audit_schema_contract", {}).get("missing_columns", []):
        rows.append({"field_name": column, "issue_type": "missing_required_column", "issue_value": 1})

    return pd.DataFrame(rows or [{"field_name": "dataset", "issue_type": "none", "issue_value": 0}])



def _build_governance_review_queue(explanation_layer: dict[str, Any]) -> pd.DataFrame:
    items = explanation_layer.get("human_review_required", []) or []
    if not items:
        return pd.DataFrame([{"queue_item": "No open review items", "status": "clear", "owner": "business"}])
    return pd.DataFrame(
        [
            {"queue_item": item, "status": "open", "owner": "business" if index == 0 else "analyst"}
            for index, item in enumerate(items)
        ]
    )



def _flatten_standardization_candidates(tool_evidence: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in tool_evidence.get("audit_standardization", {}).get("standardization_candidates", []):
        for group in candidate.get("variant_groups", []):
            rows.append(
                {
                    "field_name": candidate.get("column"),
                    "normalized_value": group.get("normalized_value"),
                    "raw_variants": ", ".join(group.get("raw_variants", [])),
                    "variant_count": group.get("variant_count"),
                }
            )
    return pd.DataFrame(rows)



def _flatten_collision_candidates(tool_evidence: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in tool_evidence.get("detect_entity_collisions", {}).get("collision_candidates", []):
        rows.append(
            {
                "match_columns": ", ".join(candidate.get("match_columns", [])),
                "normalized_key": json.dumps(candidate.get("normalized_key", {})),
                "row_count": candidate.get("row_count"),
                "sample_rows": json.dumps(candidate.get("sample_rows", [])),
            }
        )
    return pd.DataFrame(rows)



def _build_freshness_table(tool_evidence: dict[str, dict[str, Any]]) -> pd.DataFrame:
    freshness = tool_evidence.get("assess_freshness", {})
    if not freshness:
        return pd.DataFrame()
    return pd.DataFrame([freshness])


def _build_pipeline_report(
    *,
    dataset_path: str,
    question: str,
    semantic_config: dict[str, Any],
    analysis_params: dict[str, Any],
    analysis_result: dict[str, Any],
    visualization_request: str,
    bronze_dir: Path,
    silver_dir: Path,
    gold_dir: Path,
) -> dict[str, Any]:
    units = semantic_config.get("units", "unitless")
    axis_hint = semantic_config.get("axis_hint")
    expected_radius = semantic_config.get("expected_radius")
    voxel_size = float(semantic_config.get("voxel_size") or analysis_params.get("voxel_size") or 0.0)
    max_outlier_std = float(analysis_params.get("max_outlier_std", 2.0))
    min_cluster_points = int(analysis_params.get("min_cluster_points", 20))
    deviation_threshold = float(analysis_params.get("deviation_threshold", 0.02))
    slice_spacing = float(analysis_params.get("slice_spacing", voxel_size or 0.1))

    raw_copy_path = bronze_dir / Path(dataset_path).name
    shutil.copy2(dataset_path, raw_copy_path)

    cloud = load_point_cloud(dataset_path)
    cleaned_result = clean_point_cloud(cloud, voxel_size=voxel_size, max_outlier_std=max_outlier_std, neighbors=int(analysis_params.get("neighbors", 20)), estimate_normals=bool(analysis_params.get("estimate_normals", True)))
    cleaned_cloud = cleaned_result.get("artifacts", {}).get("point_cloud", cloud)

    fit_result = fit_pipe_cylinder(cleaned_cloud, axis_hint=axis_hint, expected_radius=expected_radius, units=units)
    pipe_fit = fit_result.get("artifacts", {}).get("pipe_fit")
    deviation_result: dict[str, Any] = {}
    dent_result: dict[str, Any] = {}
    ovality_result: dict[str, Any] = {}
    deviation_df = pd.DataFrame()
    dents_df = pd.DataFrame()
    dents_export_df = pd.DataFrame()
    ovality_df = pd.DataFrame()

    if pipe_fit is not None:
        deviation_result = compute_pipe_deviation_map(cleaned_cloud, pipe_fit, units=units)
        deviation_map = deviation_result.get("artifacts", {}).get("deviation_map")
        if deviation_map is not None:
            deviation_df = pd.DataFrame({
                "x": cleaned_cloud.points[:, 0],
                "y": cleaned_cloud.points[:, 1],
                "z": cleaned_cloud.points[:, 2],
                "axial_position": deviation_map.axial_positions,
                "angle_rad": deviation_map.angles,
                "angle_deg": np.degrees(deviation_map.angles),
                "deviation": deviation_map.deviations,
                "inward_depth": deviation_map.inward_depths,
                "outward_offset": deviation_map.outward_offsets,
            })
            dent_result = detect_pipe_dents(cleaned_cloud, deviation_map, deviation_threshold=deviation_threshold, min_cluster_points=min_cluster_points, severity_bands=analysis_params.get("severity_bands") or DEFAULT_SEVERITY_BANDS)
            dents_df = pd.DataFrame(dent_result.get("data", {}).get("dents", []))
            dents_export_df = dents_df.copy()
            if not dents_export_df.empty and "centroid" in dents_export_df.columns:
                dents_export_df["centroid"] = dents_export_df["centroid"].apply(lambda value: json.dumps(value) if isinstance(value, list) else value)
        ovality_result = measure_pipe_ovality(cleaned_cloud, pipe_fit, slice_spacing=slice_spacing, units=units)
        ovality_df = pd.DataFrame(ovality_result.get("data", {}).get("slices", []))

    point_cloud_sample_path = silver_dir / "point_cloud_sample.csv"
    pd.DataFrame(_sample_points(cleaned_cloud.points, max_points=4000), columns=["x", "y", "z"]).to_csv(point_cloud_sample_path, index=False)
    deviation_path = None
    if not deviation_df.empty:
        deviation_path = silver_dir / "deviation_map.csv"
        deviation_df.to_csv(deviation_path, index=False)

    dents_path = None
    if not dents_df.empty:
        dents_path = gold_dir / "dent_report.csv"
        dents_export_df.to_csv(dents_path, index=False)

    ovality_path = None
    if not ovality_df.empty:
        ovality_path = gold_dir / "ovality_profile.csv"
        ovality_df.to_csv(ovality_path, index=False)

    heatmap_bins = _build_pipe_heatmap_bins(deviation_df)
    heatmap_path = None
    if not heatmap_bins.empty:
        heatmap_path = gold_dir / "pipe_heatmap_bins.csv"
        heatmap_bins.to_csv(heatmap_path, index=False)

    axial_profile = _build_pipe_axial_profile(deviation_df)
    axial_profile_path = None
    if not axial_profile.empty:
        axial_profile_path = gold_dir / "pipe_axial_profile.csv"
        axial_profile.to_csv(axial_profile_path, index=False)

    db_path = gold_dir / "local_reporting.db"
    conn = sqlite3.connect(db_path)
    try:
        _build_pipeline_summary_table(cleaned_cloud, fit_result, dent_result, ovality_result, units).to_sql("pipeline_summary", conn, if_exists="replace", index=False)
        if not deviation_df.empty:
            deviation_df.to_sql("fact_pipe_deviation", conn, if_exists="replace", index=False)
        if not dents_df.empty:
            dents_export_df.to_sql("fact_dent_events", conn, if_exists="replace", index=False)
        if not ovality_df.empty:
            ovality_df.to_sql("fact_ovality_slices", conn, if_exists="replace", index=False)
        if not heatmap_bins.empty:
            heatmap_bins.to_sql("fact_pipe_heatmap_bins", conn, if_exists="replace", index=False)
        if not axial_profile.empty:
            axial_profile.to_sql("fact_pipe_axial_profile", conn, if_exists="replace", index=False)
    finally:
        conn.close()

    report_summary = {
        "question": question,
        "visualization_request": visualization_request,
        "pipeline_findings": analysis_result.get("insights", []),
        "human_review_required": analysis_result.get("explanation_layer", {}).get("human_review_required", []),
        "dent_count": int(len(dents_df)),
        "max_inward_depth": float(deviation_df["inward_depth"].max()) if not deviation_df.empty else 0.0,
        "artifact_paths": {
            "bronze": str(raw_copy_path.resolve()),
            "silver_point_cloud": str(point_cloud_sample_path.resolve()),
            "silver_deviation_map": str(deviation_path.resolve()) if deviation_path else None,
            "gold_dent_report": str(dents_path.resolve()) if dents_path else None,
            "gold_ovality_profile": str(ovality_path.resolve()) if ovality_path else None,
            "gold_heatmap_bins": str(heatmap_path.resolve()) if heatmap_path else None,
            "gold_axial_profile": str(axial_profile_path.resolve()) if axial_profile_path else None,
            "gold_database": str(db_path.resolve()),
        },
    }
    (gold_dir / "report_summary.json").write_text(json.dumps(report_summary, indent=2), encoding="utf-8")

    tables = ["pipeline_summary"]
    if not deviation_df.empty:
        tables.append("fact_pipe_deviation")
    if not dents_df.empty:
        tables.append("fact_dent_events")
    if not ovality_df.empty:
        tables.append("fact_ovality_slices")
    if not heatmap_bins.empty:
        tables.append("fact_pipe_heatmap_bins")
    if not axial_profile.empty:
        tables.append("fact_pipe_axial_profile")

    report_markdown = "\n".join([
        "# Local Pipeline Inspection Product",
        "",
        "## Bronze",
        f"- Raw scan copied to `{raw_copy_path}`.",
        "",
        "## Silver",
        f"- Sampled point cloud written to `{point_cloud_sample_path}`.",
        *([f"- Deviation map written to `{deviation_path}`."] if deviation_path else ["- Deviation map was not available."]),
        "",
        "## Gold",
        f"- SQLite reporting database written to `{db_path}`.",
        *([f"- Dent report written to `{dents_path}`."] if dents_path else ["- No dent report file was produced."]),
        *([f"- Ovality profile written to `{ovality_path}`."] if ovality_path else ["- No ovality profile file was produced."]),
        *([f"- Unwrapped heatmap bins written to `{heatmap_path}`."] if heatmap_path else []),
        *([f"- Axial profile written to `{axial_profile_path}`."] if axial_profile_path else []),
    ])

    return {
        "mode": "pipeline",
        "report_markdown": report_markdown,
        "layers": {
            "bronze": {"path": str(raw_copy_path.resolve()), "description": "Immutable raw copy of the uploaded scan."},
            "silver": {"path": str(point_cloud_sample_path.resolve()), "description": "Cleaned and derived geometry data for inspection."},
            "gold": {"path": str(db_path.resolve()), "description": "SQLite reporting layer with deviation, dent, ovality, and heatmap tables."},
        },
        "tables": tables,
        "visualizations": _build_pipeline_visualizations(deviation_df, dents_df, ovality_df, fit_result, visualization_request, units),
        "report_summary": report_summary,
    }


def _build_pipeline_summary_table(cleaned_cloud: Any, fit_result: dict[str, Any], dent_result: dict[str, Any], ovality_result: dict[str, Any], units: str) -> pd.DataFrame:
    fit_data = fit_result.get("data", {}) if fit_result else {}
    dents = dent_result.get("data", {}).get("dents", []) if dent_result else []
    max_ovality = ovality_result.get("data", {}).get("max_ovality", {}) if ovality_result else {}
    rows = [{"metric": "clean_point_count", "value": float(len(cleaned_cloud.points)), "notes": f"Cleaned point count in {units}."}]
    if fit_data:
        rows.append({"metric": "nominal_radius", "value": float(fit_data.get("radius", 0.0)), "notes": f"Nominal fitted radius in {units}."})
        rows.append({"metric": "fit_rmse", "value": float(fit_data.get("fit_rmse", 0.0)), "notes": f"Cylinder fit RMSE in {units}."})
    rows.append({"metric": "dent_count", "value": float(len(dents)), "notes": "Detected dent clusters."})
    if dents:
        rows.append({"metric": "max_dent_depth", "value": float(max(dent["depth"] for dent in dents)), "notes": f"Maximum dent depth in {units}."})
    if max_ovality:
        rows.append({"metric": "max_ovality", "value": float(max_ovality.get("ovality", 0.0)), "notes": "Maximum slice ovality ratio."})
    return pd.DataFrame(rows)


def _build_tabular_visualizations(
    df: pd.DataFrame,
    silver: pd.DataFrame,
    primary_metric: str | None,
    dimension: str | None,
    explanation_layer: dict[str, Any],
    visualization_request: str,
    tool_evidence: dict[str, dict[str, Any]],
    quality_issue_facts: pd.DataFrame,
) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    status_counts = silver["dq_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    plots.append({"id": "quality-status", "title": "Record quality status", "description": "Rows ready for downstream use versus rows requiring review.", "guide": _tabular_visual_guide("quality-status"), "plotly": _plotly_bar(status_counts["status"].tolist(), status_counts["count"].tolist(), "Record quality status", "#0c6b52")})

    missing_share = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
    plots.append({"id": "missingness", "title": "Missingness by field", "description": "Percentage of rows missing by column.", "guide": _tabular_visual_guide("missingness"), "plotly": _plotly_bar(missing_share.index.tolist(), missing_share.values.tolist(), "Missingness by field (%)", "#a96a2b")})

    validation = tool_evidence.get("validate_dataset", {})
    dimension_scores = []
    quality_dimensions = validation.get("quality_dimensions", {})
    if isinstance(quality_dimensions, dict):
        for key in ("completeness", "uniqueness"):
            if key in quality_dimensions:
                dimension_scores.append({"label": key.title(), "value": round(float(quality_dimensions[key]) * 100, 2)})
    freshness = tool_evidence.get("assess_freshness", {})
    if freshness:
        dimension_scores.append({"label": "Freshness age (days)", "value": round(float(freshness.get("freshness_age_days", 0.0)), 2)})
    if dimension_scores:
        plots.append({"id": "quality-dimensions", "title": "Quality dimension summary", "description": "Core quality dimensions used in the governed intake step.", "plotly": _plotly_bar([item["label"] for item in dimension_scores], [item["value"] for item in dimension_scores], "Quality dimension summary", "#375a7f")})

    if not quality_issue_facts.empty and not (len(quality_issue_facts) == 1 and quality_issue_facts.iloc[0]["issue_type"] == "none"):
        issue_pivot = quality_issue_facts.pivot_table(index="issue_type", columns="field_name", values="issue_value", aggfunc="sum", fill_value=0)
        plots.append({
            "id": "quality-issue-matrix",
            "title": "Quality issue matrix",
            "description": "Field-level issue counts and shares for the engineering handoff.",
            "plotly": {
                "data": [{
                    "type": "heatmap",
                    "x": issue_pivot.columns.tolist(),
                    "y": issue_pivot.index.tolist(),
                    "z": issue_pivot.values.round(4).tolist(),
                    "colorscale": "YlOrRd",
                }],
                "layout": {**_plotly_layout("Quality issue matrix"), "xaxis": {"title": "Field"}, "yaxis": {"title": "Issue type"}},
            },
        })

    standardization_candidates = tool_evidence.get("audit_standardization", {}).get("standardization_candidates", [])
    if standardization_candidates:
        lines = []
        for candidate in standardization_candidates[:4]:
            groups = "; ".join(
                f"{group.get('normalized_value')}: {', '.join(group.get('raw_variants', []))}"
                for group in candidate.get("variant_groups", [])[:2]
            )
            lines.append(f"- {candidate.get('column')}: {groups}")
        plots.append({"id": "standardization-playbook", "title": "Standardization playbook", "description": "Suggested label-normalization opportunities for silver-layer cleanup rules.", "markdown": "\n".join(lines)})

    collision_candidates = tool_evidence.get("detect_entity_collisions", {}).get("collision_candidates", [])
    if collision_candidates:
        lines = []
        for candidate in collision_candidates[:4]:
            lines.append(f"- {', '.join(candidate.get('match_columns', []))} -> {candidate.get('normalized_key')} ({candidate.get('row_count')} rows)")
        plots.append({"id": "entity-collisions", "title": "Duplicate-entity candidates", "description": "Potential duplicate groups that should be reviewed before deduplication rules are applied.", "markdown": "\n".join(lines)})

    if dimension and dimension in silver.columns and primary_metric and primary_metric in silver.columns:
        rollup = silver.assign(_metric=pd.to_numeric(silver[primary_metric], errors="coerce")).groupby(dimension, dropna=False)["_metric"].mean().sort_values(ascending=False).reset_index()
        plots.append({"id": "metric-by-dimension", "title": f"Average {primary_metric} by {dimension}", "description": "Business-facing rollup from the engineered silver layer.", "plotly": _plotly_bar(rollup[dimension].fillna("Unknown").astype(str).tolist(), rollup["_metric"].round(2).tolist(), f"Average {primary_metric} by {dimension}", "#375a7f")})

    if explanation_layer.get("human_review_required"):
        plots.append({"id": "human-review", "title": "Human review queue", "description": "Checklist carried forward from the analyst explanation layer.", "markdown": "\n".join(f"- {item}" for item in explanation_layer.get("human_review_required", []))})
    if visualization_request:
        plots.append({"id": "visualization-request", "title": "Business visualization request", "description": "Request captured before the engineering step.", "guide": _tabular_visual_guide("visualization-request"), "markdown": visualization_request})
    return plots

def _build_pipeline_visualizations(deviation_df: pd.DataFrame, dents_df: pd.DataFrame, ovality_df: pd.DataFrame, fit_result: dict[str, Any], visualization_request: str, units: str) -> list[dict[str, Any]]:
    plots: list[dict[str, Any]] = []
    pipe_fit = fit_result.get("artifacts", {}).get("pipe_fit") if fit_result else None

    if not deviation_df.empty:
        sampled = deviation_df.iloc[:: max(len(deviation_df) // 2500, 1)].copy()
        peak_inward = max(float(sampled["inward_depth"].max()), 1e-6)
        scatter_data = [{
            "type": "scatter3d",
            "mode": "markers",
            "x": sampled["x"].round(5).tolist(),
            "y": sampled["y"].round(5).tolist(),
            "z": sampled["z"].round(5).tolist(),
            "marker": {
                "size": 3,
                "color": sampled["inward_depth"].round(6).tolist(),
                "colorscale": [[0.0, "#e6eee9"], [0.15, "#f2d8a7"], [0.55, "#df7a39"], [1.0, "#8e1f1f"]],
                "cmin": 0,
                "cmax": peak_inward,
                "colorbar": {"title": f"Dent depth ({units})"},
            },
            "name": "Pipe wall",
        }]
        if not dents_df.empty and {"centroid", "dent_id"}.issubset(dents_df.columns):
            centroids = dents_df["centroid"].tolist()
            scatter_data.append({
                "type": "scatter3d",
                "mode": "markers+text",
                "x": [item[0] for item in centroids],
                "y": [item[1] for item in centroids],
                "z": [item[2] for item in centroids],
                "text": dents_df["dent_id"].astype(str).tolist(),
                "textposition": "top center",
                "marker": {"size": 8, "color": "#0f172a", "symbol": "diamond"},
                "name": "Detected dents",
            })
        plots.append({
            "id": "pipe-3d-view",
            "title": "3D dent-intensity view",
            "description": "Healthy pipe wall stays pale while inward dent zones become warmer and darker.",
            "guide": _pipeline_visual_guide("pipe-3d-view", units, dents_df),
            "plotly": {"data": scatter_data, "layout": _plotly_layout("3D dent-intensity view")},
        })

        heatmap_bins = _build_pipe_heatmap_bins(deviation_df)
        if not heatmap_bins.empty:
            pivot = heatmap_bins.pivot_table(index="angle_bin_deg", columns="axial_bin", values="peak_inward_depth", aggfunc="max", fill_value=0.0).sort_index(ascending=False)
            peak_heat = max(float(heatmap_bins["peak_inward_depth"].max()), 1e-6)
            heatmap_data: list[dict[str, Any]] = [{
                "type": "heatmap",
                "x": [round(float(value), 4) for value in pivot.columns.tolist()],
                "y": [round(float(value), 2) for value in pivot.index.tolist()],
                "z": pivot.values.round(6).tolist(),
                "zmin": 0,
                "zmax": peak_heat,
                "colorscale": [[0.0, "#f7f4ed"], [0.12, "#efe3ca"], [0.35, "#f4a259"], [1.0, "#7f1d1d"]],
                "colorbar": {"title": f"Dent depth ({units})"},
                "hovertemplate": "Axial %{x}<br>Angle %{y}<br>Dent depth %{z}<extra></extra>",
            }]
            if not dents_df.empty and {"axial_center", "angle_center_deg", "dent_id"}.issubset(dents_df.columns):
                overlay = dents_df.dropna(subset=["axial_center", "angle_center_deg"]).copy()
                if not overlay.empty:
                    heatmap_data.append({
                        "type": "scatter",
                        "mode": "markers+text",
                        "x": overlay["axial_center"].round(4).tolist(),
                        "y": overlay["angle_center_deg"].round(2).tolist(),
                        "text": overlay["dent_id"].astype(str).tolist(),
                        "textposition": "top center",
                        "marker": {"size": 9, "color": "#111827", "symbol": "x"},
                        "name": "Dent center",
                    })
            plots.append({
                "id": "pipe-unwrapped",
                "title": "Unwrapped dent map",
                "description": "The pipe wall is flattened into a rectangle so coherent dents read as hotspots instead of scattered points.",
                "guide": _pipeline_visual_guide("pipe-unwrapped", units, dents_df),
                "plotly": {
                    "data": heatmap_data,
                    "layout": {
                        **_plotly_layout("Unwrapped dent map"),
                        "xaxis": {"title": f"Axial position ({units})"},
                        "yaxis": {"title": "Circumferential angle (deg)", "range": [-180, 180]},
                    },
                },
            })

        axial_profile = _build_pipe_axial_profile(deviation_df)
        if not axial_profile.empty:
            plots.append({
                "id": "pipe-axial-profile",
                "title": "Axial dent profile",
                "description": "Peak inward depth by axial slice shows where the strongest deformation sits along the pipe length.",
                "guide": _pipeline_visual_guide("pipe-axial-profile", units, dents_df),
                "plotly": {
                    "data": [
                        {
                            "type": "scatter",
                            "mode": "lines",
                            "x": axial_profile["axial_bin"].round(5).tolist(),
                            "y": axial_profile["peak_inward_depth"].round(6).tolist(),
                            "line": {"color": "#8e1f1f", "width": 3},
                            "fill": "tozeroy",
                            "fillcolor": "rgba(142,31,31,0.12)",
                            "name": "Peak dent depth",
                        },
                        {
                            "type": "scatter",
                            "mode": "lines",
                            "x": axial_profile["axial_bin"].round(5).tolist(),
                            "y": axial_profile["mean_inward_depth"].round(6).tolist(),
                            "line": {"color": "#375a7f", "width": 2},
                            "name": "Mean dent depth",
                        },
                    ],
                    "layout": {
                        **_plotly_layout("Axial dent profile"),
                        "xaxis": {"title": f"Axial position ({units})"},
                        "yaxis": {"title": f"Dent depth ({units})", "rangemode": "tozero"},
                    },
                },
            })

    if pipe_fit is not None:
        mid_axial = float(np.median(pipe_fit.axial_positions))
        axial_band = max(float(np.std(pipe_fit.axial_positions) * 0.15), 1e-6)
        mask = (pipe_fit.axial_positions >= mid_axial - axial_band) & (pipe_fit.axial_positions <= mid_axial + axial_band)
        if int(mask.sum()) > 10:
            slice_inward = np.clip(pipe_fit.radius - pipe_fit.radial_distances[mask], 0.0, None)
            cross_x = (pipe_fit.radial_distances[mask] * np.cos(pipe_fit.angles[mask])).round(5)
            cross_y = (pipe_fit.radial_distances[mask] * np.sin(pipe_fit.angles[mask])).round(5)
            circle_theta = np.linspace(0, 2 * np.pi, 180)
            plots.append({
                "id": "pipe-cross-section",
                "title": "Cross-section at mid-span",
                "description": "The nominal circle is overlaid on the measured slice so inward loss shows up as a notch or flattened arc.",
                "guide": _pipeline_visual_guide("pipe-cross-section", units, dents_df),
                "plotly": {
                    "data": [
                        {
                            "type": "scattergl",
                            "mode": "markers",
                            "x": cross_x.tolist(),
                            "y": cross_y.tolist(),
                            "marker": {
                                "size": 6,
                                "color": slice_inward.round(6).tolist(),
                                "colorscale": [[0.0, "#dbe7e2"], [0.25, "#f4d7a3"], [1.0, "#8e1f1f"]],
                                "cmin": 0,
                                "cmax": max(float(slice_inward.max()), 1e-6),
                                "colorbar": {"title": f"Dent depth ({units})"},
                            },
                            "name": "Observed slice",
                        },
                        {
                            "type": "scatter",
                            "mode": "lines",
                            "x": (pipe_fit.radius * np.cos(circle_theta)).round(5).tolist(),
                            "y": (pipe_fit.radius * np.sin(circle_theta)).round(5).tolist(),
                            "line": {"color": "#c18328", "width": 2},
                            "name": "Nominal circle",
                        },
                    ],
                    "layout": {
                        **_plotly_layout("Cross-section at mid-span"),
                        "xaxis": {"title": units, "scaleanchor": "y"},
                        "yaxis": {"title": units},
                    },
                },
            })

    if not ovality_df.empty:
        plots.append({
            "id": "ovality-profile",
            "title": "Ovality profile",
            "description": "Tracks broad out-of-roundness separately from localized dents.",
            "guide": _pipeline_visual_guide("ovality-profile", units, dents_df),
            "plotly": {
                "data": [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": ovality_df["axial_start"].round(5).tolist(),
                    "y": ovality_df["ovality"].round(6).tolist(),
                    "line": {"color": "#375a7f", "width": 3},
                    "name": "Ovality",
                }],
                "layout": {
                    **_plotly_layout("Ovality profile"),
                    "xaxis": {"title": f"Axial start ({units})"},
                    "yaxis": {"title": "Ovality ratio", "rangemode": "tozero"},
                },
            },
        })

    if not dents_df.empty:
        ordered = dents_df.sort_values("depth", ascending=False).reset_index(drop=True)
        severity_colors = {"minor": "#d08c2a", "moderate": "#c05621", "severe": "#8e1f1f"}
        plots.append({
            "id": "dent-summary",
            "title": "Detected dent depths",
            "description": "Ranks dents by depth so the review queue starts with the most severe feature.",
            "guide": _pipeline_visual_guide("dent-summary", units, ordered),
            "plotly": {
                "data": [{
                    "type": "bar",
                    "x": ordered["dent_id"].astype(str).tolist(),
                    "y": ordered["depth"].round(5).tolist(),
                    "marker": {"color": [severity_colors.get(value, "#8e1f1f") for value in ordered["severity"].tolist()]},
                    "text": ordered["severity"].tolist(),
                    "textposition": "outside",
                    "name": "Dent depth",
                }],
                "layout": {**_plotly_layout("Detected dent depths"), "yaxis": {"title": f"Dent depth ({units})", "rangemode": "tozero"}},
            },
        })
        plots.append({
            "id": "dent-risk-matrix",
            "title": "Dent risk matrix",
            "description": "Combines defect depth and axial span so large deep dents stand apart from small cosmetic features.",
            "guide": _pipeline_visual_guide("dent-risk-matrix", units, ordered),
            "plotly": {
                "data": [{
                    "type": "scatter",
                    "mode": "markers+text",
                    "x": ordered["axial_length"].round(5).tolist(),
                    "y": ordered["depth"].round(5).tolist(),
                    "text": ordered["dent_id"].astype(str).tolist(),
                    "textposition": "top center",
                    "marker": {
                        "size": (ordered["circumferential_width"].fillna(0).abs() * 12 + 14).round(2).tolist(),
                        "color": [severity_colors.get(value, "#8e1f1f") for value in ordered["severity"].tolist()],
                        "opacity": 0.82,
                    },
                    "name": "Dent cluster",
                }],
                "layout": {
                    **_plotly_layout("Dent risk matrix"),
                    "xaxis": {"title": f"Axial length ({units})", "rangemode": "tozero"},
                    "yaxis": {"title": f"Dent depth ({units})", "rangemode": "tozero"},
                },
            },
        })
        markdown_lines = [
            f"- {row['dent_id']}: {row['severity']} / {row['review_priority']} review, depth {row['depth']:.4f} {units}, axial span {row['axial_length']:.4f} {units}, angle {row.get('angle_center_deg', 0.0):.1f} deg"
            for _, row in ordered.head(6).iterrows()
        ]
        plots.append({
            "id": "dent-readout",
            "title": "Dent readout",
            "description": "Condenses the key defect metrics into an engineering review queue.",
            "guide": _pipeline_visual_guide("dent-readout", units, ordered),
            "markdown": "\n".join(markdown_lines),
        })

    if visualization_request:
        plots.append({
            "id": "visualization-request",
            "title": "Business visualization request",
            "description": "Request captured before the visualization step.",
            "guide": _tabular_visual_guide("visualization-request"),
            "markdown": visualization_request,
        })
    return plots


def _build_pipe_heatmap_bins(deviation_df: pd.DataFrame, axial_bins: int = 54, angle_bins: int = 72) -> pd.DataFrame:
    if deviation_df.empty:
        return pd.DataFrame()
    working = deviation_df.copy()
    if "inward_depth" not in working.columns:
        working["inward_depth"] = np.clip(-working["deviation"], 0.0, None)
    working["axial_bin_id"] = pd.cut(working["axial_position"], bins=axial_bins, labels=False, include_lowest=True)
    working["angle_bin_id"] = pd.cut(working["angle_deg"], bins=angle_bins, labels=False, include_lowest=True)
    grouped = (
        working.groupby(["axial_bin_id", "angle_bin_id"], dropna=False)
        .agg(
            axial_bin=("axial_position", "mean"),
            angle_bin_deg=("angle_deg", "mean"),
            peak_inward_depth=("inward_depth", "max"),
            mean_inward_depth=("inward_depth", "mean"),
            min_deviation=("deviation", "min"),
            point_count=("inward_depth", "size"),
        )
        .reset_index()
        .dropna(subset=["axial_bin_id", "angle_bin_id"])
        .sort_values(["angle_bin_deg", "axial_bin"])
    )
    return grouped


def _build_pipe_axial_profile(deviation_df: pd.DataFrame, axial_bins: int = 72) -> pd.DataFrame:
    if deviation_df.empty:
        return pd.DataFrame()
    working = deviation_df.copy()
    if "inward_depth" not in working.columns:
        working["inward_depth"] = np.clip(-working["deviation"], 0.0, None)
    working["axial_bucket"] = pd.cut(working["axial_position"], bins=axial_bins, labels=False, include_lowest=True)
    grouped = (
        working.groupby("axial_bucket")
        .agg(
            axial_bin=("axial_position", "mean"),
            peak_inward_depth=("inward_depth", "max"),
            mean_inward_depth=("inward_depth", "mean"),
            point_count=("inward_depth", "size"),
        )
        .reset_index(drop=True)
    )
    return grouped


def _visual_guide(summary: str, how_to_read: list[str], look_for: list[str], healthy_example: str, concern_example: str, engineering_use: str, illustration: str) -> dict[str, Any]:
    return {
        "summary": summary,
        "how_to_read": how_to_read,
        "look_for": look_for,
        "healthy_example": healthy_example,
        "concern_example": concern_example,
        "engineering_use": engineering_use,
        "illustration": illustration,
    }


def _tabular_visual_guide(viz_id: str) -> dict[str, Any]:
    guides = {
        "quality-status": _visual_guide(
            "Shows how many rows are ready for use versus how many should be reviewed before publishing.",
            ["The x-axis lists row-status buckets.", "Higher bars mean more rows in that state."],
            ["A large review_required bar means the intake step found material quality concerns.", "A dominant ready bar means the silver layer is close to reportable."],
            "Most rows are in the ready bucket, with only a small review tail.",
            "A large review bucket suggests missing values, invalid records, or duplicate/entity issues still need attention.",
            "Use it to decide whether engineering should proceed automatically or pause for analyst review.",
            "bar-status",
        ),
        "missingness": _visual_guide(
            "Ranks fields by missing-value percentage.",
            ["Each bar is a field.", "Taller bars mean more missing records in that field."],
            ["High-cardinality business keys should rarely be missing.", "Critical metric fields with large missing share often need imputation or business follow-up."],
            "Most important fields have low or near-zero missingness.",
            "One or two operational fields spike high above the rest, indicating input pipeline or source issues.",
            "Use it to prioritise field-level remediation rules in the silver layer.",
            "bar-missingness",
        ),
        "quality-issue-matrix": _visual_guide(
            "Pivots issue types against fields so you can see where the quality load is concentrated.",
            ["Rows are issue types, columns are fields.", "Darker cells mean more evidence of that issue in that field."],
            ["Single dark columns show field-specific problems.", "Broad dark rows show cross-dataset patterns like missingness or schema mismatch."],
            "Only a few isolated cells are emphasized.",
            "Whole bands or clusters light up, meaning the dataset has systemic issues rather than isolated defects.",
            "Use it as the handoff map for engineer cleanup rules and stakeholder review.",
            "heatmap-matrix",
        ),
        "visualization-request": _visual_guide(
            "This preserves the stakeholder's reporting request so engineering and visualization stay aligned with business intent.",
            ["Read it as the contract for the last stage of the workflow."],
            ["Check whether the produced visuals actually answer the request."],
            "The request is specific and matches the final charts.",
            "The request is vague or mismatched, which usually produces the wrong report emphasis.",
            "Use it as traceability between the business question and the delivered report.",
            "request-note",
        ),
    }
    return guides.get(viz_id, _visual_guide(
        "Explains how to interpret this visualization in the decision workflow.",
        ["Read the axes and compare the strongest values first."],
        ["Look for outliers, concentrations, and any pattern that matches the business question."],
        "The chart supports the expected business pattern with only small deviations.",
        "The chart shows outliers or concentrations that need review before reporting.",
        "Use it to connect engineered outputs back to a stakeholder question.",
        "generic-chart",
    ))


def _pipeline_visual_guide(viz_id: str, units: str, context: pd.DataFrame | None = None) -> dict[str, Any]:
    dent_count = int(len(context)) if isinstance(context, pd.DataFrame) else 0
    guides = {
        "pipe-3d-view": _visual_guide(
            f"The pipe surface is shown in 3D and colored by inward dent depth in {units}; pale regions are nominal and darker red regions are deformed inward.",
            ["Rotate the view to inspect where warm colors concentrate on the pipe wall.", "The darker the color, the deeper the inward loss from the nominal fitted cylinder.", "Black diamond markers indicate detected dent centers."],
            ["Look for isolated warm patches instead of random speckling.", "If the color is uniformly pale, the pipe is close to nominal.", "If one side lights up strongly, that wall segment needs engineering attention."],
            "Healthy pipe wall appears mostly pale with only subtle texture from scan noise or mild ovality.",
            "A concerning scan shows one or more compact red patches that remain visible as you rotate the pipe, indicating a real local dent rather than random noise.",
            f"Use it for rapid triage before drilling into the unwrapped map and cross-section. Current run has {dent_count} detected dent cluster(s).",
            "pipe-3d-view",
        ),
        "pipe-unwrapped": _visual_guide(
            f"This flattens the cylindrical wall into a rectangle. X is axial position, Y is angle around the circumference, and color is inward dent depth in {units}.",
            ["Read left-to-right as pipe length and top-to-bottom as angle around the pipe.", "Most of the map should stay near zero and visually quiet.", "Hotspots mark coherent inward defects."],
            ["Small isolated islands usually mean localized dents.", "Wide bands suggest broader deformation or an area worth checking for fit issues.", "If the map is only speckled and never forms a coherent hotspot, treat it as noise or fit review rather than a defect call."],
            "A healthy map is nearly uniform with only light shading and no strong hotspots.",
            "A concerning map shows one or more dense red hotspots or streaks, which is what a real dent or deformed patch looks like when the pipe wall is unwrapped.",
            "Use it to localize exactly where along the length and around the circumference a field crew or engineer should inspect next.",
            "pipe-unwrapped",
        ),
        "pipe-axial-profile": _visual_guide(
            f"This profile collapses the 3D scan into dent intensity by axial slice, so peaks show where the strongest inward deformation sits along the pipe length in {units}.",
            ["The red line is the peak dent depth in each slice.", "The blue line is the average inward depth in that slice.", "The strongest peaks are the slices to cross-check against the heatmap and dent table."],
            ["Sharp peaks usually map to localized dents.", "A wide elevated region suggests broader deformation.", "A flat near-zero profile usually means no material dent indication."],
            "Healthy pipe stays close to zero for most of the run with only minor ripples.",
            "A concerning pipe shows one or more clear peaks rising above the local baseline, which means the defect has a real axial footprint.",
            "Use it to decide where to place cross-sections and how much pipe length a repair or reinspection window should cover.",
            "pipe-axial-profile",
        ),
        "pipe-cross-section": _visual_guide(
            "This compares a measured mid-span slice against the nominal fitted circle. It is the easiest way to show a real inward notch or flattened wall segment.",
            ["Amber line is the nominal circle.", "Colored markers are the observed slice.", "Markers that fall inside the nominal circle indicate inward loss."],
            ["A slight ellipse means ovality but not necessarily a localized dent.", "A local notch or flattened arc is the clearest geometric dent signal.", "If the whole slice is offset, check fit quality before overinterpreting."],
            "Healthy or mildly oval pipe stays close to the nominal circle with smooth shape variation.",
            "A concerning section shows a visible inward notch or cluster of points clearly inside the nominal circle over a limited arc length.",
            "Use it as the visual proof when explaining a dent to a non-specialist stakeholder.",
            "pipe-cross-section",
        ),
        "ovality-profile": _visual_guide(
            "Tracks broad out-of-roundness along the pipe. This is for shape distortion, not localized denting.",
            ["Each point is a slice along the pipe length.", "Higher values mean more difference between the largest and smallest measured radius in that slice."],
            ["A smooth low profile means shape is stable.", "A broad rise without sharp dent hotspots suggests ovality or overall deformation rather than a single local dent."],
            "Healthy pipe keeps ovality low and fairly stable along the run.",
            "A concerning segment shows sustained elevated ovality, which can indicate broader deformation even if dent clustering is limited.",
            "Use it to separate local dents from more distributed out-of-round pipe conditions.",
            "ovality-profile",
        ),
        "dent-summary": _visual_guide(
            f"Ranks the detected dents by depth in {units} so the review queue starts with the most severe feature.",
            ["Each bar is one detected dent cluster.", "Bar height is depth and bar color reflects severity."],
            ["Start with the tallest, darkest bar first.", "Compare bar order with the readout and risk matrix to understand severity and footprint together."],
            "Healthy or lightly affected scans show no bars or only very small minor bars.",
            "A concerning scan shows one or more dominant bars that stand far above the rest, indicating a prioritized engineering review item.",
            "Use it to communicate defect priority quickly to a business stakeholder or engineer.",
            "dent-summary",
        ),
        "dent-risk-matrix": _visual_guide(
            f"Plots defect depth against axial span. Bigger markers cover more circumferential width, so large deep dents immediately separate from small cosmetic features.",
            ["Right means longer defect span.", "Higher means deeper defect.", "Larger markers indicate wider circumferential coverage."],
            ["Top-right markers are the priority cases.", "Small low-left markers are often minor monitor-only items.", "Use severity color and label together with geometry."],
            "Healthy or low-risk scans cluster near the lower-left corner with small markers.",
            "A concerning scan has one or more markers moving up and to the right, showing a dent that is both deep and spatially significant.",
            "Use it to support prioritisation rather than just detection.",
            "dent-risk-matrix",
        ),
        "dent-readout": _visual_guide(
            "This is the compact engineering queue: each line summarizes depth, span, angle, and review priority for a detected dent.",
            ["Read top to bottom as a review list.", "Use angle and axial span to link the text back to the unwrapped map and 3D view."],
            ["Immediate or near-term priority items should align with the strongest hotspots and tallest bars.", "If the readout lists only monitor items, the scan is likely low severity."],
            "Healthy scans return no dent lines or only low-priority monitor items.",
            "A concerning scan produces one or more near-term or immediate review lines with clear depth and span values.",
            "Use it as the final handoff from analytics to engineering review.",
            "dent-readout",
        ),
    }
    return guides.get(viz_id, _tabular_visual_guide(viz_id))
def _plotly_bar(x: list[Any], y: list[Any], title: str, color: str) -> dict[str, Any]:
    return {"data": [{"type": "bar", "x": x, "y": y, "marker": {"color": color}, "name": title}], "layout": _plotly_layout(title)}


def _plotly_layout(title: str) -> dict[str, Any]:
    return {"title": {"text": title}, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(255,252,246,0.96)", "margin": {"l": 48, "r": 20, "t": 48, "b": 48}, "font": {"family": "Georgia, serif", "color": "#211912"}}


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    step = max(len(points) // max_points, 1)
    return points[::step][:max_points]




























