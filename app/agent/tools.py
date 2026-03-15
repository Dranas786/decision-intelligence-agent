from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.analytics.anomalies import run_anomaly_detection
from app.analytics.bayes import bayesian_ab_test
from app.analytics.common import categorical_columns, find_time_column, first_numeric, numeric_columns
from app.analytics.correlations import scan_correlations
from app.analytics.data_quality import (
    assess_freshness,
    audit_schema_contract,
    audit_standardization,
    detect_entity_collisions,
)
from app.analytics.finance import (
    backtest_signal,
    calculate_returns,
    detect_volume_spikes,
    measure_drawdown,
    measure_risk,
    optimize_portfolio,
)
from app.analytics.forecasting import forecast_metric
from app.analytics.healthcare import (
    analyze_length_of_stay,
    compare_cohorts,
    compute_readmission_rate,
    estimate_treatment_effect,
    survival_risk_analysis,
)
from app.analytics.pipeline_3d import (
    DEFAULT_SEVERITY_BANDS,
    PointCloudData,
    clean_point_cloud,
    compute_pipe_deviation_map,
    detect_pipe_dents,
    fit_pipe_cylinder,
    load_point_cloud,
    measure_pipe_ovality,
    profile_point_cloud,
)
from app.analytics.profiling import build_profile_summary
from app.analytics.regression import fit_driver_regression
from app.analytics.segmentation import run_segment_analysis
from app.analytics.validation import validate_dataset


PIPELINE_TOOL_NAMES = {
    "profile_point_cloud",
    "clean_point_cloud",
    "fit_pipe_cylinder",
    "compute_pipe_deviation_map",
    "detect_pipe_dents",
    "measure_pipe_ovality",
}


TABULAR_GENERAL_TOOLS = {
    "validate_dataset",
    "profile_table",
    "audit_schema_contract",
    "assess_freshness",
    "audit_standardization",
    "detect_entity_collisions",
    "detect_anomalies",
    "segment_drivers",
    "scan_correlations",
    "fit_driver_regression",
    "forecast_metric",
    "bayesian_ab_test",
}


def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError("Unsupported file type. Use CSV or Parquet.")



def load_resource(dataset_path: str, domain: str | None = None) -> pd.DataFrame | PointCloudData:
    if domain == "pipeline":
        return load_point_cloud(dataset_path)
    return load_dataset(dataset_path)



def summarize_resource(resource: pd.DataFrame | PointCloudData) -> dict[str, Any]:
    if isinstance(resource, pd.DataFrame):
        return {
            "resource_kind": "table",
            "columns": resource.columns.tolist(),
            "row_count": len(resource),
        }

    min_bounds = resource.points.min(axis=0)
    max_bounds = resource.points.max(axis=0)
    return {
        "resource_kind": "point_cloud",
        "columns": [],
        "row_count": int(len(resource.points)),
        "point_count": int(len(resource.points)),
        "has_normals": resource.normals is not None,
        "bounds": {"min": min_bounds.tolist(), "max": max_bounds.tolist()},
    }



def _preferred_metric(df: pd.DataFrame, semantic_config: dict[str, Any]) -> str | None:
    measures = semantic_config.get("measures") or []
    primary_metric = semantic_config.get("primary_metric")
    for candidate in [primary_metric, *(measure.get("name") if isinstance(measure, dict) else measure for measure in measures)]:
        if candidate and candidate in df.columns:
            return candidate
    return first_numeric(df)



def _preferred_segment(df: pd.DataFrame, semantic_config: dict[str, Any]) -> str | None:
    dimensions = semantic_config.get("dimensions") or []
    for candidate in dimensions:
        if isinstance(candidate, dict):
            candidate = candidate.get("name")
        if candidate and candidate in df.columns:
            return candidate
    time_col = semantic_config.get("time_col")
    for column in categorical_columns(df):
        if column != time_col:
            return column
    return None



def _resolve_general_feature_cols(df: pd.DataFrame, target_col: str | None) -> list[str]:
    if not target_col:
        return []
    return [column for column in numeric_columns(df) if column != target_col][:5]



def _resolve_pipeline_args(
    tool_name: str,
    semantic_config: dict[str, Any],
    analysis_params: dict[str, Any],
) -> dict[str, Any]:
    units = semantic_config.get("units", "unitless")
    axis_hint = semantic_config.get("axis_hint")
    expected_radius = semantic_config.get("expected_radius")
    voxel_size = semantic_config.get("voxel_size", analysis_params.get("voxel_size", 0.0))

    if tool_name == "profile_point_cloud":
        return {"units": units}
    if tool_name == "clean_point_cloud":
        return {
            "voxel_size": float(voxel_size or 0.0),
            "max_outlier_std": float(analysis_params.get("max_outlier_std", 2.0)),
            "neighbors": int(analysis_params.get("neighbors", 20)),
            "estimate_normals": bool(analysis_params.get("estimate_normals", True)),
        }
    if tool_name == "fit_pipe_cylinder":
        return {
            "axis_hint": axis_hint,
            "expected_radius": expected_radius,
            "units": units,
        }
    if tool_name == "compute_pipe_deviation_map":
        return {"units": units}
    if tool_name == "detect_pipe_dents":
        return {
            "deviation_threshold": float(analysis_params.get("deviation_threshold", 0.02)),
            "min_cluster_points": int(analysis_params.get("min_cluster_points", 20)),
            "severity_bands": analysis_params.get("severity_bands") or DEFAULT_SEVERITY_BANDS,
        }
    if tool_name == "measure_pipe_ovality":
        return {
            "slice_spacing": float(analysis_params.get("slice_spacing", voxel_size or 0.1 or 0.1)),
            "units": units,
        }
    return {}



def _resolve_tabular_args(
    tool_name: str,
    df: pd.DataFrame,
    semantic_config: dict[str, Any],
    analysis_params: dict[str, Any],
) -> dict[str, Any]:
    time_col = semantic_config.get("time_col") or find_time_column(df)
    entity_col = semantic_config.get("entity_col")
    metric_col = _preferred_metric(df, semantic_config)
    segment_col = _preferred_segment(df, semantic_config)

    if tool_name == "validate_dataset":
        required_columns = semantic_config.get("required_columns") or [
            column for column in [time_col, entity_col, metric_col] if column
        ]
        return {
            "required_columns": required_columns,
            "unique_subset": semantic_config.get("unique_subset") or ([entity_col, time_col] if entity_col and time_col else []),
            "missing_threshold": analysis_params.get("missing_threshold", 0.2),
        }
    if tool_name == "profile_table":
        return {}
    if tool_name == "audit_schema_contract":
        required_columns = semantic_config.get("required_columns") or [
            column for column in [time_col, entity_col, metric_col] if column
        ]
        return {
            "required_columns": required_columns,
            "expected_types": semantic_config.get("expected_types") or {},
        }
    if tool_name == "assess_freshness":
        return {
            "time_col": time_col,
            "warn_after_days": analysis_params.get("warn_after_days", 2.0),
            "error_after_days": analysis_params.get("error_after_days", 7.0),
        }
    if tool_name == "audit_standardization":
        return {
            "columns": semantic_config.get("standardization_columns") or categorical_columns(df)[:6],
            "top_n": analysis_params.get("top_n", 5),
        }
    if tool_name == "detect_entity_collisions":
        return {
            "entity_columns": semantic_config.get("entity_columns") or [column for column in [entity_col, segment_col] if column],
            "min_group_size": analysis_params.get("min_group_size", 2),
            "top_n": analysis_params.get("top_n", 5),
        }
    if tool_name == "detect_anomalies":
        return {
            "metric_col": metric_col,
            "date_col": time_col,
            "z_threshold": analysis_params.get("z_threshold", 2.0),
            "rolling_window": analysis_params.get("rolling_window", 5),
            "period": analysis_params.get("period"),
        }
    if tool_name == "segment_drivers":
        return {
            "metric_col": metric_col,
            "segment_col": segment_col,
            "top_n": analysis_params.get("top_n", 5),
            "date_col": time_col,
        }
    if tool_name == "scan_correlations":
        return {
            "target_col": metric_col,
            "feature_cols": _resolve_general_feature_cols(df, metric_col),
            "top_n": analysis_params.get("top_n", 5),
        }
    if tool_name == "fit_driver_regression":
        return {
            "target_col": metric_col,
            "feature_cols": _resolve_general_feature_cols(df, metric_col),
        }
    if tool_name == "forecast_metric":
        return {
            "metric_col": metric_col,
            "date_col": time_col,
            "periods": analysis_params.get("forecast_periods", 7),
        }
    if tool_name == "bayesian_ab_test":
        return {
            "outcome_col": semantic_config.get("outcome_col") or metric_col,
            "variant_col": semantic_config.get("variant_col") or semantic_config.get("group_col") or segment_col,
            "control_label": analysis_params.get("control_label"),
            "treatment_label": analysis_params.get("treatment_label"),
        }
    if tool_name == "calculate_returns":
        return {
            "price_col": semantic_config.get("price_col") or metric_col or "close",
            "entity_col": entity_col,
            "simple": analysis_params.get("simple_returns", True),
        }
    if tool_name == "measure_risk":
        return {
            "price_col": semantic_config.get("price_col") or metric_col or "close",
            "benchmark_col": semantic_config.get("benchmark_col"),
            "risk_free_rate": analysis_params.get("risk_free_rate", 0.0),
        }
    if tool_name == "measure_drawdown":
        return {"price_col": semantic_config.get("price_col") or metric_col or "close"}
    if tool_name == "detect_volume_spikes":
        return {
            "volume_col": semantic_config.get("volume_col") or "volume",
            "date_col": time_col,
            "z_threshold": analysis_params.get("z_threshold", 2.0),
        }
    if tool_name == "optimize_portfolio":
        return {
            "entity_col": entity_col or "ticker",
            "price_col": semantic_config.get("price_col") or metric_col or "close",
            "date_col": time_col or "date",
        }
    if tool_name == "backtest_signal":
        return {
            "price_col": semantic_config.get("price_col") or metric_col or "close",
            "signal_col": semantic_config.get("signal_col") or "signal",
        }
    if tool_name == "compute_readmission_rate":
        return {
            "patient_id_col": semantic_config.get("patient_id_col") or entity_col or "patient_id",
            "admission_date_col": semantic_config.get("admission_date_col") or time_col or "admission_date",
            "window_days": analysis_params.get("window_days", 30),
        }
    if tool_name == "compare_cohorts":
        return {
            "cohort_col": semantic_config.get("cohort_col") or segment_col or "cohort",
            "outcome_col": semantic_config.get("outcome_col") or metric_col,
        }
    if tool_name == "analyze_length_of_stay":
        return {
            "admission_date_col": semantic_config.get("admission_date_col") or time_col or "admission_date",
            "discharge_date_col": semantic_config.get("discharge_date_col") or "discharge_date",
            "cohort_col": semantic_config.get("cohort_col") or segment_col,
        }
    if tool_name == "survival_risk_analysis":
        return {
            "duration_col": semantic_config.get("duration_col") or "duration_days",
            "event_col": semantic_config.get("event_col") or "event",
            "cohort_col": semantic_config.get("cohort_col") or segment_col,
        }
    if tool_name == "estimate_treatment_effect":
        return {
            "treatment_col": semantic_config.get("treatment_col") or semantic_config.get("group_col") or segment_col or "treatment",
            "outcome_col": semantic_config.get("outcome_col") or metric_col,
        }
    return {}



def _resolve_args(
    tool_name: str,
    resource: pd.DataFrame | PointCloudData,
    semantic_config: dict[str, Any],
    analysis_params: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(resource, PointCloudData):
        return _resolve_pipeline_args(tool_name, semantic_config, analysis_params)
    return _resolve_tabular_args(tool_name, resource, semantic_config, analysis_params)



def _args_are_usable(resource: pd.DataFrame | PointCloudData, args: dict[str, Any]) -> tuple[bool, list[str]]:
    if not isinstance(resource, pd.DataFrame):
        return True, []

    invalid: list[str] = []
    for key, value in args.items():
        if key.endswith("_col") and value and value not in resource.columns:
            invalid.append(f"{key}='{value}'")
        if key.endswith("_cols") and value:
            missing = [column for column in value if column not in resource.columns]
            if missing:
                invalid.append(f"{key} missing {missing}")
    return (len(invalid) == 0, invalid)



def _run_profile(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return build_profile_summary(resource)



def _run_validate(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return validate_dataset(resource, **args)



def _run_schema_contract(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return audit_schema_contract(resource, **args)



def _run_freshness(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return assess_freshness(resource, **args)



def _run_standardization(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return audit_standardization(resource, **args)



def _run_entity_collisions(resource: pd.DataFrame, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    return detect_entity_collisions(resource, **args)



def _run_profile_point_cloud(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("raw_point_cloud", resource)
    return profile_point_cloud(cloud, **args)



def _run_clean_point_cloud(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("raw_point_cloud", resource)
    return clean_point_cloud(cloud, **args)



def _run_fit_pipe_cylinder(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("point_cloud", state.get("raw_point_cloud", resource))
    return fit_pipe_cylinder(cloud, **args)



def _run_compute_pipe_deviation_map(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("point_cloud", state.get("raw_point_cloud", resource))
    pipe_fit = state.get("pipe_fit")
    if pipe_fit is None:
        return {
            "insights": ["Deviation map skipped because no pipe fit was available."],
            "insight_objects": [],
            "diagnostics": ["Run fit_pipe_cylinder before compute_pipe_deviation_map."],
            "charts": [],
            "data": {},
            "artifacts": {},
        }
    return compute_pipe_deviation_map(cloud, pipe_fit, **args)



def _run_detect_pipe_dents(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("point_cloud", state.get("raw_point_cloud", resource))
    deviation_map = state.get("deviation_map")
    if deviation_map is None:
        return {
            "insights": ["Dent detection skipped because no deviation map was available."],
            "insight_objects": [],
            "diagnostics": ["Run compute_pipe_deviation_map before detect_pipe_dents."],
            "charts": [],
            "data": {"dents": []},
            "artifacts": {},
        }
    return detect_pipe_dents(cloud, deviation_map, **args)



def _run_measure_pipe_ovality(resource: PointCloudData, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cloud = state.get("point_cloud", state.get("raw_point_cloud", resource))
    pipe_fit = state.get("pipe_fit")
    if pipe_fit is None:
        return {
            "insights": ["Ovality analysis skipped because no pipe fit was available."],
            "insight_objects": [],
            "diagnostics": ["Run fit_pipe_cylinder before measure_pipe_ovality."],
            "charts": [],
            "data": {},
            "artifacts": {},
        }
    return measure_pipe_ovality(cloud, pipe_fit, **args)


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "validate_dataset": {"domain": "general", "runner": _run_validate, "description": "Run basic data quality checks."},
    "profile_table": {"domain": "general", "runner": _run_profile, "description": "Profile the dataset."},
    "audit_schema_contract": {"domain": "general", "runner": _run_schema_contract, "description": "Check required columns, types, and schema-version drift."},
    "assess_freshness": {"domain": "general", "runner": _run_freshness, "description": "Assess recency of the latest records."},
    "audit_standardization": {"domain": "general", "runner": _run_standardization, "description": "Detect text-label variants and standardization opportunities."},
    "detect_entity_collisions": {"domain": "general", "runner": _run_entity_collisions, "description": "Screen for duplicate-entity risk using normalized keys."},
    "detect_anomalies": {"domain": "general", "runner": lambda resource, args, state: run_anomaly_detection(resource, **args), "description": "Detect metric anomalies over time."},
    "segment_drivers": {"domain": "general", "runner": lambda resource, args, state: run_segment_analysis(resource, **args), "description": "Analyze segment contribution."},
    "scan_correlations": {"domain": "general", "runner": lambda resource, args, state: scan_correlations(resource, **args), "description": "Find correlated numeric drivers."},
    "fit_driver_regression": {"domain": "general", "runner": lambda resource, args, state: fit_driver_regression(resource, **args), "description": "Fit driver regression."},
    "forecast_metric": {"domain": "general", "runner": lambda resource, args, state: forecast_metric(resource, **args), "description": "Forecast a metric over time."},
    "bayesian_ab_test": {"domain": "general", "runner": lambda resource, args, state: bayesian_ab_test(resource, **args), "description": "Run Bayesian experiment analysis."},
    "calculate_returns": {"domain": "finance", "runner": lambda resource, args, state: calculate_returns(resource, **args), "description": "Compute return series."},
    "measure_risk": {"domain": "finance", "runner": lambda resource, args, state: measure_risk(resource, **args), "description": "Compute volatility and benchmark risk."},
    "measure_drawdown": {"domain": "finance", "runner": lambda resource, args, state: measure_drawdown(resource, **args), "description": "Measure drawdowns."},
    "detect_volume_spikes": {"domain": "finance", "runner": lambda resource, args, state: detect_volume_spikes(resource, **args), "description": "Detect abnormal volume."},
    "optimize_portfolio": {"domain": "finance", "runner": lambda resource, args, state: optimize_portfolio(resource, **args), "description": "Optimize portfolio weights."},
    "backtest_signal": {"domain": "finance", "runner": lambda resource, args, state: backtest_signal(resource, **args), "description": "Backtest a trading signal."},
    "compute_readmission_rate": {"domain": "healthcare", "runner": lambda resource, args, state: compute_readmission_rate(resource, **args), "description": "Compute readmission rate."},
    "compare_cohorts": {"domain": "healthcare", "runner": lambda resource, args, state: compare_cohorts(resource, **args), "description": "Compare cohort outcomes."},
    "analyze_length_of_stay": {"domain": "healthcare", "runner": lambda resource, args, state: analyze_length_of_stay(resource, **args), "description": "Analyze length of stay."},
    "survival_risk_analysis": {"domain": "healthcare", "runner": lambda resource, args, state: survival_risk_analysis(resource, **args), "description": "Run survival analysis."},
    "estimate_treatment_effect": {"domain": "healthcare", "runner": lambda resource, args, state: estimate_treatment_effect(resource, **args), "description": "Estimate treatment effect."},
    "profile_point_cloud": {"domain": "pipeline", "runner": _run_profile_point_cloud, "description": "Profile a point-cloud scan."},
    "clean_point_cloud": {"domain": "pipeline", "runner": _run_clean_point_cloud, "description": "Downsample and denoise the point cloud."},
    "fit_pipe_cylinder": {"domain": "pipeline", "runner": _run_fit_pipe_cylinder, "description": "Fit a nominal pipe cylinder."},
    "compute_pipe_deviation_map": {"domain": "pipeline", "runner": _run_compute_pipe_deviation_map, "description": "Compute signed radial deviations."},
    "detect_pipe_dents": {"domain": "pipeline", "runner": _run_detect_pipe_dents, "description": "Detect dent clusters from inward deviations."},
    "measure_pipe_ovality": {"domain": "pipeline", "runner": _run_measure_pipe_ovality, "description": "Measure cross-sectional ovality."},
}



def list_available_tools(domain: str | None = None, tool_whitelist: list[str] | None = None) -> list[dict[str, Any]]:
    whitelist = set(tool_whitelist or [])
    tools = []
    for name, definition in TOOL_REGISTRY.items():
        if domain == "pipeline":
            if definition["domain"] != "pipeline":
                continue
        elif domain and domain != "general":
            if definition["domain"] not in {"general", domain}:
                continue
        elif domain == "general" and definition["domain"] != "general":
            continue

        if whitelist and name not in whitelist:
            continue
        tools.append({"name": name, "domain": definition["domain"], "description": definition["description"]})
    return tools



def build_tool_step(
    tool_name: str,
    resource: pd.DataFrame | PointCloudData,
    semantic_config: dict[str, Any] | None = None,
    analysis_params: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    semantic_config = semantic_config or {}
    analysis_params = analysis_params or {}
    args = _resolve_args(tool_name, resource, semantic_config, analysis_params)
    usable, invalid = _args_are_usable(resource, args)
    return {"tool": tool_name, "args": args, "usable": usable}, invalid



def execute_tool(
    tool_name: str,
    resource: pd.DataFrame | PointCloudData,
    args: dict[str, Any],
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    return TOOL_REGISTRY[tool_name]["runner"](resource, args, state or {})
