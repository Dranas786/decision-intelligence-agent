import math
from pathlib import Path

import numpy as np
import pandas as pd

from app.analytics.anomalies import run_anomaly_detection
from app.analytics.bayes import bayesian_ab_test
from app.analytics.correlations import scan_correlations
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
    PointCloudData,
    clean_point_cloud,
    compute_pipe_deviation_map,
    detect_pipe_dents,
    fit_pipe_cylinder,
    measure_pipe_ovality,
    profile_point_cloud,
)
from app.analytics.profiling import build_profile_summary
from app.analytics.regression import fit_driver_regression
from app.analytics.segmentation import run_segment_analysis
from app.analytics.validation import validate_dataset


rng = np.random.default_rng(7)



def make_pipe_points(
    radius: float = 1.0,
    length: float = 8.0,
    axial_steps: int = 50,
    angular_steps: int = 80,
    dent_depth: float = 0.0,
    dent_axial_center: float = 0.0,
    dent_axial_width: float = 0.8,
    dent_angle_center: float = 0.0,
    dent_angle_width: float = 0.8,
    ovality: float = 0.0,
    noise: float = 0.002,
) -> np.ndarray:
    points: list[list[float]] = []
    axial_values = np.linspace(-length / 2, length / 2, axial_steps)
    angles = np.linspace(-math.pi, math.pi, angular_steps, endpoint=False)

    for axial in axial_values:
        for angle in angles:
            local_radius = radius * (1 + ovality * math.cos(2 * angle))
            if dent_depth > 0:
                axial_factor = math.exp(-((axial - dent_axial_center) ** 2) / max(dent_axial_width ** 2, 1e-6))
                angle_delta = ((angle - dent_angle_center + math.pi) % (2 * math.pi)) - math.pi
                angular_factor = math.exp(-(angle_delta ** 2) / max(dent_angle_width ** 2, 1e-6))
                local_radius -= dent_depth * axial_factor * angular_factor
            y = local_radius * math.cos(angle)
            z = local_radius * math.sin(angle)
            points.append([
                axial + float(rng.normal(0.0, noise)),
                y + float(rng.normal(0.0, noise)),
                z + float(rng.normal(0.0, noise)),
            ])
    return np.asarray(points, dtype=float)



def test_general_tools_produce_structured_results():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "revenue": [100, 102, 103, 104, 70, 106, 107, 108],
            "marketing_spend": [10, 11, 10, 12, 8, 13, 12, 13],
            "orders": [5, 5, 6, 6, 4, 7, 7, 7],
            "channel": ["paid", "paid", "organic", "organic", "paid", "email", "email", "email"],
            "variant": ["control", "control", "control", "control", "treatment", "treatment", "treatment", "treatment"],
            "converted": [0, 0, 1, 0, 1, 1, 1, 1],
        }
    )

    assert build_profile_summary(df)["data"]["row_count"] == 8
    assert validate_dataset(df, required_columns=["date", "revenue"])["data"]["missing_columns"] == []
    assert run_anomaly_detection(df, metric_col="revenue", date_col="date", rolling_window=3)["data"]["anomalies"]
    assert run_segment_analysis(df, metric_col="revenue", segment_col="channel", date_col="date")["data"]["segments"]
    assert scan_correlations(df, target_col="revenue", feature_cols=["marketing_spend", "orders"])["data"]["correlations"]
    assert fit_driver_regression(df, target_col="revenue", feature_cols=["marketing_spend", "orders"])["data"]["r_squared"] >= 0
    assert forecast_metric(df, metric_col="revenue", date_col="date", periods=3)["data"]["forecast"]
    assert bayesian_ab_test(df, outcome_col="converted", variant_col="variant")["data"]["win_probability"] >= 0



def test_finance_tools_produce_structured_results():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    finance_df = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "ticker": ["AAA"] * 6 + ["BBB"] * 6,
            "close": [100, 101, 103, 102, 104, 106, 50, 52, 51, 53, 54, 56],
            "benchmark": [200, 201, 202, 203, 204, 206, 200, 201, 202, 203, 204, 206],
            "volume": [1000, 1100, 1050, 4000, 1200, 1300, 800, 820, 840, 860, 880, 900],
            "signal": [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        }
    )
    single_asset_df = finance_df.loc[finance_df["ticker"] == "AAA"].reset_index(drop=True)

    assert calculate_returns(single_asset_df, price_col="close")["data"]["returns"]
    assert measure_risk(single_asset_df, price_col="close", benchmark_col="benchmark")["data"]["volatility"] >= 0
    assert measure_drawdown(single_asset_df, price_col="close")["data"]["max_drawdown"] <= 0
    assert detect_volume_spikes(single_asset_df, volume_col="volume", date_col="date")["data"]["spikes"]
    assert optimize_portfolio(finance_df, entity_col="ticker", price_col="close", date_col="date")["data"]["weights"]
    assert backtest_signal(single_asset_df, price_col="close", signal_col="signal")["data"]["equity_curve"]



def test_healthcare_tools_produce_structured_results():
    healthcare_df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p3", "p3", "p4"],
            "admission_date": pd.to_datetime(
                ["2026-01-01", "2026-01-20", "2026-01-05", "2026-01-03", "2026-02-10", "2026-01-07"]
            ),
            "discharge_date": pd.to_datetime(
                ["2026-01-05", "2026-01-25", "2026-01-08", "2026-01-08", "2026-02-20", "2026-01-11"]
            ),
            "cohort": ["A", "A", "B", "A", "A", "B"],
            "outcome": [0.8, 0.7, 0.6, 0.9, 0.85, 0.5],
            "duration_days": [12, 8, 20, 10, 14, 16],
            "event": [1, 0, 1, 0, 1, 1],
            "treatment": [0, 1, 0, 1, 1, 0],
        }
    )

    assert compute_readmission_rate(healthcare_df, patient_id_col="patient_id", admission_date_col="admission_date")["data"]["readmission_rate"] >= 0
    assert compare_cohorts(healthcare_df, cohort_col="cohort", outcome_col="outcome")["data"]["cohorts"]
    assert analyze_length_of_stay(
        healthcare_df,
        admission_date_col="admission_date",
        discharge_date_col="discharge_date",
        cohort_col="cohort",
    )["data"]["mean_length_of_stay"] > 0
    assert survival_risk_analysis(healthcare_df, duration_col="duration_days", event_col="event")["data"]["timeline"]
    assert estimate_treatment_effect(healthcare_df, treatment_col="treatment", outcome_col="outcome")["data"]["effect"] != 0



def test_pipeline_tools_detect_dent_and_ovality():
    points = make_pipe_points(dent_depth=0.12, dent_axial_width=0.5, dent_angle_width=0.4)
    cloud = PointCloudData(points=points)

    assert profile_point_cloud(cloud, units="m")["data"]["point_count"] == len(points)

    cleaned_result = clean_point_cloud(cloud, voxel_size=0.03, max_outlier_std=2.5)
    cleaned_cloud = cleaned_result["artifacts"]["point_cloud"]
    fit_result = fit_pipe_cylinder(cleaned_cloud, units="m")
    pipe_fit = fit_result["artifacts"]["pipe_fit"]
    deviation_result = compute_pipe_deviation_map(cleaned_cloud, pipe_fit, units="m")
    deviation_map = deviation_result["artifacts"]["deviation_map"]
    dent_result = detect_pipe_dents(
        cleaned_cloud,
        deviation_map,
        deviation_threshold=0.05,
        min_cluster_points=12,
    )

    assert fit_result["data"]["radius"] > 0.8
    assert deviation_result["data"]["min_deviation"] < 0
    assert dent_result["data"]["dents"]

    oval_points = make_pipe_points(ovality=0.08, noise=0.001)
    oval_cloud = PointCloudData(points=oval_points)
    oval_fit = fit_pipe_cylinder(oval_cloud, units="m")["artifacts"]["pipe_fit"]
    ovality_result = measure_pipe_ovality(oval_cloud, oval_fit, slice_spacing=0.4, units="m")
    assert ovality_result["data"]["slices"]
