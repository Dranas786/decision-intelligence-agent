from __future__ import annotations

from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result

try:
    from lifelines import KaplanMeierFitter
except ImportError:  # pragma: no cover
    KaplanMeierFitter = None


def compute_readmission_rate(
    df: pd.DataFrame,
    patient_id_col: str,
    admission_date_col: str,
    window_days: int = 30,
) -> dict[str, Any]:
    for column in (patient_id_col, admission_date_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df[[patient_id_col, admission_date_col]].copy()
    working_df[admission_date_col] = pd.to_datetime(working_df[admission_date_col], errors="coerce")
    working_df = working_df.dropna().sort_values([patient_id_col, admission_date_col])
    if working_df.empty:
        return build_result(insights=["Readmission analysis skipped because no valid admissions were available."])

    working_df["next_admission"] = working_df.groupby(patient_id_col)[admission_date_col].shift(-1)
    working_df["days_to_readmit"] = (working_df["next_admission"] - working_df[admission_date_col]).dt.days
    readmit_flag = working_df["days_to_readmit"].between(0, window_days, inclusive="both")
    readmission_rate = float(readmit_flag.mean())

    return build_result(
        insights=[f"Computed {window_days}-day readmission rate of {readmission_rate:.1%}."],
        insight_objects=[
            build_insight(
                tool="compute_readmission_rate",
                title="Readmission rate",
                message="Readmission window analysis completed.",
                evidence={"window_days": window_days, "readmission_rate": readmission_rate},
            )
        ],
        data={"readmission_rate": readmission_rate},
    )


def compare_cohorts(
    df: pd.DataFrame,
    cohort_col: str,
    outcome_col: str,
) -> dict[str, Any]:
    for column in (cohort_col, outcome_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    grouped = df[[cohort_col, outcome_col]].dropna().groupby(cohort_col)[outcome_col].agg(["mean", "count"]).reset_index()
    if grouped.empty:
        return build_result(insights=["Cohort comparison skipped because no valid rows were available."])

    rows = []
    insights = []
    for _, row in grouped.iterrows():
        label = str(row[cohort_col])
        mean_value = float(row["mean"])
        count = int(row["count"])
        rows.append({"cohort": label, "mean_outcome": mean_value, "count": count})
        insights.append(f"Cohort '{label}' has mean outcome {mean_value:.2f} across {count} rows.")

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="compare_cohorts",
                title="Cohort outcomes",
                message="Cohort comparison completed.",
                evidence={"cohorts": rows},
            )
        ],
        charts=[{"type": "bar", "title": f"{outcome_col} by {cohort_col}"}],
        data={"cohorts": rows},
    )


def analyze_length_of_stay(
    df: pd.DataFrame,
    admission_date_col: str,
    discharge_date_col: str,
    cohort_col: str | None = None,
) -> dict[str, Any]:
    for column in (admission_date_col, discharge_date_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df.copy()
    working_df[admission_date_col] = pd.to_datetime(working_df[admission_date_col], errors="coerce")
    working_df[discharge_date_col] = pd.to_datetime(working_df[discharge_date_col], errors="coerce")
    working_df["length_of_stay"] = (
        working_df[discharge_date_col] - working_df[admission_date_col]
    ).dt.days
    working_df = working_df.dropna(subset=["length_of_stay"])

    if working_df.empty:
        return build_result(insights=["Length-of-stay analysis skipped because no complete stays were available."])

    mean_los = float(working_df["length_of_stay"].mean())
    insights = [f"Average length of stay is {mean_los:.2f} days."]
    data: dict[str, Any] = {"mean_length_of_stay": mean_los}

    if cohort_col and cohort_col in working_df.columns:
        grouped = (
            working_df.groupby(cohort_col)["length_of_stay"].agg(["mean", "count"]).reset_index().to_dict(orient="records")
        )
        insights.append(f"Computed cohort-level length of stay across {len(grouped)} groups.")
        data["cohort_breakdown"] = grouped

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="analyze_length_of_stay",
                title="Length of stay",
                message="Length-of-stay analysis completed.",
                evidence=data,
            )
        ],
        data=data,
    )


def survival_risk_analysis(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    cohort_col: str | None = None,
) -> dict[str, Any]:
    for column in (duration_col, event_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df[[duration_col, event_col] + ([cohort_col] if cohort_col and cohort_col in df.columns else [])].dropna()
    if working_df.empty:
        return build_result(insights=["Survival analysis skipped because no valid rows were available."])

    diagnostics: list[str] = []
    if KaplanMeierFitter is not None:
        fitter = KaplanMeierFitter()
        fitter.fit(working_df[duration_col], event_observed=working_df[event_col])
        median_survival = fitter.median_survival_time_
        timeline = fitter.survival_function_.reset_index().head(10).to_dict(orient="records")
    else:
        diagnostics.append("lifelines not installed; used empirical survival fallback.")
        sorted_df = working_df.sort_values(duration_col)
        total = len(sorted_df)
        surviving = total
        timeline = []
        median_survival = None
        for _, row in sorted_df.iterrows():
            if row[event_col]:
                surviving -= 1
            survival_prob = surviving / total
            timeline.append({"timeline": float(row[duration_col]), "survival": float(survival_prob)})
            if median_survival is None and survival_prob <= 0.5:
                median_survival = float(row[duration_col])
        timeline = timeline[:10]

    insights = [f"Estimated median survival/event-free time is {median_survival}."]
    if cohort_col and cohort_col in working_df.columns:
        insights.append(f"Survival analysis included stratification column '{cohort_col}'.")

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="survival_risk_analysis",
                title="Survival summary",
                message="Time-to-event analysis completed.",
                evidence={"median_survival": median_survival, "timeline": timeline},
            )
        ],
        diagnostics=diagnostics,
        charts=[{"type": "line", "title": "Survival curve"}],
        data={"median_survival": median_survival, "timeline": timeline},
    )


def estimate_treatment_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> dict[str, Any]:
    for column in (treatment_col, outcome_col):
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found in dataset.")

    working_df = df[[treatment_col, outcome_col]].dropna().copy()
    treatment_groups = working_df.groupby(treatment_col)[outcome_col].mean().to_dict()
    if len(treatment_groups) < 2:
        return build_result(
            insights=["Treatment-effect estimate skipped because fewer than two treatment groups were present."],
            diagnostics=["Treatment-effect estimation requires at least two groups."],
        )

    labels = list(treatment_groups.keys())
    effect = float(treatment_groups[labels[1]] - treatment_groups[labels[0]])
    insights = [f"Estimated treatment effect is {effect:.4f} comparing '{labels[1]}' to '{labels[0]}'."]

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="estimate_treatment_effect",
                title="Treatment effect",
                message="Simple treatment-effect estimate completed.",
                evidence={"group_means": treatment_groups, "effect": effect},
            )
        ],
        data={"group_means": treatment_groups, "effect": effect},
    )
