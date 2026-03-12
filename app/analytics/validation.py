from __future__ import annotations

from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result

SENSITIVE_TOKENS = ("email", "phone", "ssn", "sin", "dob", "birth", "address", "owner", "name")
COUNT_TOKENS = ("count", "qty", "quantity", "visits", "volume")
DATE_TOKENS = ("date", "time", "timestamp")


def _sensitive_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in SENSITIVE_TOKENS)]


def _date_conformity_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for column in df.columns:
        lowered = column.lower()
        if any(token in lowered for token in DATE_TOKENS) and df[column].dtype == "object":
            parsed = pd.to_datetime(df[column], errors="coerce")
            parse_failure_share = float(parsed.isna().mean())
            if 0 < parse_failure_share < 1:
                issues.append(f"{column} mixes valid and invalid date-like values ({parse_failure_share:.1%} failed parsing).")
    return issues


def _consistency_opportunities(df: pd.DataFrame) -> list[str]:
    opportunities: list[str] = []
    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    for column in object_columns[:8]:
        series = df[column].dropna().astype(str)
        if series.empty:
            continue
        normalized = series.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
        if normalized.nunique() < series.nunique():
            opportunities.append(f"{column} contains label variants that could be standardized.")
    return opportunities


def _validity_issues(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    for column in df.select_dtypes(include=["number"]).columns:
        lowered = column.lower()
        if any(token in lowered for token in COUNT_TOKENS):
            negative_share = float((df[column] < 0).mean())
            if negative_share > 0:
                issues.append(f"{column} contains negative values in a count-like field ({negative_share:.1%} of rows).")
    return issues


def validate_dataset(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    unique_subset: list[str] | None = None,
    missing_threshold: float = 0.2,
) -> dict[str, Any]:
    required_columns = required_columns or []
    unique_subset = unique_subset or []

    diagnostics: list[str] = []
    insight_objects: list[dict[str, Any]] = []
    insights: list[str] = []

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        diagnostics.append(f"Missing required columns: {missing_columns}")

    duplicate_rate = 0.0
    if len(df) > 0:
        duplicate_rate = float(df.duplicated(subset=unique_subset or None).mean())

    missing_share = (
        df.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict()
        if len(df.columns) > 0
        else {}
    )

    high_missing = {key: value for key, value in missing_share.items() if value >= missing_threshold}
    completeness_score = float(1.0 - df.isna().mean().mean()) if len(df.columns) else 1.0
    uniqueness_score = float(1.0 - duplicate_rate)
    conformity_issues = _date_conformity_issues(df)
    consistency_opportunities = _consistency_opportunities(df)
    validity_issues = _validity_issues(df)
    sensitive_fields = _sensitive_columns(df.columns.tolist())

    human_review_required: list[str] = []
    if missing_columns:
        human_review_required.append(f"Define or provide the missing required columns: {missing_columns}.")
    if high_missing:
        human_review_required.append("Review high-missingness fields before publishing downstream outputs.")
    if sensitive_fields:
        human_review_required.append(f"Review sensitive-looking fields before exposing the dataset: {sensitive_fields}.")
    human_review_required.extend(conformity_issues[:2])
    human_review_required.extend(validity_issues[:2])

    methodology = [
        "Completeness: measure missing-value share by field.",
        "Uniqueness: measure duplicate rate using the supplied or inferred grain.",
        "Conformity: inspect parsability and type-format alignment.",
        "Consistency: inspect categorical standardization opportunities.",
        "Validity: inspect obvious rule breaks such as negative counts.",
        "Governance: flag sensitive fields and items requiring human review.",
    ]

    quality_dimensions = {
        "completeness": round(completeness_score, 4),
        "uniqueness": round(uniqueness_score, 4),
        "conformity_issues": conformity_issues,
        "consistency_opportunities": consistency_opportunities,
        "validity_issues": validity_issues,
    }

    if high_missing:
        insights.append(f"Columns with elevated missingness: {high_missing}.")
        insight_objects.append(
            build_insight(
                tool="validate_dataset",
                title="High missingness",
                message="One or more columns exceed the configured missing-value threshold.",
                severity="warning",
                evidence={"missing_share": high_missing, "threshold": missing_threshold},
            )
        )

    if consistency_opportunities:
        insights.append(f"Standardization opportunities detected: {consistency_opportunities}.")
    if conformity_issues:
        insights.append(f"Conformity issues detected: {conformity_issues}.")
    if validity_issues:
        insights.append(f"Validity issues detected: {validity_issues}.")
    if sensitive_fields:
        insights.append(f"Potential governance-sensitive fields detected: {sensitive_fields}.")

    insights.append(
        f"Dataset validation found duplicate rate {duplicate_rate:.1%}, completeness score {completeness_score:.1%}, "
        f"and {len(missing_columns)} missing required columns."
    )
    insight_objects.append(
        build_insight(
            tool="validate_dataset",
            title="Validation summary",
            message="Expectation-style validation completed across completeness, uniqueness, conformity, consistency, and validity.",
            evidence={
                "duplicate_rate": duplicate_rate,
                "missing_columns": missing_columns,
                "top_missing_share": missing_share,
                "quality_dimensions": quality_dimensions,
                "sensitive_fields": sensitive_fields,
                "human_review_required": human_review_required,
                "methodology": methodology,
            },
            severity="warning" if missing_columns or high_missing else "info",
        )
    )

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        diagnostics=diagnostics,
        data={
            "duplicate_rate": duplicate_rate,
            "missing_columns": missing_columns,
            "top_missing_share": missing_share,
            "quality_dimensions": quality_dimensions,
            "sensitive_fields": sensitive_fields,
            "human_review_required": human_review_required,
            "methodology": methodology,
        },
    )
