from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result

SENSITIVE_TOKENS = ("email", "phone", "ssn", "sin", "dob", "birth", "address", "owner", "name")
AMBIGUOUS_COLUMN_NAMES = {"id", "name", "value", "data", "field", "column", "desc", "description"}


def _sensitive_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if any(token in column.lower() for token in SENSITIVE_TOKENS)]


def _ambiguous_columns(columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        lowered = column.lower()
        if lowered in AMBIGUOUS_COLUMN_NAMES or lowered.startswith("col"):
            flagged.append(column)
    return flagged


def _likely_key_columns(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    row_count = len(df)
    if not row_count:
        return candidates

    for column in df.columns:
        non_null = df[column].notna().mean()
        unique_ratio = df[column].nunique(dropna=True) / row_count
        if non_null == 1.0 and unique_ratio >= 0.98:
            candidates.append(column)

    if candidates:
        return candidates[:3]

    id_like = [column for column in df.columns if column.lower().endswith("id")]
    return id_like[:3]


def _business_key_candidates(df: pd.DataFrame) -> list[list[str]]:
    categorical_like = [
        column for column in df.columns
        if df[column].dtype == "object" or column.lower().endswith("id")
    ][:6]
    row_count = len(df)
    if row_count == 0:
        return []

    candidates: list[list[str]] = []
    for left, right in combinations(categorical_like, 2):
        combined_unique = df[[left, right]].drop_duplicates().shape[0] / row_count
        if combined_unique >= 0.95:
            candidates.append([left, right])
    return candidates[:3]


def build_profile_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a richer profile of the dataframe for the agent.
    """

    row_count = len(df)
    column_count = len(df.columns)

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    missing_counts = df.isna().sum().to_dict()
    missing_share = df.isna().mean().round(4).sort_values(ascending=False).to_dict()
    duplicate_rate = float(df.duplicated().mean()) if row_count else 0.0
    cardinality = {
        column: int(df[column].nunique(dropna=True))
        for column in df.columns[: min(column_count, 10)]
    }
    inferred_types = {column: str(dtype) for column, dtype in df.dtypes.items()}
    likely_keys = _likely_key_columns(df)
    business_key_candidates = _business_key_candidates(df)
    sensitive_fields = _sensitive_columns(df.columns.tolist())
    ambiguous_columns = _ambiguous_columns(df.columns.tolist())

    if likely_keys:
        grain = f"One row likely represents a unique {likely_keys[0]} record."
    elif business_key_candidates:
        grain = f"One row may be unique at the combination grain {business_key_candidates[0]}."
    else:
        grain = "Row grain is not obvious from the current columns alone."

    methodology = [
        "Completeness: inspect missing-value share by column.",
        "Uniqueness: inspect duplicate rate and candidate business keys.",
        "Conformity: inspect inferred types and field naming patterns.",
        "Consistency: inspect repeated categorical values and grain assumptions.",
        "Governance: flag sensitive or ambiguous fields for review.",
    ]

    summary = (
        f"Dataset has {row_count} rows and {column_count} columns, duplicate rate {duplicate_rate:.1%}. "
        f"Likely keys: {likely_keys or 'none detected'}."
    )

    evidence = {
        "row_count": row_count,
        "column_count": column_count,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "duplicate_rate": duplicate_rate,
        "likely_keys": likely_keys,
        "business_key_candidates": business_key_candidates,
        "grain": grain,
        "sensitive_fields": sensitive_fields,
        "ambiguous_columns": ambiguous_columns,
    }

    insights = [summary, grain]
    if sensitive_fields:
        insights.append(f"Potentially sensitive fields detected by name pattern: {sensitive_fields}.")
    if ambiguous_columns:
        insights.append(f"Columns that may need business definitions: {ambiguous_columns}.")

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="profile_table",
                title="Profile summary",
                message="Dataset profiling completed with basic governance-oriented metadata.",
                evidence=evidence,
            )
        ],
        data={
            "summary": summary,
            "row_count": row_count,
            "column_count": column_count,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_counts": missing_counts,
            "missing_share": missing_share,
            "duplicate_rate": duplicate_rate,
            "cardinality": cardinality,
            "inferred_types": inferred_types,
            "likely_keys": likely_keys,
            "business_key_candidates": business_key_candidates,
            "grain": grain,
            "sensitive_fields": sensitive_fields,
            "ambiguous_columns": ambiguous_columns,
            "methodology": methodology,
        },
    )
