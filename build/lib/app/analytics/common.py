from __future__ import annotations

from typing import Any

import pandas as pd


def build_result(
    *,
    insights: list[str] | None = None,
    insight_objects: list[dict[str, Any]] | None = None,
    diagnostics: list[str] | None = None,
    charts: list[dict[str, Any]] | None = None,
    data: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "insights": insights or [],
        "insight_objects": insight_objects or [],
        "diagnostics": diagnostics or [],
        "charts": charts or [],
        "data": data or {},
        "artifacts": artifacts or {},
    }


def build_insight(
    *,
    tool: str,
    title: str,
    message: str,
    evidence: dict[str, Any] | None = None,
    severity: str = "info",
) -> dict[str, Any]:
    return {
        "tool": tool,
        "title": title,
        "message": message,
        "severity": severity,
        "evidence": evidence or {},
    }


def find_time_column(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    if preferred and preferred in df.columns:
        return preferred

    for column in df.columns:
        lowered = column.lower()
        if any(token in lowered for token in ("date", "time", "month", "day")):
            return column

    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    return datetime_columns[0] if datetime_columns else None


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    numeric = set(numeric_columns(df))
    return [column for column in df.columns if column not in numeric]


def first_numeric(df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    for column in numeric_columns(df):
        if column not in exclude:
            return column
    return None


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def safe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
