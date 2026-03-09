from __future__ import annotations

from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result


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
        df.isna().mean().sort_values(ascending=False).head(5).round(4).to_dict()
        if len(df.columns) > 0
        else {}
    )

    high_missing = {key: value for key, value in missing_share.items() if value >= missing_threshold}
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

    insights.append(
        f"Dataset validation found duplicate rate {duplicate_rate:.1%} and "
        f"{len(missing_columns)} missing required columns."
    )
    insight_objects.append(
        build_insight(
            tool="validate_dataset",
            title="Validation summary",
            message="Basic expectation-style validation completed.",
            evidence={
                "duplicate_rate": duplicate_rate,
                "missing_columns": missing_columns,
                "top_missing_share": missing_share,
            },
            severity="warning" if missing_columns else "info",
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
        },
    )
