from __future__ import annotations

from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result, safe_float


def scan_correlations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    feature_cols = feature_cols or []
    candidate_cols = [col for col in feature_cols if col in df.columns and col != target_col]

    if not candidate_cols:
        candidate_cols = [
            column
            for column in df.select_dtypes(include=["number"]).columns.tolist()
            if column != target_col
        ]

    if not candidate_cols:
        return build_result(
            insights=["No numeric features available for correlation analysis."],
            diagnostics=["Correlation scan skipped because no numeric feature columns were available."],
        )

    correlation_frame = df[[target_col] + candidate_cols].dropna()
    if correlation_frame.empty:
        return build_result(insights=["No valid rows available for correlation analysis."])

    correlations = (
        correlation_frame.corr(numeric_only=True)[target_col]
        .drop(labels=[target_col], errors="ignore")
        .dropna()
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(top_n)
    )

    insights = []
    insight_objects = []
    rows = []
    for feature, value in correlations.items():
        direction = "positive" if value >= 0 else "negative"
        insights.append(f"Feature '{feature}' has {direction} correlation {value:.2f} with '{target_col}'.")
        rows.append({"feature": feature, "correlation": float(value)})
        insight_objects.append(
            build_insight(
                tool="scan_correlations",
                title=f"Correlation: {feature}",
                message=f"{feature} shows a {direction} relationship with {target_col}.",
                evidence={"feature": feature, "target": target_col, "correlation": safe_float(value)},
            )
        )

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        data={"correlations": rows},
        charts=[{"type": "bar", "title": f"Top correlations with {target_col}"}],
    )
