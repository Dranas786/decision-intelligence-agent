from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result


def run_segment_analysis(
    df: pd.DataFrame,
    metric_col: str,
    segment_col: str,
    top_n: int = 5,
    date_col: str | None = None,
) -> dict[str, Any]:
    """
    Analyze a metric by segment and return top contributing groups and optional deltas.
    """

    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataset.")

    if segment_col not in df.columns:
        raise ValueError(f"Segment column '{segment_col}' not found in dataset.")

    working_columns = [segment_col, metric_col] + ([date_col] if date_col and date_col in df.columns else [])
    working_df = df[working_columns].copy().dropna(subset=[segment_col, metric_col])

    if working_df.empty:
        return build_result(insights=["No valid rows available for segment analysis."])

    grouped = (
        working_df.groupby(segment_col, dropna=False)[metric_col]
        .agg(["sum", "mean", "count"])
        .reset_index()
        .sort_values("sum", ascending=False)
    )

    total_metric = grouped["sum"].sum()
    grouped["share_of_total"] = 0.0 if total_metric == 0 else grouped["sum"] / total_metric
    grouped["period_over_period_delta"] = None

    if date_col and date_col in working_df.columns:
        dated = working_df.copy()
        dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce")
        dated = dated.dropna(subset=[date_col]).sort_values(date_col)
        if not dated.empty:
            dated["period"] = dated[date_col].dt.to_period("M")
            period_group = (
                dated.groupby(["period", segment_col])[metric_col].sum().reset_index().sort_values(["period", segment_col])
            )
            period_group["delta"] = period_group.groupby(segment_col)[metric_col].diff()
            delta_map = period_group.dropna(subset=["delta"]).sort_values("period").groupby(segment_col)["delta"].last().to_dict()
            grouped["period_over_period_delta"] = grouped[segment_col].map(delta_map)

    top_segments = grouped.head(top_n).copy()

    segments = []
    insights = []
    insight_objects = []

    for _, row in top_segments.iterrows():
        segment_name = str(row[segment_col])
        segment_record = {
            "segment": segment_name,
            "sum": float(row["sum"]),
            "mean": float(row["mean"]),
            "count": int(row["count"]),
            "share_of_total": float(row["share_of_total"]),
            "period_over_period_delta": None if pd.isna(row["period_over_period_delta"]) else float(row["period_over_period_delta"]),
        }
        segments.append(segment_record)
        delta_text = ""
        if segment_record["period_over_period_delta"] is not None:
            delta_text = f", period-over-period delta {segment_record['period_over_period_delta']:.2f}"
        insights.append(
            f"Segment '{segment_name}' contributed {row['sum']:.2f} total {metric_col} "
            f"across {int(row['count'])} rows ({row['share_of_total']:.1%} of total){delta_text}."
        )
        insight_objects.append(
            build_insight(
                tool="segment_drivers",
                title=f"Segment: {segment_name}",
                message="Segment contribution analysis completed.",
                evidence=segment_record,
            )
        )

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        charts=[{"type": "bar", "title": f"{metric_col} by {segment_col}"}],
        data={"segments": segments},
    )
