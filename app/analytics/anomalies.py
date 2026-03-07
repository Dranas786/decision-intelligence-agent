from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result


def run_anomaly_detection(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str,
    z_threshold: float = 2.0,
    rolling_window: int = 5,
    period: str | None = None,
) -> dict[str, Any]:
    """
    Detect anomalies in a time-based metric using rolling z-scores.
    """

    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataset.")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset.")

    working_df = df[[date_col, metric_col]].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna(subset=[date_col, metric_col]).sort_values(date_col)

    if working_df.empty:
        return build_result(insights=["No valid rows available for anomaly detection."])

    if period:
        working_df = working_df.set_index(date_col).resample(period)[metric_col].sum().reset_index()

    rolling_mean = working_df[metric_col].rolling(window=rolling_window, min_periods=2).mean()
    rolling_std = working_df[metric_col].rolling(window=rolling_window, min_periods=2).std()
    global_std = working_df[metric_col].std()
    working_df["z_score"] = (working_df[metric_col] - rolling_mean) / rolling_std.replace(0, global_std)
    working_df["z_score"] = working_df["z_score"].fillna(0.0)

    anomaly_df = working_df[working_df["z_score"].abs() >= z_threshold].copy()
    if anomaly_df.empty:
        return build_result(
            insights=[f"No anomalies detected for '{metric_col}' using z-threshold {z_threshold}."],
            data={"anomalies": []},
        )

    anomalies = []
    insights = []
    insight_objects = []

    for _, row in anomaly_df.iterrows():
        direction = "spike" if row[metric_col] > working_df[metric_col].mean() else "drop"
        date_str = row[date_col].strftime("%Y-%m-%d")
        anomaly = {
            "date": date_str,
            "metric": metric_col,
            "value": float(row[metric_col]),
            "z_score": float(row["z_score"]),
            "direction": direction,
        }
        anomalies.append(anomaly)
        insights.append(
            f"Detected an unusual {direction} in '{metric_col}' on {date_str} "
            f"(value={row[metric_col]:.2f}, z={row['z_score']:.2f})."
        )
        insight_objects.append(
            build_insight(
                tool="detect_anomalies",
                title=f"{direction.title()} anomaly",
                message=f"Anomalous movement detected for {metric_col}.",
                evidence=anomaly,
                severity="warning",
            )
        )

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        charts=[{"type": "line", "title": f"Anomalies in {metric_col}"}],
        data={"anomalies": anomalies},
    )
