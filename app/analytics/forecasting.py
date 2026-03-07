from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.analytics.common import build_insight, build_result

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None


def forecast_metric(
    df: pd.DataFrame,
    metric_col: str,
    date_col: str,
    periods: int = 7,
) -> dict[str, Any]:
    if metric_col not in df.columns:
        raise ValueError(f"Metric column '{metric_col}' not found in dataset.")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset.")

    working_df = df[[date_col, metric_col]].copy()
    working_df[date_col] = pd.to_datetime(working_df[date_col], errors="coerce")
    working_df = working_df.dropna().sort_values(date_col)

    if len(working_df) < 3:
        return build_result(
            insights=["Forecast skipped because fewer than three valid observations were available."],
            diagnostics=["Forecasting requires at least three non-null rows."],
        )

    diagnostics: list[str] = []
    forecast_rows: list[dict[str, Any]]

    if Prophet is not None:
        model_df = working_df.rename(columns={date_col: "ds", metric_col: "y"})
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(model_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future).tail(periods)
        forecast_rows = [
            {
                "date": row["ds"].strftime("%Y-%m-%d"),
                "forecast": float(row["yhat"]),
                "lower": float(row["yhat_lower"]),
                "upper": float(row["yhat_upper"]),
            }
            for _, row in forecast.iterrows()
        ]
    else:
        diagnostics.append("prophet not installed; used linear-trend fallback forecast.")
        x = np.arange(len(working_df), dtype=float)
        y = working_df[metric_col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        residual_std = float(np.std(y - (slope * x + intercept)))
        future_index = np.arange(len(working_df), len(working_df) + periods, dtype=float)
        start_date = working_df[date_col].max()
        forecast_rows = []
        for offset, value_index in enumerate(future_index, start=1):
            forecast_value = slope * value_index + intercept
            forecast_rows.append(
                {
                    "date": (start_date + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
                    "forecast": float(forecast_value),
                    "lower": float(forecast_value - residual_std),
                    "upper": float(forecast_value + residual_std),
                }
            )

    insights = [
        f"Forecasted '{metric_col}' for the next {periods} periods; first prediction is "
        f"{forecast_rows[0]['forecast']:.2f} on {forecast_rows[0]['date']}."
    ]
    insight_objects = [
        build_insight(
            tool="forecast_metric",
            title="Forecast summary",
            message="Forward forecast completed.",
            evidence={"metric": metric_col, "date_col": date_col, "forecast": forecast_rows},
        )
    ]

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        diagnostics=diagnostics,
        charts=[{"type": "line", "title": f"Forecast for {metric_col}"}],
        data={"forecast": forecast_rows},
    )
