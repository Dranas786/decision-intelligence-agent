from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.analytics.common import build_insight, build_result

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover
    sm = None


def fit_driver_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    usable_features = [column for column in feature_cols if column in df.columns and column != target_col]
    if not usable_features:
        return build_result(
            insights=["Regression skipped because no valid feature columns were supplied."],
            diagnostics=["No usable feature columns available for regression."],
        )

    working_df = df[[target_col] + usable_features].dropna()
    if len(working_df) < 3:
        return build_result(
            insights=["Regression skipped because too few rows remained after dropping nulls."],
            diagnostics=["Regression requires at least three complete rows."],
        )

    x = working_df[usable_features]
    y = working_df[target_col]

    coefficients: list[dict[str, Any]] = []
    diagnostics: list[str] = []

    if sm is not None:
        x_model = sm.add_constant(x, has_constant="add")
        model = sm.OLS(y, x_model).fit()
        for feature, value in model.params.items():
            if feature == "const":
                continue
            coefficients.append(
                {
                    "feature": feature,
                    "coefficient": float(value),
                    "p_value": float(model.pvalues.get(feature, np.nan)),
                }
            )
        r_squared = float(model.rsquared)
    else:
        diagnostics.append("statsmodels not installed; used numpy least-squares fallback.")
        design_matrix = np.column_stack([np.ones(len(x))] + [x[column].to_numpy(dtype=float) for column in usable_features])
        solution, *_ = np.linalg.lstsq(design_matrix, y.to_numpy(dtype=float), rcond=None)
        predictions = design_matrix @ solution
        residual = y.to_numpy(dtype=float) - predictions
        total = y.to_numpy(dtype=float) - y.mean()
        r_squared = float(1 - ((residual ** 2).sum() / max((total ** 2).sum(), 1e-9)))
        for index, feature in enumerate(usable_features, start=1):
            coefficients.append(
                {
                    "feature": feature,
                    "coefficient": float(solution[index]),
                    "p_value": None,
                }
            )

    coefficients.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    top_terms = coefficients[: min(5, len(coefficients))]
    insights = [
        f"Regression on '{target_col}' produced R^2 {r_squared:.2f}. Top driver: "
        f"{top_terms[0]['feature']} ({top_terms[0]['coefficient']:.2f})."
    ]
    insight_objects = [
        build_insight(
            tool="fit_driver_regression",
            title="Regression summary",
            message="Driver regression completed.",
            evidence={"target": target_col, "r_squared": r_squared, "coefficients": top_terms},
        )
    ]

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        diagnostics=diagnostics,
        data={"target": target_col, "coefficients": coefficients, "r_squared": r_squared},
        charts=[{"type": "bar", "title": f"Regression coefficients for {target_col}"}],
    )
