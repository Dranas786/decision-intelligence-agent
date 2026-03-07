from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.analytics.common import build_insight, build_result


def bayesian_ab_test(
    df: pd.DataFrame,
    outcome_col: str,
    variant_col: str,
    control_label: str | None = None,
    treatment_label: str | None = None,
    samples: int = 5000,
) -> dict[str, Any]:
    if outcome_col not in df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in dataset.")
    if variant_col not in df.columns:
        raise ValueError(f"Variant column '{variant_col}' not found in dataset.")

    working_df = df[[outcome_col, variant_col]].dropna().copy()
    if working_df.empty:
        return build_result(insights=["No valid rows available for Bayesian A/B analysis."])

    variants = working_df[variant_col].astype(str).unique().tolist()
    if len(variants) < 2:
        return build_result(
            insights=["Bayesian A/B test skipped because fewer than two variants were present."],
            diagnostics=["A/B testing requires at least two distinct variants."],
        )

    control = control_label if control_label in variants else variants[0]
    treatment = treatment_label if treatment_label in variants else variants[1]

    control_values = working_df.loc[working_df[variant_col].astype(str) == control, outcome_col].astype(float)
    treatment_values = working_df.loc[working_df[variant_col].astype(str) == treatment, outcome_col].astype(float)

    if set(control_values.unique()).issubset({0.0, 1.0}) and set(treatment_values.unique()).issubset({0.0, 1.0}):
        control_success = float(control_values.sum())
        treatment_success = float(treatment_values.sum())
        control_draws = np.random.default_rng(42).beta(control_success + 1, len(control_values) - control_success + 1, samples)
        treatment_draws = np.random.default_rng(43).beta(
            treatment_success + 1, len(treatment_values) - treatment_success + 1, samples
        )
        win_probability = float((treatment_draws > control_draws).mean())
        uplift = float(treatment_draws.mean() - control_draws.mean())
    else:
        rng = np.random.default_rng(44)
        control_draws = rng.normal(control_values.mean(), max(control_values.std(ddof=1), 1e-6), samples)
        treatment_draws = rng.normal(treatment_values.mean(), max(treatment_values.std(ddof=1), 1e-6), samples)
        win_probability = float((treatment_draws > control_draws).mean())
        uplift = float(treatment_draws.mean() - control_draws.mean())

    insights = [
        f"Variant '{treatment}' has {win_probability:.1%} posterior win probability versus '{control}' "
        f"with expected uplift {uplift:.4f}."
    ]
    insight_objects = [
        build_insight(
            tool="bayesian_ab_test",
            title="Bayesian A/B result",
            message="Bayesian uplift analysis completed.",
            evidence={
                "control": control,
                "treatment": treatment,
                "win_probability": win_probability,
                "expected_uplift": uplift,
            },
        )
    ]

    return build_result(
        insights=insights,
        insight_objects=insight_objects,
        charts=[{"type": "distribution", "title": f"Posterior uplift: {treatment} vs {control}"}],
        data={
            "control": control,
            "treatment": treatment,
            "win_probability": win_probability,
            "expected_uplift": uplift,
        },
    )
