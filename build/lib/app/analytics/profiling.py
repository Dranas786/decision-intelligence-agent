from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result


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

    summary = (
        f"Dataset has {row_count} rows and {column_count} columns, duplicate rate {duplicate_rate:.1%}. "
        f"Numeric columns: {numeric_columns}. Categorical/date-like columns: {categorical_columns}."
    )

    return build_result(
        insights=[summary],
        insight_objects=[
            build_insight(
                tool="profile_table",
                title="Profile summary",
                message="Dataset profiling completed.",
                evidence={
                    "row_count": row_count,
                    "column_count": column_count,
                    "numeric_columns": numeric_columns,
                    "categorical_columns": categorical_columns,
                    "duplicate_rate": duplicate_rate,
                },
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
        },
    )
