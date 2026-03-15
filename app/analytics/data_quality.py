from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from app.analytics.common import build_insight, build_result


ENTITY_HINTS = (
    "id",
    "name",
    "email",
    "phone",
    "store",
    "customer",
    "account",
    "location",
    "neighbour",
    "government",
    "address",
)
SCHEMA_VERSION_HINTS = ("schema_version", "version", "schema")


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _object_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if df[column].dtype == "object"]


def _find_version_column(df: pd.DataFrame) -> str | None:
    for column in df.columns:
        lowered = column.lower()
        if lowered in SCHEMA_VERSION_HINTS or lowered.endswith("_version"):
            return column
    return None


def audit_schema_contract(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    expected_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    required_columns = required_columns or []
    expected_types = expected_types or {}

    missing_columns = [column for column in required_columns if column not in df.columns]
    unexpected_columns = [column for column in df.columns if required_columns and column not in required_columns]
    actual_types = {column: str(dtype) for column, dtype in df.dtypes.items()}

    type_mismatches: list[dict[str, str]] = []
    for column, expected in expected_types.items():
        if column not in df.columns:
            continue
        actual = actual_types[column]
        if expected.lower() not in actual.lower():
            type_mismatches.append({"column": column, "expected": expected, "actual": actual})

    version_column = _find_version_column(df)
    schema_versions: list[str] = []
    if version_column is not None:
        schema_versions = (
            df[version_column]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

    schema_status = "healthy"
    if missing_columns or type_mismatches:
        schema_status = "warning"
    if len(schema_versions) > 1:
        schema_status = "review_required"

    data = {
        "schema_status": schema_status,
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns[:20],
        "actual_types": actual_types,
        "type_mismatches": type_mismatches,
        "schema_version_column": version_column,
        "schema_versions": schema_versions,
    }

    insights = [
        f"Schema contract status is {schema_status}. Missing columns: {missing_columns or 'none'}."
    ]
    if type_mismatches:
        insights.append(f"Type mismatches detected: {type_mismatches}.")
    if len(schema_versions) > 1:
        insights.append(
            f"Schema version drift detected in {version_column}: {schema_versions}."
        )

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="audit_schema_contract",
                title="Schema contract",
                message="Schema presence, type alignment, and schema-version consistency were checked.",
                evidence=data,
                severity="warning" if schema_status != "healthy" else "info",
            )
        ],
        data=data,
        charts=[{"type": "schema", "title": "Schema contract summary"}],
    )


def assess_freshness(
    df: pd.DataFrame,
    time_col: str,
    warn_after_days: float = 2.0,
    error_after_days: float = 7.0,
) -> dict[str, Any]:
    if time_col not in df.columns:
        return build_result(
            insights=["Freshness assessment skipped because the configured time column was not found."],
            diagnostics=[f"Time column '{time_col}' is missing."],
        )

    parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    valid = parsed.dropna()
    if valid.empty:
        return build_result(
            insights=[f"Freshness assessment skipped because '{time_col}' could not be parsed."],
            diagnostics=[f"No parseable timestamps were found in '{time_col}'."],
        )

    latest = valid.max()
    earliest = valid.min()
    now = pd.Timestamp.utcnow()
    age_days = float((now - latest).total_seconds() / 86400.0)
    parse_failure_share = float((parsed.isna() & df[time_col].notna()).mean())

    freshness_status = "fresh"
    if age_days > error_after_days:
        freshness_status = "error"
    elif age_days > warn_after_days:
        freshness_status = "warning"

    data = {
        "time_col": time_col,
        "freshness_status": freshness_status,
        "latest_timestamp": latest.isoformat(),
        "earliest_timestamp": earliest.isoformat(),
        "freshness_age_days": round(age_days, 4),
        "warn_after_days": warn_after_days,
        "error_after_days": error_after_days,
        "parse_failure_share": round(parse_failure_share, 4),
    }

    insights = [
        f"Freshness status is {freshness_status}; the newest record in {time_col} is {age_days:.2f} days old."
    ]
    if parse_failure_share > 0:
        insights.append(
            f"{time_col} has parse failures in {parse_failure_share:.1%} of non-null rows, which weakens freshness monitoring."
        )

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="assess_freshness",
                title="Freshness check",
                message="Recency of the most recent record was compared against warning and error thresholds.",
                evidence=data,
                severity="warning" if freshness_status != "fresh" else "info",
            )
        ],
        data=data,
        charts=[{"type": "freshness", "title": "Freshness status"}],
    )


def audit_standardization(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    target_columns = [column for column in (columns or _object_columns(df)) if column in df.columns][:10]
    opportunities: list[dict[str, Any]] = []

    for column in target_columns:
        raw = df[column].dropna().astype(str)
        if raw.empty or raw.nunique() < 2:
            continue
        normalized = _normalize_text(raw)
        if normalized.nunique() >= raw.nunique():
            continue

        candidate_groups: list[dict[str, Any]] = []
        grouped = (
            pd.DataFrame({"raw": raw, "normalized": normalized})
            .groupby("normalized")["raw"]
            .agg(lambda values: sorted(set(values))[:5])
        )
        for normalized_value, raw_variants in grouped.items():
            if len(raw_variants) > 1:
                candidate_groups.append(
                    {
                        "normalized_value": normalized_value,
                        "raw_variants": raw_variants,
                        "variant_count": len(raw_variants),
                    }
                )

        if candidate_groups:
            opportunities.append(
                {
                    "column": column,
                    "raw_unique_count": int(raw.nunique()),
                    "normalized_unique_count": int(normalized.nunique()),
                    "variant_groups": candidate_groups[:3],
                }
            )

    data = {"standardization_candidates": opportunities[:top_n]}
    insights = (
        [f"Standardization opportunities detected in {[item['column'] for item in opportunities[:top_n]]}."]
        if opportunities
        else ["No obvious text-standardization opportunities were detected in the reviewed fields."]
    )

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="audit_standardization",
                title="Standardization review",
                message="Object columns were normalized to detect label variants and candidate mapping rules.",
                evidence=data,
            )
        ],
        data=data,
        charts=[{"type": "standardization", "title": "Standardization opportunities"}],
    )


def detect_entity_collisions(
    df: pd.DataFrame,
    entity_columns: list[str] | None = None,
    min_group_size: int = 2,
    top_n: int = 5,
) -> dict[str, Any]:
    preferred_columns = entity_columns or [
        column
        for column in df.columns
        if any(token in column.lower() for token in ENTITY_HINTS)
    ]
    candidate_columns = [column for column in preferred_columns if column in df.columns][:5]
    if len(candidate_columns) < 2:
        return build_result(
            insights=["Entity-collision review skipped because there were not enough identifying columns."],
            diagnostics=["Provide at least two useful entity columns to evaluate duplicate-entity risk."],
        )

    collision_groups: list[dict[str, Any]] = []
    combos = list(combinations(candidate_columns, min(2, len(candidate_columns))))
    reviewed_columns = set()

    for combo in combos:
        combo = list(combo)
        combo_key = tuple(combo)
        if combo_key in reviewed_columns:
            continue
        reviewed_columns.add(combo_key)

        working = df[combo].copy()
        for column in combo:
            working[column] = _normalize_text(working[column])
        grouped = working.groupby(combo, dropna=False).size().reset_index(name="row_count")
        duplicates = grouped[grouped["row_count"] >= min_group_size]
        if duplicates.empty:
            continue

        for _, row in duplicates.head(top_n).iterrows():
            match_mask = pd.Series(True, index=df.index)
            preview_filters: dict[str, str] = {}
            for column in combo:
                normalized = str(row[column])
                preview_filters[column] = normalized
                match_mask &= _normalize_text(df[column]) == normalized
            preview = df.loc[match_mask, candidate_columns].head(3).to_dict(orient="records")
            collision_groups.append(
                {
                    "match_columns": combo,
                    "normalized_key": preview_filters,
                    "row_count": int(row["row_count"]),
                    "sample_rows": preview,
                }
            )

    collision_groups = collision_groups[:top_n]
    data = {"collision_candidates": collision_groups, "reviewed_columns": candidate_columns}
    insights = (
        [f"Potential duplicate-entity groups were found using normalized keys across {candidate_columns}."]
        if collision_groups
        else ["No obvious duplicate-entity groups were found using the reviewed identifying columns."]
    )

    return build_result(
        insights=insights,
        insight_objects=[
            build_insight(
                tool="detect_entity_collisions",
                title="Duplicate-entity review",
                message="Potential duplicate entities were screened using normalized identifying fields.",
                evidence=data,
                severity="warning" if collision_groups else "info",
            )
        ],
        data=data,
        charts=[{"type": "entity_collision", "title": "Duplicate-entity candidates"}],
    )
