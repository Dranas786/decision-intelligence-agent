from __future__ import annotations

from typing import Any

from app.agent.tools import build_tool_step, execute_tool, list_available_tools, load_resource, summarize_resource
from app.analytics.pipeline_3d import PointCloudData
from app.llm.answer_builder import build_final_answer
from app.llm.factory import get_llm_client
from app.rag.factory import get_rag_service


SUPPORTED_DOMAINS = {"general", "finance", "healthcare", "pipeline"}


TOOL_LABELS = {
    "validate_dataset": "validated data quality expectations",
    "profile_table": "profiled the dataset structure and grain",
    "audit_schema_contract": "checked schema contract and version consistency",
    "assess_freshness": "measured data freshness and recency",
    "audit_standardization": "screened label variants and standardization opportunities",
    "detect_entity_collisions": "screened duplicate-entity risk",
    "detect_anomalies": "checked for anomalies and unexpected movement",
    "segment_drivers": "reviewed segment-level contribution and drivers",
    "scan_correlations": "screened numeric relationships",
    "fit_driver_regression": "measured effect sizes with regression",
    "forecast_metric": "built a forward-looking forecast",
    "bayesian_ab_test": "evaluated experiment-style uplift",
    "calculate_returns": "computed return series",
    "measure_risk": "measured risk and volatility",
    "measure_drawdown": "measured downside drawdowns",
    "detect_volume_spikes": "screened for unusual volume",
    "optimize_portfolio": "generated portfolio weights",
    "backtest_signal": "backtested a trading signal",
    "compute_readmission_rate": "measured readmission risk",
    "compare_cohorts": "compared cohort outcomes",
    "analyze_length_of_stay": "measured length-of-stay utilization",
    "survival_risk_analysis": "ran time-to-event analysis",
    "estimate_treatment_effect": "estimated treatment effect",
    "profile_point_cloud": "profiled the point-cloud scan",
    "clean_point_cloud": "cleaned and downsampled the point cloud",
    "fit_pipe_cylinder": "fit the nominal pipe cylinder",
    "compute_pipe_deviation_map": "computed the radial deviation map",
    "detect_pipe_dents": "detected localized dent clusters",
    "measure_pipe_ovality": "measured ovality across pipe slices",
}

def _append_tool_if_missing(plan: list[dict[str, Any]], tool_name: str, args: dict[str, Any] | None = None) -> None:
    if any(step.get("tool") == tool_name for step in plan):
        return
    plan.append({"tool": tool_name, "args": args or {}})



def _enrich_plan(
    plan: list[dict[str, Any]],
    question: str,
    domain: str,
) -> list[dict[str, Any]]:
    enriched_plan = list(plan)
    q = question.lower()

    if domain == "general":
        quality_request = any(word in q for word in ["quality", "governance", "validity", "prepare", "clean", "standardize", "schema"])
        duplicate_request = any(word in q for word in ["duplicate", "duplicates", "entity", "deduplicate", "collision"])

        if quality_request:
            _append_tool_if_missing(enriched_plan, "audit_schema_contract")
            _append_tool_if_missing(enriched_plan, "audit_standardization")
            _append_tool_if_missing(enriched_plan, "assess_freshness")
        if duplicate_request or quality_request:
            _append_tool_if_missing(enriched_plan, "detect_entity_collisions")

        if any(word in q for word in ["anomaly", "anomalies", "outlier", "outliers", "spike", "spikes"]):
            _append_tool_if_missing(enriched_plan, "detect_anomalies")

        if any(word in q for word in ["forecast", "predict", "projection", "project future"]):
            _append_tool_if_missing(enriched_plan, "forecast_metric")

        if any(word in q for word in ["driver", "drivers", "cause", "causes", "factor", "factors"]):
            has_driver_tool = any(
                step.get("tool") in {"segment_drivers", "scan_correlations", "fit_driver_regression"}
                for step in enriched_plan
            )
            if not has_driver_tool:
                _append_tool_if_missing(enriched_plan, "segment_drivers")

    elif domain == "pipeline":
        if any(word in q for word in ["dent", "dents", "deformation", "deformations"]):
            _append_tool_if_missing(enriched_plan, "detect_pipe_dents")

        if any(word in q for word in ["ovality", "roundness", "shape distortion"]):
            _append_tool_if_missing(enriched_plan, "measure_pipe_ovality")

    elif domain == "finance":
        if any(word in q for word in ["drawdown", "downside risk", "loss risk"]):
            _append_tool_if_missing(enriched_plan, "measure_drawdown")

    elif domain == "healthcare":
        if any(word in q for word in ["readmission", "readmissions"]):
            _append_tool_if_missing(enriched_plan, "compute_readmission_rate")

    return enriched_plan



def _tool_action_label(tool_name: str) -> str:
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))



def _build_explanation_layer(
    question: str,
    domain: str,
    resource_summary: dict[str, Any],
    executed_tools: list[dict[str, Any]],
    insights: list[str],
    insight_objects: list[dict[str, Any]],
    diagnostics: list[str],
) -> dict[str, Any]:
    methodology = [
        "Profile the dataset to infer shape, types, grain, and likely keys.",
        "Validate core quality dimensions: completeness, uniqueness, conformity, consistency, validity, and freshness.",
        "Apply deterministic tools only; the model does not compute or clean data by itself.",
        "Surface governance notes such as schema drift, sensitive fields, ambiguous definitions, duplicate-entity risk, and low-confidence areas.",
        "Separate automated actions from items that still require human review.",
    ]

    actions_taken = [
        _tool_action_label(step["tool"])
        for step in executed_tools
        if step.get("status") == "executed"
    ]

    profile_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "profile_table"), {})
    validation_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "validate_dataset"), {})
    schema_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "audit_schema_contract"), {})
    freshness_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "assess_freshness"), {})
    standardization_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "audit_standardization"), {})
    collision_evidence = next((obj.get("evidence", {}) for obj in insight_objects if obj.get("tool") == "detect_entity_collisions"), {})

    governance_notes: list[str] = []
    human_review_required: list[str] = []

    sensitive_fields = profile_evidence.get("sensitive_fields") or validation_evidence.get("sensitive_fields") or []
    ambiguous_columns = profile_evidence.get("ambiguous_columns") or []
    human_review_required.extend(validation_evidence.get("human_review_required") or [])

    if sensitive_fields:
        governance_notes.append(f"Potentially sensitive fields detected by name pattern: {sensitive_fields}.")
    if ambiguous_columns:
        governance_notes.append(f"Some columns may need stronger business definitions: {ambiguous_columns}.")
    if schema_evidence.get("missing_columns"):
        governance_notes.append(f"Schema contract is missing columns required for downstream use: {schema_evidence.get('missing_columns')}.")
        human_review_required.append("Resolve missing schema-contract fields before publishing downstream outputs.")
    if schema_evidence.get("schema_versions") and len(schema_evidence.get("schema_versions", [])) > 1:
        governance_notes.append(
            f"Schema version drift is present in {schema_evidence.get('schema_version_column')}: {schema_evidence.get('schema_versions')}."
        )
        human_review_required.append("Review schema-version drift before combining records into one reporting layer.")
    if freshness_evidence.get("freshness_status") and freshness_evidence.get("freshness_status") != "fresh":
        governance_notes.append(
            f"Data freshness status is {freshness_evidence.get('freshness_status')} with newest record age {freshness_evidence.get('freshness_age_days')} days."
        )
        human_review_required.append("Confirm whether the dataset is current enough for the intended business use.")
    if standardization_evidence.get("standardization_candidates"):
        governance_notes.append("Several text fields contain label variants that should be standardized before reporting.")
    if collision_evidence.get("collision_candidates"):
        governance_notes.append("Potential duplicate-entity groups were found and should be reviewed before deduplicating records.")
        human_review_required.append("Review duplicate-entity candidates before applying entity consolidation rules.")
    governance_notes.extend(diagnostics[:3])

    dataset_profile = {
        "resource_kind": resource_summary.get("resource_kind", domain),
        "row_count": resource_summary.get("row_count"),
        "columns": resource_summary.get("columns", []),
        "likely_keys": profile_evidence.get("likely_keys", []),
        "business_key_candidates": profile_evidence.get("business_key_candidates", []),
        "grain": profile_evidence.get("grain"),
        "schema_status": schema_evidence.get("schema_status"),
        "freshness_status": freshness_evidence.get("freshness_status"),
    }

    quality_findings = insights[:10]

    if domain == "pipeline":
        methodology = [
            "Profile the point cloud before running geometry checks.",
            "Clean and downsample the scan to stabilize later calculations.",
            "Fit a nominal cylinder to establish the expected pipe surface.",
            "Measure deviations from that nominal cylinder to detect dents and ovality.",
            "Flag fit-quality caveats and measurements that require engineering review.",
        ]
        dataset_profile = {
            "resource_kind": resource_summary.get("resource_kind", "point_cloud"),
            "point_count": resource_summary.get("point_count"),
            "bounds": resource_summary.get("bounds"),
            "has_normals": resource_summary.get("has_normals"),
        }

    return {
        "question": question,
        "domain": domain,
        "dataset_profile": dataset_profile,
        "quality_methodology": methodology,
        "quality_findings": quality_findings,
        "actions_taken": actions_taken,
        "governance_notes": list(dict.fromkeys(note for note in governance_notes if note)),
        "human_review_required": list(dict.fromkeys(note for note in human_review_required if note)),
    }



def _build_grounded_answer_input(
    question: str,
    domain: str,
    resource_summary: dict[str, Any],
    executed_tools: list[dict[str, Any]],
    insights: list[str],
    insight_objects: list[dict[str, Any]],
    diagnostics: list[str],
    retrieved_chunks: list[dict[str, Any]],
    explanation_layer: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.append("You are preparing a grounded answer based on deterministic analytics results and retrieved documents.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append(f"Domain: {domain}")
    lines.append("")

    lines.append("Resource summary:")
    lines.append(f"- Row count: {resource_summary.get('row_count', 'unknown')}")
    lines.append(f"- Columns: {resource_summary.get('columns', [])}")
    lines.append("")

    lines.append("Executed tools:")
    if executed_tools:
        for tool in executed_tools:
            lines.append(f"- {tool.get('tool')} ({tool.get('status')})")
    else:
        lines.append("- No tools were executed.")
    lines.append("")

    lines.append("Analytics insights:")
    if insights:
        for idx, insight in enumerate(insights, start=1):
            lines.append(f"{idx}. {insight}")
    else:
        lines.append("No analytics insights were produced.")
    lines.append("")

    lines.append("Structured insight objects:")
    if insight_objects:
        for idx, obj in enumerate(insight_objects[:5], start=1):
            lines.append(f"{idx}. {obj}")
    else:
        lines.append("No structured insight objects were produced.")
    lines.append("")

    lines.append("Explanation layer:")
    lines.append(f"- Dataset profile: {explanation_layer.get('dataset_profile', {})}")
    lines.append(f"- Quality methodology: {explanation_layer.get('quality_methodology', [])}")
    lines.append(f"- Actions taken: {explanation_layer.get('actions_taken', [])}")
    lines.append(f"- Governance notes: {explanation_layer.get('governance_notes', [])}")
    lines.append(f"- Human review required: {explanation_layer.get('human_review_required', [])}")
    lines.append("")

    lines.append("Diagnostics:")
    if diagnostics:
        for idx, item in enumerate(diagnostics, start=1):
            lines.append(f"{idx}. {item}")
    else:
        lines.append("No diagnostics.")
    lines.append("")

    lines.append("Retrieved document context:")
    if retrieved_chunks:
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            lines.append(
                f"[Source {idx}] document_id={chunk.get('document_id')} "
                f"chunk_id={chunk.get('chunk_id')} score={chunk.get('score')}"
            )
            lines.append(chunk.get("text", ""))
            lines.append("")
    else:
        lines.append("No external document context retrieved.")
        lines.append("")

    lines.append("Use the analytics results as the primary source of computed truth.")
    lines.append("Use retrieved document text as supporting business/domain context.")
    lines.append("Explain what changed, why it matters, what governance concerns remain, and what still needs human review.")
    lines.append("Do not invent calculations that are not present above.")

    return "\n".join(lines)



def _build_analysis_brief(
    question: str,
    insights: list[str],
    insight_objects: list[dict[str, Any]],
    diagnostics: list[str],
    retrieved_chunks: list[dict[str, Any]],
    explanation_layer: dict[str, Any],
) -> dict[str, Any]:
    question_lower = question.lower()

    if any(x in question_lower for x in ["forecast", "predict", "future"]):
        question_type = "forecasting"
    elif any(x in question_lower for x in ["anomaly", "outlier", "spike"]):
        question_type = "anomaly_detection"
    elif any(x in question_lower for x in ["driver", "cause", "factor"]):
        question_type = "driver_analysis"
    elif any(x in question_lower for x in ["quality", "governance", "validity"]):
        question_type = "data_quality_governance"
    else:
        question_type = "general_analysis"

    document_context = [
        {
            "document_id": chunk.get("document_id"),
            "text": chunk.get("text"),
        }
        for chunk in retrieved_chunks[:3]
    ]

    return {
        "question": question,
        "question_type": question_type,
        "key_findings": insights[:5],
        "supporting_evidence": insight_objects[:5],
        "document_context": document_context,
        "risks_or_caveats": diagnostics[:5],
        "recommended_next_steps": explanation_layer.get("human_review_required", [])[:5],
        "explanation_layer": explanation_layer,
    }



def run_agent(
    dataset_path: str,
    question: str,
    table_name: str = "data",
    domain: str | None = None,
    semantic_config: dict[str, Any] | None = None,
    tool_whitelist: list[str] | None = None,
    analysis_params: dict[str, Any] | None = None,
    use_rag: bool = False,
    rag_limit: int = 3,
) -> dict[str, Any]:
    semantic_config = semantic_config or {}
    analysis_params = analysis_params or {}
    resolved_domain = domain or "general"

    if resolved_domain not in SUPPORTED_DOMAINS:
        raise ValueError(f"Unsupported domain '{resolved_domain}'. Use one of: {sorted(SUPPORTED_DOMAINS)}.")

    resource = load_resource(dataset_path, domain=resolved_domain)
    available_tools = list_available_tools(domain=resolved_domain, tool_whitelist=tool_whitelist)
    llm = get_llm_client()
    resource_summary = summarize_resource(resource)

    tool_context = {
        "available_tools": available_tools,
        "question": question,
        "columns": resource_summary.get("columns", []),
        "row_count": resource_summary.get("row_count", 0),
        "domain": resolved_domain,
        "semantic_config": semantic_config,
        "analysis_params": analysis_params,
        "table_name": table_name,
        "resource_summary": resource_summary,
    }

    plan_response = llm.plan(tool_context)
    raw_plan = plan_response.get("plan", [])
    plan = _enrich_plan(raw_plan, question=question, domain=resolved_domain)

    insights: list[str] = []
    insight_objects: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    diagnostics: list[str] = list(plan_response.get("diagnostics", []))
    executed_tools: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    retrieved_chunks: list[dict[str, Any]] = []
    rag_prompt: str | None = None

    if isinstance(resource, PointCloudData):
        state["raw_point_cloud"] = resource

    for step in plan:
        tool_name = step.get("tool")
        built_step, invalid_args = build_tool_step(
            tool_name,
            resource,
            semantic_config=semantic_config,
            analysis_params={**analysis_params, **step.get("args", {})},
        )

        if invalid_args or not built_step["usable"]:
            diagnostics.append(f"Skipped {tool_name}: invalid arguments {invalid_args}.")
            executed_tools.append({"tool": tool_name, "status": "skipped", "args": built_step["args"]})
            continue

        result = execute_tool(tool_name, resource, built_step["args"], state=state)
        artifacts = result.get("artifacts", {})
        if artifacts:
            state.update(artifacts)

        insights.extend(result.get("insights", []))
        insight_objects.extend(result.get("insight_objects", []))
        charts.extend(result.get("charts", []))
        diagnostics.extend(result.get("diagnostics", []))
        executed_tools.append({"tool": tool_name, "status": "executed", "args": built_step["args"]})

    if use_rag:
        try:
            rag_service = get_rag_service()
            rag_results = rag_service.retrieve_chunks(question=question, limit=rag_limit)

            retrieved_chunks = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in rag_results
            ]

            rag_prompt = rag_service.build_prompt_for_question(
                question=question,
                limit=rag_limit,
                retrieved_chunks=rag_results,
            )

            if not retrieved_chunks:
                diagnostics.append("RAG was enabled, but no supporting document chunks were retrieved.")
        except Exception as exc:
            diagnostics.append(f"RAG retrieval failed: {exc}")

    explanation_layer = _build_explanation_layer(
        question=question,
        domain=resolved_domain,
        resource_summary=resource_summary,
        executed_tools=executed_tools,
        insights=insights,
        insight_objects=insight_objects,
        diagnostics=diagnostics,
    )

    combined_context = {
        "question": question,
        "domain": resolved_domain,
        "dataset_path": dataset_path,
        "table_name": table_name,
        "resource_summary": resource_summary,
        "plan": [step["tool"] for step in plan],
        "executed_tools": executed_tools,
        "insights": insights,
        "insight_objects": insight_objects,
        "diagnostics": diagnostics,
        "retrieved_chunks": retrieved_chunks,
        "rag_used": use_rag,
        "explanation_layer": explanation_layer,
    }

    analysis_brief = _build_analysis_brief(
        question=question,
        insights=insights,
        insight_objects=insight_objects,
        diagnostics=diagnostics,
        retrieved_chunks=retrieved_chunks,
        explanation_layer=explanation_layer,
    )

    grounded_answer_input = _build_grounded_answer_input(
        question=question,
        domain=resolved_domain,
        resource_summary=resource_summary,
        executed_tools=executed_tools[:10],
        insights=insights[:8],
        insight_objects=insight_objects[:5],
        diagnostics=diagnostics[:6],
        retrieved_chunks=retrieved_chunks[:rag_limit],
        explanation_layer=explanation_layer,
    )

    combined_context["analysis_brief"] = analysis_brief

    final_answer = build_final_answer(
        question=question,
        grounded_answer_input=grounded_answer_input,
        combined_context=combined_context,
    )
    return {
        "plan": [step["tool"] for step in plan],
        "executed_tools": executed_tools,
        "insights": insights,
        "insight_objects": insight_objects,
        "diagnostics": diagnostics,
        "charts": charts,
        "rag_used": use_rag,
        "retrieved_chunks": retrieved_chunks,
        "rag_prompt": rag_prompt,
        "combined_context": combined_context,
        "grounded_answer_input": grounded_answer_input,
        "final_answer": final_answer,
        "analysis_brief": analysis_brief,
        "explanation_layer": explanation_layer,
    }






