from __future__ import annotations

from typing import Any

from app.agent.tools import build_tool_step, execute_tool, list_available_tools, load_resource, summarize_resource
from app.analytics.pipeline_3d import PointCloudData
from app.llm.answer_builder import build_final_answer
from app.llm.factory import get_llm_client
from app.rag.factory import get_rag_service


SUPPORTED_DOMAINS = {"general", "finance", "healthcare", "pipeline"}


def _append_tool_if_missing(plan: list[dict[str, Any]], tool_name: str, args: dict[str, Any] | None = None) -> None:
    """Add a tool step only if it is not already present in the plan."""
    if any(step.get("tool") == tool_name for step in plan):
        return
    plan.append({"tool": tool_name, "args": args or {}})


def _enrich_plan(
    plan: list[dict[str, Any]],
    question: str,
    domain: str,
) -> list[dict[str, Any]]:
    """
    Apply simple rule-based corrections so explicit user intent is not missed
    even if the LLM planner omits an obvious tool.
    """
    enriched_plan = list(plan)
    q = question.lower()

    if domain == "general":
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


def _build_grounded_answer_input(
    question: str,
    domain: str,
    resource_summary: dict[str, Any],
    executed_tools: list[dict[str, Any]],
    insights: list[str],
    insight_objects: list[dict[str, Any]],
    diagnostics: list[str],
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Build one combined text block that gathers analytics results and RAG evidence
    into a single grounded answer context for a later summarizer or final answer model.
    """
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
    lines.append("Do not invent calculations that are not present above.")

    return "\n".join(lines)


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
    """
    Main agent workflow:
    1. Load the analysis resource
    2. Ask planner for a tool plan
    3. Enrich the plan with simple intent-based safeguards
    4. Execute analytics tools from the registry
    5. Optionally retrieve supporting RAG context
    6. Build one combined grounded context bundle
    7. Generate a final grounded answer
    8. Return structured insights
    """

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

            rag_prompt = rag_service.build_prompt_for_question(question=question, limit=rag_limit)

            if not retrieved_chunks:
                diagnostics.append("RAG was enabled, but no supporting document chunks were retrieved.")
        except Exception as exc:
            diagnostics.append(f"RAG retrieval failed: {exc}")

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
    }

    grounded_answer_input = _build_grounded_answer_input(
        question=question,
        domain=resolved_domain,
        resource_summary=resource_summary,
        executed_tools=executed_tools[:10],   # limit tool history
        insights=insights[:8],                # limit insights
        insight_objects=insight_objects[:5],  # limit structured objects
        diagnostics=diagnostics[:6],          # limit diagnostics
        retrieved_chunks=retrieved_chunks[:rag_limit],  # keep retrieval size bounded
    )

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
    }