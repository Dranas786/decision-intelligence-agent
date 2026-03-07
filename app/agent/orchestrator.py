from __future__ import annotations

from typing import Any

from app.agent.tools import build_tool_step, execute_tool, list_available_tools, load_resource, summarize_resource
from app.analytics.pipeline_3d import PointCloudData
from app.llm.factory import get_llm_client


SUPPORTED_DOMAINS = {"general", "finance", "healthcare", "pipeline"}



def run_agent(
    dataset_path: str,
    question: str,
    table_name: str = "data",
    domain: str | None = None,
    semantic_config: dict[str, Any] | None = None,
    tool_whitelist: list[str] | None = None,
    analysis_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Main agent workflow:
    1. Load the analysis resource
    2. Ask planner for a tool plan
    3. Execute analytics tools from the registry
    4. Return structured insights
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
    plan = plan_response.get("plan", [])

    insights: list[str] = []
    insight_objects: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    diagnostics: list[str] = list(plan_response.get("diagnostics", []))
    executed_tools: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

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

    return {
        "plan": [step["tool"] for step in plan],
        "executed_tools": executed_tools,
        "insights": insights,
        "insight_objects": insight_objects,
        "diagnostics": diagnostics,
        "charts": charts,
    }
