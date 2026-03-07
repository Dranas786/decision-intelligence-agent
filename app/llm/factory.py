from __future__ import annotations

from typing import Any


class RuleBasedPlanner:
    """
    Lightweight planner that selects tools using question keywords, domain, and semantic hints.
    """

    def plan(self, tool_context: dict[str, Any]) -> dict[str, Any]:
        available_tools = tool_context.get("available_tools", [])
        available_names = [tool["name"] for tool in available_tools]
        question = (tool_context.get("question") or "").lower()
        domain = tool_context.get("domain") or "general"
        semantic_config = tool_context.get("semantic_config") or {}

        plan: list[dict[str, Any]] = []
        diagnostics: list[str] = []

        def maybe_add(tool_name: str) -> None:
            if tool_name in available_names and tool_name not in [step["tool"] for step in plan]:
                plan.append({"tool": tool_name, "args": {}})

        if domain == "pipeline":
            for tool_name in [
                "profile_point_cloud",
                "clean_point_cloud",
                "fit_pipe_cylinder",
                "compute_pipe_deviation_map",
            ]:
                maybe_add(tool_name)
            if any(keyword in question for keyword in ("dent", "deformation", "damage", "collapse", "defect")):
                maybe_add("detect_pipe_dents")
            if any(keyword in question for keyword in ("ovality", "roundness", "shape", "cross-section", "cross section")):
                maybe_add("measure_pipe_ovality")
        else:
            maybe_add("validate_dataset")
            maybe_add("profile_table")

            if domain == "finance":
                for tool_name in [
                    "calculate_returns",
                    "measure_risk",
                    "measure_drawdown",
                    "detect_volume_spikes",
                ]:
                    maybe_add(tool_name)
                if any(keyword in question for keyword in ("portfolio", "allocation", "optimize", "weights")):
                    maybe_add("optimize_portfolio")
                if any(keyword in question for keyword in ("signal", "backtest", "strategy")) or semantic_config.get("signal_col"):
                    maybe_add("backtest_signal")
            elif domain == "healthcare":
                for tool_name in [
                    "compute_readmission_rate",
                    "compare_cohorts",
                    "analyze_length_of_stay",
                    "survival_risk_analysis",
                    "estimate_treatment_effect",
                ]:
                    maybe_add(tool_name)
            else:
                if any(keyword in question for keyword in ("anomaly", "drop", "spike", "change", "trend")):
                    maybe_add("detect_anomalies")
                if any(keyword in question for keyword in ("segment", "channel", "cohort", "driver", "group")):
                    maybe_add("segment_drivers")
                if any(keyword in question for keyword in ("correlation", "driver", "root cause", "relationship")):
                    maybe_add("scan_correlations")
                    maybe_add("fit_driver_regression")
                if any(keyword in question for keyword in ("forecast", "predict", "projection", "future")):
                    maybe_add("forecast_metric")
                if any(keyword in question for keyword in ("experiment", "a/b", "ab test", "variant", "uplift")):
                    maybe_add("bayesian_ab_test")

        if len(plan) <= 1:
            diagnostics.append("Planner selected only default tools because the question did not strongly match a specialized workflow.")

        return {"plan": plan, "diagnostics": diagnostics}



def get_llm_client() -> RuleBasedPlanner:
    return RuleBasedPlanner()
