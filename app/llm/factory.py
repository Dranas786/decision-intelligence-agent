from __future__ import annotations

import json
import os
from typing import Any
from urllib import request


class RuleBasedPlanner:
    """
    Fallback planner if the real model is unavailable.
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
            diagnostics.append(
                "Planner selected only default tools because the question did not strongly match a specialized workflow."
            )

        return {"plan": plan, "diagnostics": diagnostics}

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        combined_context = payload.get("combined_context", {}) or {}
        insights = combined_context.get("insights", [])
        diagnostics = combined_context.get("diagnostics", [])

        fallback_lines: list[str] = []
        if insights:
            fallback_lines.append("Key findings:")
            for item in insights[:5]:
                fallback_lines.append(f"- {item}")
        if diagnostics:
            fallback_lines.append("")
            fallback_lines.append("Diagnostics:")
            for item in diagnostics[:3]:
                fallback_lines.append(f"- {item}")

        if not fallback_lines:
            fallback_lines.append("Analysis completed, but no model-generated answer was available.")

        return {"answer": "\n".join(fallback_lines)}


class GroqClient:
    """
    Groq OpenAI-compatible client using the chat completions endpoint.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.base_url = (os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
        self.planner_model = os.getenv("GROQ_PLANNER_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        self.answer_model = os.getenv("GROQ_ANSWER_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        self.timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()

        return ""

    def plan(self, tool_context: dict[str, Any]) -> dict[str, Any]:
        available_tools = tool_context.get("available_tools", [])
        valid_tool_names = {tool.get("name") for tool in available_tools}

        prompt = "\n".join(
            [
                "You are the planner for a decision intelligence agent.",
                "Choose a SHORT ordered list of deterministic tools.",
                "The Python tools do all real calculations.",
                "Never invent tool names.",
                "Prefer validation/profile first when useful, core analysis next, forecasting last if requested.",
                "Only use tools from the available list.",
                "Return ONLY valid JSON with keys: plan, diagnostics.",
                'Each plan item must be an object like {"tool": "...", "args": {...}}.',
                "",
                f"Question: {tool_context.get('question', '')}",
                f"Domain: {tool_context.get('domain', 'general')}",
                f"Semantic config: {json.dumps(tool_context.get('semantic_config', {}))}",
                f"Analysis params: {json.dumps(tool_context.get('analysis_params', {}))}",
                f"Resource summary: {json.dumps(tool_context.get('resource_summary', {}))}",
                f"Available tools: {json.dumps(available_tools)}",
            ]
        )

        response = self._post_chat(
            {
                "model": self.planner_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only compact JSON. Do not include markdown fences or extra explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
        )

        raw = self._extract_text(response)
        parsed = json.loads(raw)

        cleaned_plan: list[dict[str, Any]] = []
        for step in parsed.get("plan", []):
            tool_name = step.get("tool")
            if tool_name in valid_tool_names:
                args = step.get("args", {})
                cleaned_plan.append(
                    {
                        "tool": tool_name,
                        "args": args if isinstance(args, dict) else {},
                    }
                )

        diagnostics = parsed.get("diagnostics", [])
        if not isinstance(diagnostics, list):
            diagnostics = []

        return {"plan": cleaned_plan, "diagnostics": diagnostics}

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._post_chat(
            {
                "model": self.answer_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Write a grounded final answer using only the provided analytics and document context.",
                    },
                    {"role": "user", "content": payload.get("prompt", "")},
                ],
                "temperature": 0.2,
            }
        )
        return {"answer": self._extract_text(response)}


class HybridLLMClient:
    def __init__(self) -> None:
        self.fallback = RuleBasedPlanner()
        self.primary = GroqClient()

    def plan(self, tool_context: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.primary.plan(tool_context)
            if result.get("plan"):
                return result
        except Exception:
            pass
        return self.fallback.plan(tool_context)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self.primary.generate(payload)
            answer = result.get("answer", "")
            if isinstance(answer, str) and answer.strip():
                return result
        except Exception:
            pass
        return self.fallback.generate(payload)


_llm_client: HybridLLMClient | None = None


def get_llm_client() -> HybridLLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = HybridLLMClient()
    return _llm_client