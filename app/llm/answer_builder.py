from __future__ import annotations

from typing import Any

from app.llm.factory import get_llm_client


def _build_structured_fallback(combined_context: dict[str, Any]) -> str:
    explanation_layer = combined_context.get("explanation_layer", {}) or {}
    insights = combined_context.get("insights", [])
    diagnostics = combined_context.get("diagnostics", [])
    retrieved_chunks = combined_context.get("retrieved_chunks", [])

    fallback_parts: list[str] = []

    dataset_profile = explanation_layer.get("dataset_profile", {})
    if dataset_profile:
        fallback_parts.append("Dataset profile:")
        for key, value in dataset_profile.items():
            if value not in (None, [], {}):
                fallback_parts.append(f"- {key}: {value}")

    quality_findings = explanation_layer.get("quality_findings") or insights[:5]
    if quality_findings:
        if fallback_parts:
            fallback_parts.append("")
        fallback_parts.append("Quality issues found:")
        for item in quality_findings[:6]:
            fallback_parts.append(f"- {item}")

    actions_taken = explanation_layer.get("actions_taken", [])
    if actions_taken:
        fallback_parts.append("")
        fallback_parts.append("Actions taken:")
        for item in actions_taken[:6]:
            fallback_parts.append(f"- {item}")

    governance_notes = explanation_layer.get("governance_notes", [])
    if governance_notes:
        fallback_parts.append("")
        fallback_parts.append("Governance notes:")
        for item in governance_notes[:6]:
            fallback_parts.append(f"- {item}")

    human_review_required = explanation_layer.get("human_review_required", [])
    if human_review_required:
        fallback_parts.append("")
        fallback_parts.append("Human review required:")
        for item in human_review_required[:6]:
            fallback_parts.append(f"- {item}")

    if retrieved_chunks:
        fallback_parts.append("")
        fallback_parts.append("Supporting context was retrieved from the document store and used only as domain guidance.")

    if diagnostics:
        fallback_parts.append("")
        fallback_parts.append("Diagnostics:")
        for item in diagnostics[:3]:
            fallback_parts.append(f"- {item}")

    if not fallback_parts:
        fallback_parts.append("No grounded answer could be generated, but the analysis completed.")

    return "\n".join(fallback_parts)


def build_final_answer(
    question: str,
    grounded_answer_input: str,
    combined_context: dict[str, Any] | None = None,
) -> str:
    """
    Build a final grounded answer using the assembled analytics + RAG context.

    This function should never perform calculations itself.
    It only turns already-grounded context into a readable final answer.
    """

    combined_context = combined_context or {}

    prompt = "\n".join(
        [
            "You are a decision intelligence assistant focused on data quality, governance, and explainable analytics.",
            "Write a clear final answer using only the grounded context below.",
            "Prioritize deterministic analytics findings as the source of computed truth.",
            "Use retrieved document context only as supporting policy, domain, or business guidance.",
            "Do not invent calculations, facts, thresholds, or conclusions not present in the provided context.",
            "When relevant, explain the methodology in plain language so a stakeholder can trust what was done.",
            "Prefer this structure when applicable: Dataset profile, Quality issues found, Actions taken, Governance notes, Human review required, Recommendation.",
            "Keep the answer concise but specific.",
            "",
            f"User question: {question}",
            "",
            "Grounded context:",
            grounded_answer_input,
        ]
    )

    llm = None
    try:
        llm = get_llm_client()
    except Exception:
        llm = None

    if llm is not None:
        try:
            response = llm.generate(
                {
                    "question": question,
                    "prompt": prompt,
                    "combined_context": combined_context,
                }
            )

            if isinstance(response, dict):
                answer = response.get("answer") or response.get("text") or response.get("output")
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()

            if isinstance(response, str) and response.strip():
                return response.strip()

        except Exception:
            pass

    return _build_structured_fallback(combined_context)
