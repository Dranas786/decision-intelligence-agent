from __future__ import annotations

from typing import Any

from app.llm.factory import get_llm_client


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
            "You are a decision intelligence assistant.",
            "Write a clear final answer using only the grounded context below.",
            "Prioritize deterministic analytics findings as the source of computed truth.",
            "Use retrieved document context only as supporting policy, domain, or business guidance.",
            "Do not invent calculations, facts, thresholds, or conclusions not present in the provided context.",
            "",
            f"User question: {question}",
            "",
            "Grounded context:",
            grounded_answer_input,
            "",
            "Write a concise but useful answer.",
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

    insights = combined_context.get("insights", [])
    retrieved_chunks = combined_context.get("retrieved_chunks", [])
    diagnostics = combined_context.get("diagnostics", [])

    fallback_parts: list[str] = []

    if insights:
        fallback_parts.append("Key findings:")
        for item in insights[:5]:
            fallback_parts.append(f"- {item}")

    if retrieved_chunks:
        fallback_parts.append("")
        fallback_parts.append("Supporting document context was retrieved and should be considered alongside the analytics results.")

    if diagnostics:
        fallback_parts.append("")
        fallback_parts.append("Diagnostics:")
        for item in diagnostics[:3]:
            fallback_parts.append(f"- {item}")

    if not fallback_parts:
        fallback_parts.append("No grounded answer could be generated, but the analysis completed.")

    return "\n".join(fallback_parts)
