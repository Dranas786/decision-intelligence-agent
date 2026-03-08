from __future__ import annotations

from app.rag.schemas import RetrievedChunk


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """
    Build a grounded prompt using the user question and retrieved chunks.
    """

    cleaned_question = question.strip()

    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        return (
            "You are a helpful assistant.\n\n"
            f"User question:\n{cleaned_question}\n\n"
            "No external context was retrieved."
        )

    context_sections: list[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        source_label = chunk.metadata.get("title", chunk.document_id)

        context_sections.append(
            f"[Source {index}] {source_label}\n"
            f"Score: {chunk.score:.4f}\n"
            f"{chunk.text}"
        )

    context_block = "\n\n".join(context_sections)

    prompt = (
        "You are a helpful assistant.\n"
        "Use the retrieved context below to answer the user's question.\n"
        "If the answer is not supported by the context, say that clearly.\n\n"
        f"User question:\n{cleaned_question}\n\n"
        f"Retrieved context:\n{context_block}"
    )

    return prompt