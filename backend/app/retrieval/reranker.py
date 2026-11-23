# Responsibility: Cross-encoder reranking
# This file receives candidate chunks from the retriever.
# - Loads cross-encoder/ms-marco-MiniLM-L-6-v2.
# - Creates question–chunk pairs.
# - Predicts a relevance score for each pair.
# - Sorts candidates by their scores.
# - Returns the top-ranked chunks.

from sentence_transformers import CrossEncoder

from backend.app.config import RERANKER_MODEL_NAME


def load_reranker() -> CrossEncoder:
    """Load the cross-encoder model."""
    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_chunks(
    question: str,
    retrieved_chunks: list[dict],
    reranker: CrossEncoder,
    top_k: int = 5,
) -> list[dict]:
    """Rerank retrieved chunks by question–content relevance."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not retrieved_chunks:
        return []

    question_chunk_pairs = [
        (question, chunk["content"])
        for chunk in retrieved_chunks
    ]

    relevance_scores = reranker.predict(question_chunk_pairs)

    reranked_chunks = []

    for chunk, score in zip(retrieved_chunks, relevance_scores):
        reranked_chunk = chunk.copy()
        reranked_chunk["rerank_score"] = float(score)
        reranked_chunks.append(reranked_chunk)

    reranked_chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True,
    )

    return reranked_chunks[:top_k]