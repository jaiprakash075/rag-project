# Responsibility: Orchestrate the entire RAG pipeline

# This file connects all the existing components.

# Its flow should be:
# Receive question
#       |
#       v
# Check Redis cache
#       |
#     Hit? ---- Yes ---> Return answer
#       |
#       No
#       |
#       v
# Retrieve chunks
#       |
#       v
# Rerank chunks
#       |
#       v
# Generate Gemini answer
#       |
#       v
# Save answer in Redis
#       |
#       v
# Return answer + sources




import logging
from typing import Any

from backend.app.caching.semantic_caching import SemanticCache
from backend.app.generation.answer_generator import generate_answer
from backend.app.retrieval.reranker import load_reranker, rerank_chunks
from backend.app.retrieval.retriever import load_retriever, retrieve_chunks

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates caching, hybrid retrieval, reranking, and Gemini generation."""

    def __init__(self, use_cache: bool = True):
        (
            self.embedding_model,
            self.collection,
            self.bm25_index,
            self.corpus_records,
            self.records_by_id,
        ) = load_retriever()

        self.reranker = load_reranker()
        self.cache = (
            SemanticCache(embedding_model=self.embedding_model)
            if use_cache
            else None
        )

    def answer_question(
        self,
        question: str,
        retrieve_top_k: int = 10,
        rerank_top_k: int = 5,
        generate_top_n: int = 3,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Execute the full RAG workflow for a single user question."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        # 1. Check semantic cache
        if use_cache and self.cache is not None:
            cached_result = self.cache.get(cleaned_question)
            if cached_result is not None:
                return {
                    "answer": cached_result["answer"],
                    "sources": cached_result.get("sources", []),
                    "cached": True,
                }

        # 2. Hybrid retrieval (Semantic + BM25)
        retrieved_chunks = retrieve_chunks(
            question=cleaned_question,
            embedding_model=self.embedding_model,
            collection=self.collection,
            bm25_index=self.bm25_index,
            corpus_records=self.corpus_records,
            records_by_id=self.records_by_id,
            top_k=retrieve_top_k,
        )

        # 3. Cross-encoder reranking
        reranked_chunks = rerank_chunks(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
            reranker=self.reranker,
            top_k=rerank_top_k,
        )

        # 4. Generate grounded answer via Gemini
        result = generate_answer(
            question=cleaned_question,
            reranked_chunks=reranked_chunks,
            top_n=generate_top_n,
        )

        # 5. Store result in semantic cache
        if use_cache and self.cache is not None and result.get("sources"):
            self.cache.set(cleaned_question, result)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "cached": False,
        }


_default_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Lazily initialize and return the shared RAGPipeline instance."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def answer_question(
    question: str,
    retrieve_top_k: int = 10,
    rerank_top_k: int = 5,
    generate_top_n: int = 3,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Module-level convenience entrypoint used by FastAPI routes and CLI scripts."""
    pipeline = get_rag_pipeline()
    return pipeline.answer_question(
        question=question,
        retrieve_top_k=retrieve_top_k,
        rerank_top_k=rerank_top_k,
        generate_top_n=generate_top_n,
        use_cache=use_cache,
    )