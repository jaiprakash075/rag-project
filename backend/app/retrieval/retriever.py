# Responsibility: Hybrid retrieval
# This is currently your main retrieval implementation.
# It combines:
# - Semantic search using sentence embeddings and cosine similarity.
# - BM25 keyword search.
# - Score normalization.
# - Weighted score fusion.
# - An exact-term matching bonus.
# - Sorting and returning the top-ranked chunks.

import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.app.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

TOKEN_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "of", "in", "on", "at", "to", "for", "from",
    "and", "or", "who", "what", "which", "where",
    "when", "how", "do", "does", "did", "can",
    "could", "please", "me", "tell", "about", "give",
    "show", "find", "having", "has", "have",
}

RRF_K = 60


def tokenize_text(text: str) -> list[str]:
    """Convert text into lowercase words and numbers for keyword search."""
    return re.findall(r"\b\w+\b", text.casefold())


def normalize_scores(scores_by_id: dict[str, float]) -> dict[str, float]:
    """Scale scores to the range 0 to 1 using min-max normalization."""
    if not scores_by_id:
        return {}

    lowest_score = min(scores_by_id.values())
    highest_score = max(scores_by_id.values())

    if highest_score == lowest_score:
        return {chunk_id: 0.0 for chunk_id in scores_by_id}

    return {
        chunk_id: (score - lowest_score) / (highest_score - lowest_score)
        for chunk_id, score in scores_by_id.items()
    }


def get_meaningful_query_terms(question: str) -> list[str]:
    """Remove common words so matching focuses on useful query terms."""
    query_tokens = tokenize_text(question)

    return [
        token
        for token in query_tokens
        if token not in TOKEN_STOPWORDS
    ]


def load_retriever(embedding_model: SentenceTransformer | None = None):
    """
    Load the embedding model, ChromaDB collection, and BM25 index.

    The BM25 index is built once when the retriever starts,
    rather than rebuilding it for every question.
    """
    if embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    stored_chunks = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )

    chunk_ids = stored_chunks["ids"]
    documents = stored_chunks["documents"] or []
    metadatas = stored_chunks["metadatas"] or []
    stored_embeddings = stored_chunks["embeddings"]

    if (
        stored_embeddings is None
        or len(stored_embeddings) != len(chunk_ids)
    ):
        raise ValueError(
            "Some ChromaDB records are missing embeddings. "
            "Check your vector_store.py and stored collection."
        )

    if not chunk_ids:
        raise ValueError(
            f"The ChromaDB collection '{COLLECTION_NAME}' is empty. "
            "Generate embeddings and populate the collection first."
        )

    corpus_records = []
    tokenized_documents = []

    for chunk_id, content, metadata, embedding in zip(
        chunk_ids,
        documents,
        metadatas,
        stored_embeddings,
    ):
        metadata = metadata or {}
        content = content or ""

        title = metadata.get("title", "")
        url = metadata.get("url", "")

        searchable_text = f"{title} {url} {content}"

        corpus_records.append({
            "chunk_id": chunk_id,
            "content": content,
            "title": title,
            "url": url,
            "embedding": np.asarray(
                embedding,
                dtype=np.float32,
            ).tolist(),
        })

        tokenized_documents.append(
            tokenize_text(searchable_text)
        )

    print("Building BM25 keyword index...")
    bm25_index = BM25Okapi(tokenized_documents)

    records_by_id = {
        record["chunk_id"]: record
        for record in corpus_records
    }

    print(f"Loaded {len(corpus_records)} chunks.")
    print("Hybrid retriever is ready.\n")

    return (
        embedding_model,
        collection,
        bm25_index,
        corpus_records,
        records_by_id,
    )


def retrieve_chunks(
    question: str,
    embedding_model,
    collection,
    bm25_index,
    corpus_records: list[dict],
    records_by_id: dict,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve chunks using normalized hybrid score fusion."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = len(corpus_records)

    if total_chunks == 0:
        return []

    # ---------------------------------------------------------
    # 1. SEMANTIC SEARCH: score every chunk
    # ---------------------------------------------------------
    question_vector = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_matrix = np.asarray(
        [chunk["embedding"] for chunk in corpus_records],
        dtype=np.float32,
    )

    question_norm = np.linalg.norm(question_vector)
    chunk_norms = np.linalg.norm(embedding_matrix, axis=1)

    denominators = np.maximum(
        question_norm * chunk_norms,
        1e-12,
    )

    similarity_values = (
        embedding_matrix @ question_vector
    ) / denominators

    similarity_values = np.clip(
        similarity_values,
        -1.0,
        1.0,
    )

    semantic_similarity_by_id = {
        chunk["chunk_id"]: float(similarity)
        for chunk, similarity in zip(
            corpus_records,
            similarity_values,
        )
    }

    semantic_normalized = normalize_scores(
        semantic_similarity_by_id
    )

    semantic_ranks = {
        chunk_id: rank
        for rank, (chunk_id, _) in enumerate(
            sorted(
                semantic_similarity_by_id.items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            start=1,
        )
    }

    # ---------------------------------------------------------
    # 2. KEYWORD SEARCH: score every chunk using BM25
    # ---------------------------------------------------------
    query_tokens = tokenize_text(question)

    if query_tokens:
        keyword_scores = bm25_index.get_scores(query_tokens)
    else:
        keyword_scores = [0.0] * total_chunks

    keyword_score_by_id = {
        record["chunk_id"]: float(score)
        for record, score in zip(corpus_records, keyword_scores)
    }

    keyword_normalized = normalize_scores(keyword_score_by_id)

    keyword_ranks = {
        chunk_id: rank
        for rank, (chunk_id, score) in enumerate(
            sorted(
                keyword_score_by_id.items(),
                key=lambda item: item[1],
                reverse=True,
            ),
            start=1,
        )
        if score > 0
    }

    # ---------------------------------------------------------
    # 3. EXACT-TERM MATCHING
    # ---------------------------------------------------------
    meaningful_terms = get_meaningful_query_terms(question)

    # ---------------------------------------------------------
    # 4. COMBINE THE SCORES
    # ---------------------------------------------------------
    semantic_weight = 0.40
    keyword_weight = 0.60
    exact_term_bonus = 0.30

    hybrid_results = []

    for chunk in corpus_records:
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        title = chunk["title"]

        content_tokens = set(
            tokenize_text(f"{title} {content}")
        )

        all_query_terms_found = (
            bool(meaningful_terms)
            and all(
                term in content_tokens
                for term in meaningful_terms
            )
        )

        semantic_score = semantic_similarity_by_id[chunk_id]
        semantic_score_normalized = semantic_normalized[chunk_id]

        bm25_score = keyword_score_by_id[chunk_id]
        keyword_score_normalized = keyword_normalized[chunk_id]

        exact_term_match = all_query_terms_found

        hybrid_score = (
            semantic_weight * semantic_score_normalized
            + keyword_weight * keyword_score_normalized
        )

        if exact_term_match:
            hybrid_score += exact_term_bonus

        has_keyword_match = bm25_score > 0
        retrieval_method = (
            "semantic + keyword" if has_keyword_match else "semantic"
        )

        cosine_distance = 1.0 - semantic_score

        hybrid_results.append({
            "chunk_id": chunk_id,
            "content": content,
            "title": title,
            "url": chunk["url"],
            "distance": cosine_distance,
            "similarity": semantic_score,
            "semantic_normalized_score": semantic_score_normalized,
            "bm25_score": bm25_score,
            "keyword_normalized_score": keyword_score_normalized,
            "semantic_rank": semantic_ranks.get(chunk_id),
            "keyword_rank": keyword_ranks.get(chunk_id),
            "exact_term_match": exact_term_match,
            "hybrid_score": hybrid_score,
            "retrieval_method": retrieval_method,
        })

    hybrid_results.sort(
        key=lambda result: result["hybrid_score"],
        reverse=True,
    )

    return hybrid_results[:top_k]


def main():
    """Run an interactive hybrid-search test."""
    (
        embedding_model,
        collection,
        bm25_index,
        corpus_records,
        records_by_id,
    ) = load_retriever()

    print(f"Connected to collection: {COLLECTION_NAME}")
    print(f"Available chunks: {collection.count()}")
    print("Search method: Semantic + BM25 Hybrid")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.casefold() == "exit":
            print("Search stopped.")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        results = retrieve_chunks(
            question=question,
            embedding_model=embedding_model,
            collection=collection,
            bm25_index=bm25_index,
            corpus_records=corpus_records,
            records_by_id=records_by_id,
            top_k=5,
        )

        if not results:
            print("No matching chunks were retrieved.\n")
            continue

        print("\nRetrieved chunks:")

        for rank, result in enumerate(results, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Raw cosine similarity: {result['similarity']:.4f}")
            print(
                "Normalized semantic score: "
                f"{result['semantic_normalized_score']:.4f}"
            )
            print(f"BM25 score: {result['bm25_score']:.4f}")
            print(
                "Normalized keyword score: "
                f"{result['keyword_normalized_score']:.4f}"
            )
            print(f"Exact-term match: {result['exact_term_match']}")
            print(f"Semantic rank: {result['semantic_rank']}")
            print(f"Keyword rank: {result['keyword_rank']}")
            print(f"Hybrid score: {result['hybrid_score']:.4f}")
            print(f"Matched by: {result['retrieval_method']}")
            print(f"Content:\n{result['content']}")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()