
from app.retriever import load_retriever, retrieve_chunks
from app.reranker import load_reranker, rerank_chunks
from app.answer_generator import generate_answer
from app.semantic_caching import SemanticCache


def main():
    # Load all five values returned by load_retriever().
    (
        embedding_model,
        collection,
        bm25_index,
        corpus_records,
        records_by_id,
    ) = load_retriever()

    # Load the cross-encoder model.
    reranker = load_reranker()

    # Initialize semantic caching using the existing embedding model.
    semantic_cache = SemanticCache(
        embedding_model=embedding_model
    )

    print("RAG pipeline and semantic cache are ready.")

    while True:
        question = input("\nAsk a question: ").strip()

        if question.casefold() == "exit":
            print("Search stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # Step 1: Check semantic cache before running RAG.
        answer_result = semantic_cache.get(question)

        if answer_result is not None:
            # Cache hit: skip retrieval, reranking and Gemini.
            print("\n[SEMANTIC CACHE HIT]")
            print("Using the previously generated answer.")

        else:
            # Cache miss: run the existing RAG pipeline.
            print("\n[SEMANTIC CACHE MISS]")
            print("Running retrieval and reranking...")

            # Step 2: Retrieve candidate chunks.
            retrieved_chunks = retrieve_chunks(
                question=question,
                embedding_model=embedding_model,
                collection=collection,
                bm25_index=bm25_index,
                corpus_records=corpus_records,
                records_by_id=records_by_id,
                top_k=10,
            )

            if not retrieved_chunks:
                print("No chunks retrieved.")
                continue

            print(
                f"\nRetrieved candidates: "
                f"{len(retrieved_chunks)}"
            )

            # Step 3: Rerank the retrieved chunks.
            reranked_chunks = rerank_chunks(
                question=question,
                retrieved_chunks=retrieved_chunks,
                reranker=reranker,
                top_k=5,
            )

            # Step 4: Display the final ranking.
            print("\nRERANKED RESULTS")

            for rank, chunk in enumerate(
                reranked_chunks,
                start=1,
            ):
                print(f"\n--- Result {rank} ---")
                print(
                    f"Rerank score: "
                    f"{chunk['rerank_score']:.4f}"
                )
                print(f"Chunk ID: {chunk['chunk_id']}")
                print(f"URL: {chunk['url']}")
                print(f"Content:\n{chunk['content']}")

            # Step 5: Generate the answer using Gemini.
            print("\nGenerating answer with Gemini...")

            answer_result = generate_answer(
                question=question,
                reranked_chunks=reranked_chunks,
                top_n=3,
            )

            # Step 6: Cache the successful answer and sources.
            semantic_cache.set(
                question=question,
                answer_result=answer_result,
            )

        # Display the answer for both cache hits and cache misses.
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(answer_result["answer"])

        print("\nSOURCES")

        for source in answer_result.get("sources", []):
            source_number = source.get("source_number", "")
            title = source.get("title", "Untitled source")
            url = source.get("url", "")

            print(f"[{source_number}] {title}")
            print(f"URL: {url}\n")


if __name__ == "__main__":
    main()
