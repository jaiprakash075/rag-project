# 1. Loads the embedding model and retriever.
# 2. Loads the cross-encoder.
# 3. Initializes the semantic cache.
# 4. Accepts questions through the terminal.
# 5. Invokes the RAG pipeline.
# 6. Displays the answer and citations.


from backend.app.services.rag_pipeline import answer_question, get_rag_pipeline


def main() -> None:
    """Interactive PowerShell terminal interface for the RAG chatbot."""
    print("Initializing IT Department RAG Pipeline...")
    get_rag_pipeline()

    print("\nRAG Chatbot ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if question.casefold() in {"exit", "quit"}:
            print("Chat ended.")
            break

        if not question:
            print("Please enter a valid question.\n")
            continue

        try:
            result = answer_question(question=question)
        except Exception as error:
            print(f"\nError generating answer: {error}\n")
            continue

        cache_tag = "[CACHED]" if result.get("cached") else "[LIVE]"
        print(f"\nAnswer {cache_tag}:\n{result['answer']}")

        sources = result.get("sources", [])
        if sources:
            print("\nSources:")
            for source in sources:
                print(
                    f"  [{source['source_number']}] "
                    f"{source['title']} - {source['url']}"
                )

        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()