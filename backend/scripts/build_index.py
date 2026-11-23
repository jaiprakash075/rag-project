# Runs the indexing pipeline: chunking → embeddings → ChromaDB storage
# scripts/chat_cli.py
# Allows you to interact with the chatbot through PowerShell
# scripts/check_index.py
# Inspects chunks in JSON and verifies their presence in ChromaDB

import argparse

from backend.app.indexing.embeddings import generate_embeddings
from backend.app.indexing.vector_store import store_embeddings
from backend.app.ingestion.chunker import build_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the RAG index: chunk pages, generate embeddings, and store in ChromaDB."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing ChromaDB collection before indexing.",
    )
    args = parser.parse_args()

    print("=== Step 1/3: Chunking Website Pages ===")
    build_chunks()

    print("\n=== Step 2/3: Generating Embeddings ===")
    generate_embeddings()

    print("\n=== Step 3/3: Storing Vectors in ChromaDB ===")
    store_embeddings(rebuild=args.rebuild)


if __name__ == "__main__":
    main()