# - Load the generated embeddings.
# - Validate chunk IDs, text and embedding dimensions.
# - Create or access the ChromaDB collection.
# - Insert embeddings, documents and metadata in batches.
# - Verify the number of stored records.

import json
from pathlib import Path

import chromadb

from backend.app.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDINGS_FILE,
)

BATCH_SIZE = 100


def load_embeddings(input_file: Path = EMBEDDINGS_FILE) -> list[dict]:
    """Load the generated embeddings from the JSON file."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {input_file}\n"
            "Run the embedding generator first."
        )

    with input_file.open("r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)

    if not isinstance(embedded_chunks, list):
        raise ValueError("Expected a list of embedded chunks.")

    return embedded_chunks


def validate_embeddings(embedded_chunks: list[dict]) -> None:
    """Check that every chunk has valid data before storing it."""
    seen_ids = set()

    for chunk in embedded_chunks:
        chunk_id = chunk.get("chunk_id")
        content = chunk.get("content")
        embedding = chunk.get("embedding")

        if not chunk_id or not isinstance(chunk_id, str):
            raise ValueError("Every chunk must have a valid chunk_id.")

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk ID found: {chunk_id}")

        seen_ids.add(chunk_id)

        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty content in chunk: {chunk_id}")

        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"Missing embedding in chunk: {chunk_id}")

        if not all(isinstance(value, (int, float)) for value in embedding):
            raise ValueError(f"Invalid embedding values in chunk: {chunk_id}")

    if embedded_chunks:
        expected_dimensions = len(embedded_chunks[0]["embedding"])

        for chunk in embedded_chunks:
            if len(chunk["embedding"]) != expected_dimensions:
                raise ValueError(
                    f"Inconsistent embedding dimensions: {chunk['chunk_id']}"
                )


def delete_collection(
    chroma_client: chromadb.PersistentClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Explicitly delete an existing ChromaDB collection if present."""
    existing_names = [col.name for col in chroma_client.list_collections()]
    if collection_name in existing_names:
        chroma_client.delete_collection(name=collection_name)
        print(f"Deleted existing ChromaDB collection: '{collection_name}'.")
    else:
        print(f"Collection '{collection_name}' does not exist; nothing to delete.")


def store_embeddings(
    input_file: Path = EMBEDDINGS_FILE,
    rebuild: bool = False,
) -> None:
    """Store text chunks, embeddings, and metadata in ChromaDB."""
    embedded_chunks = load_embeddings(input_file)

    if not embedded_chunks:
        print("No embeddings found. Nothing to store.")
        return

    validate_embeddings(embedded_chunks)
    print(f"Loaded {len(embedded_chunks)} embedded chunks.")

    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

    if rebuild:
        delete_collection(chroma_client, COLLECTION_NAME)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for batch_start in range(0, len(embedded_chunks), BATCH_SIZE):
        batch = embedded_chunks[batch_start : batch_start + BATCH_SIZE]

        chunk_ids = []
        chunk_texts = []
        chunk_vectors = []
        chunk_metadata = []

        for chunk in batch:
            chunk_ids.append(chunk["chunk_id"])
            chunk_texts.append(chunk["content"])
            chunk_vectors.append(chunk["embedding"])
            chunk_metadata.append({
                "url": chunk["url"],
                "title": chunk["title"],
                "chunk_number": int(chunk["chunk_number"]),
            })

        collection.upsert(
            ids=chunk_ids,
            documents=chunk_texts,
            embeddings=chunk_vectors,
            metadatas=chunk_metadata,
        )

        print(f"Stored batch: {batch_start + 1}–{batch_start + len(batch)}")

    stored_count = collection.count()

    print("\nChromaDB indexing completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Records in collection: {stored_count}")
    print(f"Database directory: {CHROMA_DB_DIR}")

    sample_record = collection.get(
        ids=[embedded_chunks[0]["chunk_id"]],
        include=["documents", "metadatas"],
    )

    if sample_record["ids"]:
        print("\nSample stored record:")
        print(f"ID: {sample_record['ids'][0]}")
        print(f"Title: {sample_record['metadatas'][0]['title']}")
        print(f"URL: {sample_record['metadatas'][0]['url']}")
        print(f"Text: {sample_record['documents'][0][:200]}")


if __name__ == "__main__":
    store_embeddings(rebuild=False)