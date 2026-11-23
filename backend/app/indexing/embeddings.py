# Responsibility: Embedding generation
# Your current implementation:
# - Loads website_chunks.json.
# - Loads sentence-transformers/all-MiniLM-L6-v2.
# - Converts each chunk into a 384-dimensional vector.
# - Processes chunks in batches.
# - Normalizes the embeddings.
# - Preserves the chunk metadata.
# - Saves the output to website_embeddings.json.

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from backend.app.config import (
    CHUNKS_FILE,
    EMBEDDINGS_FILE,
    EMBEDDING_MODEL_NAME,
)

BATCH_SIZE = 32


def load_chunks(input_file: Path = CHUNKS_FILE) -> list[dict]:
    """Load the website chunks from the JSON file."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_file}\n"
            "Run the chunker first."
        )

    with input_file.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Expected website_chunks.json to contain a list.")

    return chunks


def generate_embeddings(
    input_file: Path = CHUNKS_FILE,
    output_file: Path = EMBEDDINGS_FILE,
) -> list[dict]:
    """Generate a normalized embedding vector for each website chunk."""
    website_chunks = load_chunks(input_file)

    if not website_chunks:
        print("No chunks found. Nothing to embed.")
        return []

    for chunk in website_chunks:
        if not chunk.get("chunk_id") or not chunk.get("content", "").strip():
            raise ValueError(
                "Every chunk must have a chunk_id and non-empty content."
            )

    print(f"Loaded {len(website_chunks)} chunks.")
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Maximum input tokens: {embedding_model.max_seq_length}")

    chunk_texts = [chunk["content"] for chunk in website_chunks]

    embedding_vectors = embedding_model.encode(
        chunk_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedded_chunks = []

    for chunk, vector in zip(website_chunks, embedding_vectors):
        embedded_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "title": chunk["title"],
            "chunk_number": chunk["chunk_number"],
            "content": chunk["content"],
            "embedding": vector.tolist(),
        })

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(embedded_chunks, file, ensure_ascii=False)

    print("\nEmbedding generation completed.")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vector dimensions: {embedding_vectors.shape[1]}")
    print(f"Saved to: {output_file}")

    return embedded_chunks


if __name__ == "__main__":
    generate_embeddings()