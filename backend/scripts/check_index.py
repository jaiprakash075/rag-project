# Inspects chunks in JSON and verifies their presence in ChromaDB


import argparse
import json

import chromadb

from backend.app.config import (
    CHROMA_DB_DIR,
    CHUNKS_FILE,
    COLLECTION_NAME,
)


def check_index(url_filter: str = "teaching", limit: int = 5) -> None:
    """Verify that chunks in website_chunks.json exist in ChromaDB."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    matched_chunks = [
        chunk
        for chunk in chunks
        if url_filter.casefold() in chunk.get("url", "").casefold()
        or url_filter.casefold() in chunk.get("title", "").casefold()
    ]

    print(f"Total chunks in JSON: {len(chunks)}")
    print(f"Chunks matching '{url_filter}': {len(matched_chunks)}")

    if not matched_chunks:
        return

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print(f"Total records in ChromaDB '{COLLECTION_NAME}': {collection.count()}\n")

    sample_ids = [chunk["chunk_id"] for chunk in matched_chunks[:limit]]
    stored = collection.get(
        ids=sample_ids,
        include=["documents", "metadatas"],
    )

    stored_map = {
        chunk_id: (doc, meta)
        for chunk_id, doc, meta in zip(
            stored["ids"],
            stored["documents"] or [],
            stored["metadatas"] or [],
        )
    }

    for chunk in matched_chunks[:limit]:
        cid = chunk["chunk_id"]
        print(f"--- Chunk ID: {cid} ---")
        print(f"Title: {chunk['title']} | URL: {chunk['url']}")
        if cid in stored_map:
            doc, _ = stored_map[cid]
            print("Status: FOUND in ChromaDB")
            print(f"Content preview: {doc[:200]}...\n")
        else:
            print("Status: MISSING from ChromaDB!\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect processed chunks and verify their storage in ChromaDB."
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="teaching",
        help="Substring to filter chunk URLs or titles (default: 'teaching').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of matched chunks to display (default: 5).",
    )
    args = parser.parse_args()

    check_index(url_filter=args.filter, limit=args.limit)


if __name__ == "__main__":
    main()