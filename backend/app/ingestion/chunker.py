# Responsibility: Text preprocessing and chunking
# This file receives extracted website text and divides it into smaller chunks.
# Its responsibilities:
# - Remove duplicate paragraphs.
# - Clean unwanted whitespace.
# - Apply recursive text splitting.
# - Maintain chunk overlap.
# - Generate unique chunk IDs.
# - Attach title and URL metadata.

import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.config import (
    CHUNKS_FILE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PAGES_FILE,
)


def preprocess_content(content: str) -> str:
    """Clean extracted webpage text while preserving paragraph boundaries."""
    cleaned_paragraphs = []
    seen_paragraphs = set()

    for line in content.splitlines():
        paragraph = re.sub(r"\s+", " ", line).strip()

        if not paragraph:
            continue

        if paragraph in seen_paragraphs:
            continue

        seen_paragraphs.add(paragraph)
        cleaned_paragraphs.append(paragraph)

    return "\n\n".join(cleaned_paragraphs)


def split_into_chunks(
    content: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using RecursiveCharacterTextSplitter."""
    if not content.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        length_function=len,
    )

    raw_chunks = splitter.split_text(content)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]


def build_chunks(
    input_file: Path = PAGES_FILE,
    output_file: Path = CHUNKS_FILE,
) -> list[dict]:
    """Load crawled pages, preprocess them, and create indexed chunks."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Run the website crawler first."
        )

    with input_file.open("r", encoding="utf-8") as file:
        pages = json.load(file)

    all_chunks = []
    seen_pages = set()

    for page in pages:
        page_url = page.get("url", "")
        page_title = page.get("title", "Untitled page")
        page_content = page.get("content", "")

        if not page_url or page_url in seen_pages:
            continue

        seen_pages.add(page_url)

        cleaned_content = preprocess_content(page_content)

        if not cleaned_content:
            print(f"Skipping empty page: {page_url}")
            continue

        page_chunks = split_into_chunks(cleaned_content)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            all_chunks.append({
                "chunk_id": f"page_{len(seen_pages)}_chunk_{chunk_number}",
                "url": page_url,
                "title": page_title,
                "chunk_number": chunk_number,
                "content": chunk_text,
            })

        print(f"Processed: {page_title} -> {len(page_chunks)} chunks")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nTotal pages processed: {len(seen_pages)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Output saved to: {output_file}")

    return all_chunks


if __name__ == "__main__":
    build_chunks()