# Runs the website crawler and saves extracted pages
# scripts/build_index.py
# Runs the indexing pipeline: chunking → embeddings → ChromaDB storage
# scripts/chat_cli.py
# Allows you to interact with the chatbot through PowerShell
# scripts/check_index.py
# Inspects chunks in JSON and verifies their presence in ChromaDB
# Let's look at the existing files that should move.

import argparse

from backend.app.ingestion.crawler import crawl_website, save_pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl the MIT IT Department website and save raw pages."
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=30,
        help="Maximum number of pages to crawl (default: 30).",
    )
    args = parser.parse_args()

    extracted_pages = crawl_website(max_pages=args.max_pages)
    save_pages(extracted_pages)


if __name__ == "__main__":
    main()