# Responsibility: Website crawling
# Your current file already implements:
# - Visiting the IT department website.
# - Discovering links and navigating to other pages.
# - Restricting the crawl to the specified website section.
# - Checking robots.txt.
# - Avoiding repeated URLs.
# - Saving extracted pages into JSON.

import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from backend.app.config import (
    ALLOWED_HOST,
    BASE_URL,
    CRAWLER_REQUEST_DELAY,
    CRAWLER_USER_AGENT,
    PAGES_FILE,
)
from backend.app.ingestion.content_extraction import extract_page_content

HEADERS = {
    "User-Agent": CRAWLER_USER_AGENT,
}


def normalize_url(url: str) -> str | None:
    """Keep URLs within the IT Department website section."""
    url = urldefrag(url)[0]
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https"):
        return None

    if parsed_url.hostname != ALLOWED_HOST:
        return None

    page_path = parsed_url.path.rstrip("/")

    if page_path != "/it" and not page_path.startswith("/it/"):
        return None

    return parsed_url._replace(
        scheme="https",
        query="",
        fragment="",
    ).geturl().rstrip("/")


def load_robots_rules(session: requests.Session) -> RobotFileParser:
    """Read the website's robots.txt before crawling."""
    robots_url = f"https://{ALLOWED_HOST}/robots.txt"
    response = session.get(robots_url, timeout=15)

    if response.status_code == 404:
        print("robots.txt not found; check the site's crawling policy.")
        rules = RobotFileParser()
        rules.set_url(robots_url)
        rules.parse([])
        return rules

    response.raise_for_status()

    rules = RobotFileParser()
    rules.set_url(robots_url)
    rules.parse(response.text.splitlines())
    return rules


def crawl_website(max_pages: int = 30) -> list[dict]:
    """Crawl permitted pages within the IT Department website."""
    session = requests.Session()
    session.headers.update(HEADERS)

    robots_rules = load_robots_rules(session)

    pages = []
    visited_urls = set()
    pending_urls = deque([BASE_URL])

    while pending_urls and len(visited_urls) < max_pages:
        current_url = normalize_url(pending_urls.popleft())

        if not current_url or current_url in visited_urls:
            continue

        if not robots_rules.can_fetch(HEADERS["User-Agent"], current_url):
            print(f"Blocked by robots.txt: {current_url}")
            visited_urls.add(current_url)
            continue

        visited_urls.add(current_url)
        print(f"[{len(visited_urls)}/{max_pages}] {current_url}")

        try:
            response = session.get(current_url, timeout=20)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type.lower():
                continue

            final_url = normalize_url(response.url)
            if not final_url:
                print("Skipping redirect outside the IT portal.")
                continue

            page = extract_page_content(
                response.text,
                final_url,
            )

            if not page["content"]:
                print(f"No text extracted: {final_url}")
                continue

            pages.append(page)

            for link in page["links"]:
                next_url = normalize_url(
                    urljoin(final_url, link["href"])
                )

                if next_url and next_url not in visited_urls:
                    pending_urls.append(next_url)

        except requests.RequestException as error:
            print(f"Request failed for {current_url}: {error}")

        time.sleep(CRAWLER_REQUEST_DELAY)

    session.close()
    return pages


def save_pages(pages: list[dict], output_file: Path = PAGES_FILE) -> None:
    """Save extracted website pages as UTF-8 JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            pages,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nPages extracted: {len(pages)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    extracted_pages = crawl_website(max_pages=30)
    save_pages(extracted_pages)