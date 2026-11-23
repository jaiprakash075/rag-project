
from bs4 import BeautifulSoup


def extract_page_content(html: str, page_url: str) -> dict:
    """Extract page content and all links before cleaning HTML."""

    soup = BeautifulSoup(html, "html.parser")

    # 1. Extract the title.
    page_title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled page"
    )

    # 2. IMPORTANT: Extract links BEFORE removing navigation.
    page_links = []

    for anchor in soup.find_all("a", href=True):
        link_text = anchor.get_text(" ", strip=True)
        link_url = anchor["href"].strip()

        if link_url:
            page_links.append({
                "text": link_text,
                "href": link_url,
            })

    # 3. Remove unwanted elements only after collecting links.
    for element in soup.select(
        "script, style, noscript, iframe, "
        "nav, footer, header, form"
    ):
        element.decompose()

    # 4. Locate the main page content.
    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find("body")
    )

    # 5. Extract readable text.
    content = ""

    if main_content:
        content = main_content.get_text(
            separator="\n",
            strip=True,
        )

    # 6. Return content AND links to the crawler.
    return {
        "url": page_url,
        "title": page_title,
        "content": content,
        "links": page_links,
    }
