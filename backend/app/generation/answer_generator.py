# Responsibility: Generate the final answer using Gemini
# This file receives the top-ranked chunks from the reranker.
# It:
# 1. Selects the top three chunks.
# 2. Constructs the context using those chunks.
# 3. Creates system instructions for Gemini.
# 4. Sends the question and context to Gemini.
# 5. Receives the generated answer.
# 6. Returns the answer and source metadata.

from google import genai

from backend.app.config import GEMINI_API_KEY, GEMINI_MODEL


def generate_answer(
    question: str,
    reranked_chunks: list[dict],
    top_n: int = 3,
) -> dict:
    """Generate an answer using Gemini and reranked RAG chunks."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to backend/.env."
        )

    if not reranked_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the "
                "IT department knowledge base to answer your question."
            ),
            "sources": [],
        }

    selected_chunks = reranked_chunks[:top_n]

    context_parts = []
    sources = []

    for source_number, chunk in enumerate(selected_chunks, start=1):
        title = chunk.get("title", "Untitled page")
        url = chunk.get("url", "")
        content = chunk.get("content", "")

        context_parts.append(
            f"[Source {source_number}]\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{content}"
        )

        sources.append({
            "source_number": source_number,
            "title": title,
            "url": url,
            "chunk_id": chunk.get("chunk_id"),
        })

    context = "\n\n".join(context_parts)

    instructions = """
You are a helpful RAG assistant for the IT Department website
at Madras Institute of Technology, Anna University.

Answer the user's question using only the supplied context.

Rules:
1. Do not invent facts that are not present in the context.
2. If the context does not contain the answer, clearly say so.
3. Give a clear, direct answer.
4. Use numbered source citations such as [Source 1] in your answer.
5. Cite only sources that support the relevant statements.
6. If sources disagree or information is incomplete, mention that.
"""

    prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using the context above.
Include relevant source citations in your answer.
"""

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "system_instruction": instructions,
            "temperature": 0.2,
            "max_output_tokens": 500,
        },
    )

    answer_text = response.text

    if not answer_text:
        raise RuntimeError(
            "Gemini returned an empty response. "
            "Check the model response and API configuration."
        )

    return {
        "answer": answer_text.strip(),
        "sources": sources,
    }