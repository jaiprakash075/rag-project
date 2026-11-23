# Responsibility: Define API endpoints
# For example:
# POST /ask
# It receives the question from the frontend, validates it, calls answer_question(), and returns the result.
# It should not contain the retrieval or Gemini implementation itself.


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.rag_pipeline import answer_question

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question about the MIT IT Department.",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to check and store answers in the semantic cache.",
    )


class SourceItem(BaseModel):
    source_number: int
    title: str
    url: str
    chunk_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    cached: bool


@router.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Verify that the API service is running."""
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(payload: AskRequest) -> AskResponse:
    """Accept a user question, run the RAG pipeline, and return the answer."""
    cleaned_question = payload.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty or whitespace only.",
        )

    try:
        result = answer_question(
            question=cleaned_question,
            use_cache=payload.use_cache,
        )
        return AskResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {error}",
        ) from error