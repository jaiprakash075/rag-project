# Responsibility: FastAPI application entry point
# This file creates the FastAPI application, includes API routers, and manages application startup and shutdown.
# The server can then be started using Uvicorn.


from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router as api_router
from backend.app.services.rag_pipeline import get_rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload embedding, retriever, reranker, and cache resources at startup."""
    get_rag_pipeline()
    yield


app = FastAPI(
    title="MIT IT Department RAG Chatbot API",
    description="Hybrid retrieval, cross-encoder reranking, and Gemini-powered Q&A API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    return {
        "message": "MIT IT Department RAG Chatbot API is running.",
        "docs": "/docs",
    }