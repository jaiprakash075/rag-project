
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_DIR / "backend"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

load_dotenv(BACKEND_DIR / ".env")

CACHE_COLLECTION_NAME = "semantic_answer_cache"

SIMILARITY_THRESHOLD = float(
    os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90")
)

CACHE_TTL_SECONDS = int(
    os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400")
)

logger = logging.getLogger(__name__)


class SemanticCache:
    """Cache RAG answers using semantic similarity."""

    def __init__(
        self,
        embedding_model,
        threshold=SIMILARITY_THRESHOLD,
        ttl_seconds=CACHE_TTL_SECONDS,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        self.embedding_model = embedding_model
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds

        # Reuse the project's persistent ChromaDB folder.
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        # Keep cached answers separate from knowledge chunks.
        self.collection = self.client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def normalize_question(question: str) -> str:
        """Normalize whitespace and letter case."""
        return " ".join(question.strip().casefold().split())

    def create_embedding(self, question: str) -> list[float]:
        """Convert a question into an embedding vector."""
        question_vector = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            question_vector,
            dtype=np.float32,
        ).tolist()

    def get(self, question: str) -> dict | None:
        """
        Return a cached answer when a sufficiently similar,
        non-expired entry exists. Otherwise return None.
        """
        if not question.strip():
            return None

        try:
            existing_count = self.collection.count()

            if existing_count == 0:
                logger.info("Semantic cache is empty.")
                return None

            question_vector = self.create_embedding(question)

            # Inspect several nearest matches, since the closest
            # entry might have expired.
            search_results = self.collection.query(
                query_embeddings=[question_vector],
                n_results=min(10, existing_count),
                include=["metadatas", "distances"],
            )

            matched_metadata = search_results["metadatas"][0]
            matched_distances = search_results["distances"][0]

            current_time = time.time()

            for metadata, distance in zip(
                matched_metadata,
                matched_distances,
            ):
                if not metadata:
                    continue

                # Chroma returns cosine DISTANCE:
                # distance = 1 - cosine similarity.
                similarity = 1.0 - float(distance)

                if similarity < self.threshold:
                    continue

                created_at = float(
                    metadata.get("created_at", 0)
                )

                # A TTL of zero or less disables expiry.
                if self.ttl_seconds > 0:
                    age_seconds = current_time - created_at

                    if age_seconds > self.ttl_seconds:
                        continue

                cached_answer = metadata.get("answer")

                if not cached_answer:
                    continue

                try:
                    cached_sources = json.loads(
                        metadata.get("sources_json", "[]")
                    )
                except (TypeError, json.JSONDecodeError):
                    cached_sources = []

                logger.info(
                    "Semantic cache HIT: similarity=%.4f",
                    similarity,
                )

                # Preserve the answer generator's return structure.
                return {
                    "answer": cached_answer,
                    "sources": cached_sources,
                }

            logger.info("Semantic cache MISS.")
            return None

        except Exception:
            # A cache failure should not prevent normal RAG answering.
            logger.exception(
                "Semantic cache lookup failed; continuing without cache."
            )
            return None

    def set(self, question: str, answer_result: dict) -> None:
        """Save a successfully generated answer and its sources."""
        if not question.strip():
            return

        answer_text = answer_result.get("answer")

        if not answer_text:
            return

        try:
            normalized_question = self.normalize_question(question)

            # Identical normalized questions update the same entry.
            cache_id = hashlib.sha256(
                normalized_question.encode("utf-8")
            ).hexdigest()

            question_vector = self.create_embedding(question)

            metadata = {
                "answer": answer_text,
                "sources_json": json.dumps(
                    answer_result.get("sources", []),
                    ensure_ascii=False,
                ),
                "created_at": time.time(),
            }

            self.collection.upsert(
                ids=[cache_id],
                documents=[normalized_question],
                embeddings=[question_vector],
                metadatas=[metadata],
            )

            logger.info("Saved answer in semantic cache.")

        except Exception:
            # Cache storage is optional; the answer is still usable.
            logger.exception("Failed to save answer in semantic cache.")
