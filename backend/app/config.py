# Responsibility: Central configuration
# Instead of scattering constants across several Python files, we should centralize:
# - Project and data paths.
# - ChromaDB storage path and collection name.
# - Embedding model name.
# - Cross-encoder model name.
# - Gemini model name.
# - Redis URL.
# - Semantic-cache threshold and TTL.
# - Chunk size and overlap.

import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(BACKEND_DIR / ".env")

# Data & Storage Paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
STORAGE_DIR = PROJECT_ROOT / "storage"

PAGES_FILE = RAW_DATA_DIR / "website_pages.json"
CHUNKS_FILE = PROCESSED_DATA_DIR / "website_chunks.json"
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "website_embeddings.json"
CHROMA_DB_DIR = STORAGE_DIR / "chroma_db"

# ChromaDB Settings
COLLECTION_NAME = "it_department_knowledge"

# Crawler Settings
BASE_URL = "https://departments.auegov.ac.in/it"
ALLOWED_HOST = "departments.auegov.ac.in"
CRAWLER_USER_AGENT = "ITDepartmentRAGBot/1.0 (educational project)"
CRAWLER_REQUEST_DELAY = 1.0

# Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 70

# Model Settings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Gemini Generation Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-pro-latest")

# Redis & Semantic Cache Settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))