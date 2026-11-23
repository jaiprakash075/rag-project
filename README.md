# MIT IT Department RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built to answer questions about the Department of Information Technology, Madras Institute of Technology (MIT), Anna University.

## Features

* **Web Crawling & Extraction:** Crawls pages within the MIT IT Department portal while respecting `robots.txt` and cleans HTML content.

* **Recursive Text Chunking:** Deduplicates paragraphs and splits text into overlapping chunks using `RecursiveCharacterTextSplitter` (500 characters, 70 overlap).

* **Hybrid Retrieval:** Combines ChromaDB cosine similarity search with BM25 keyword search and exact-term matching.

* **Cross-Encoder Reranking:** Reranks retrieved chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

* **Grounded Answer Generation:** Uses Google Gemini (or a local Ollama model) to generate cited answers from the top chunks.

* **Semantic Caching:** Caches previously answered questions to avoid redundant LLM calls.

* **CLI & FastAPI Interfaces:** Supports both terminal chat and REST API endpoints (`POST /api/ask`).

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py             # FastAPI endpoints (/ask, /health)
│   │   ├── caching/
│   │   │   └── semantic_caching.py   # Semantic answer cache
│   │   ├── generation/
│   │   │   └── answer_generator.py   # Prompt construction & LLM generation
│   │   ├── indexing/
│   │   │   ├── embeddings.py         # Sentence-transformer embedding generation
│   │   │   └── vector_store.py       # ChromaDB storage and validation
│   │   ├── ingestion/
│   │   │   ├── crawler.py            # Website crawler
│   │   │   ├── content_extraction.py # HTML parsing and link extraction
│   │   │   └── chunker.py            # Text cleaning and recursive chunking
│   │   ├── retrieval/
│   │   │   ├── retriever.py          # Hybrid search (Semantic + BM25)
│   │   │   └── reranker.py           # Cross-encoder reranking
│   │   ├── services/
│   │   │   └── rag_pipeline.py       # End-to-end RAG pipeline orchestration
│   │   ├── config.py                 # Central paths, models, and settings
│   │   └── main.py                   # FastAPI app entrypoint
│   ├── scripts/
│   │   ├── crawl_website.py          # Run the website crawler
│   │   ├── build_index.py            # Run chunking, embedding, and ChromaDB indexing
│   │   ├── check_index.py            # Inspect chunks in ChromaDB
│   │   └── chat_cli.py               # Terminal chat runner
│   └── requirements.txt
├── data/
│   ├── raw/                          # Crawled website JSON
│   └── processed/                    # Chunks and embeddings JSON
└── storage/
    └── chroma_db/                    # Persistent ChromaDB vector store

```

## Setup

1. **Clone the repository and create a virtual environment:**

   ```
   git clone <your-repo-url>
   cd rag-chatbot/backend
   python -m venv .venv
   
   ```

2. **Activate the virtual environment:**

   * Windows (PowerShell):

     ```
     .\.venv\Scripts\Activate.ps1
     
     ```

   * Linux/macOS:

     ```
     source .venv/bin/activate
     
     ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   pip install "fastapi[standard]" langchain-text-splitters
   
   ```

4. **Configure environment variables:**
   Create `backend/.env` and add your Gemini API key:

   ```
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   SEMANTIC_CACHE_THRESHOLD=0.90
   SEMANTIC_CACHE_TTL_SECONDS=86400
   
   ```

## Usage

Run commands from the project root (`rag-chatbot/`):

* **Crawl the website:**

  ```
  python -m backend.scripts.crawl_website --max-pages 30
  
  ```

* **Build or rebuild the vector index:**

  ```
  python -m backend.scripts.build_index --rebuild
  
  ```

* **Verify indexed chunks:**

  ```
  python -m backend.scripts.check_index --filter teaching
  
  ```

* **Chat via terminal:**

  ```
  python -m backend.scripts.chat_cli
  
  ```

* **Start the FastAPI server:**

  ```
  uvicorn backend.app.main:app --reload
  
  ```

  API docs will be available at `http://127.0.0.1:8000/docs`.

## API Reference

### `POST /api/ask`

**Request Body:**

```
{
  "question": "What programs are offered by the IT department?",
  "use_cache": true
}

```

**Response Body:**

```
{
  "answer": "The Department of Information Technology offers B.Tech and M.Tech programs [Source 1].",
  "sources": [
    {
      "source_number": 1,
      "title": "Department of Information Technology - MIT",
      "url": "https://departments.auegov.ac.in/it",
      "chunk_id": "page_1_chunk_1"
    }
  ],
  "cached": false
}

```