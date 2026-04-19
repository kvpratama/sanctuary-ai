# Sanctuary AI

AI-powered document analysis and retrieval system. Upload PDFs, ingest them into a vector store, and chat with your documents using RAG (Retrieval-Augmented Generation).


## RAG Strategies

The system supports six pluggable RAG strategies, configurable via the `RAG_STRATEGY` environment variable. Each strategy trades off latency, cost, and answer quality differently.

### `naive_rag`
Retrieves the top-`k` chunks most similar to the user's query, then passes them directly to the LLM to generate an answer. Fast and cheap, but sensitive to query phrasing — if the user's wording doesn't match the document's language well, retrieval quality suffers.

### `query_rewrite`
Before retrieval, an LLM rewrites the user's query into a search-optimized form better suited for semantic similarity matching. The rewritten query is used for retrieval, but the **original** query is used for answer generation to preserve the user's intent. Best balance of quality and cost for most use cases.

### `multi_query`
Generates `n` rephrased variants of the original query using an LLM, retrieves chunks for each variant, then deduplicates results before generating a single answer. Casts a wider retrieval net, which helps for broad or ambiguous questions — at the cost of more LLM calls and slightly lower precision.

### `self_correcting`
Adds a relevance grading loop after retrieval: each retrieved chunk is scored by an LLM, and irrelevant ones are filtered out. If too few relevant chunks remain (below `MIN_RELEVANT_CHUNKS`), the query is rewritten and retrieval is retried (up to `MAX_RETRIES` times). **v2** moves the rewrite step *before* the first retrieval attempt, improving correctness at the cost of an extra LLM call upfront.

### `agentic_rag`
Runs a fully autonomous LLM agent equipped with a `search_docs` tool. The agent decides what to search for, how many times to search, and when it has gathered enough context to answer. Iteration is capped by `ToolCallLimitMiddleware`. Most flexible for complex multi-part questions, but highest latency and lowest consistency.


### Strategy Comparison

| Strategy | Correctness | Groundedness | Relevance | Retrieval Quality | Best for |
|---|---|---|---|---|---|
| `naive_rag` | 0.65 | **0.85** | 0.85 | 0.70 | Simple, fast lookups |
| `query_rewrite` | 0.70 | **0.85** | **0.95** | 0.70 | Most queries; best precision |
| `multi_query` | 0.65 | **0.85** | 0.85 | 0.60 | Broad or ambiguous questions |
| `self_correcting` v1 | 0.70 | 0.75 | 0.75 | 0.75 | Noisy documents |
| `self_correcting` v2 | **0.75** | 0.75 | 0.70 | **0.75** | Highest correctness needed |
| `agentic_rag` | 0.55 | 0.75 | 0.75 | 0.70 | Complex, multi-part questions |

> **Metrics** (LLM-as-judge scores averaged across eval dataset, 0–1):
> - **Correctness** — does the answer match the expected answer?
> - **Relevance** — does the answer address the question?
> - **Groundedness** — is the answer supported by retrieved chunks, without hallucination?
> - **Retrieval Quality** — do the retrieved chunks contain information useful for answering?

> **Recommended Strategy:** `query_rewrite` — it delivers the highest relevance score and strong groundedness with only a modest latency increase over `naive_rag`.


## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI, Uvicorn |
| RAG framework | LangGraph, LangChain |
| LLM providers | OpenAI, Google Gemini |
| Database / vectors | Supabase (PostgreSQL + pgvector) |
| Evaluation | LangSmith |
| Package manager | uv |


## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check |
| `POST` | `/ingest/{document_id}` | Ingest a PDF (download, chunk, embed, store) |
| `POST` | `/chat/{document_id}` | Chat with a document via SSE streaming |

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Supabase project (for database + vector store)
- OpenAI and/or Google Gemini API keys

### Setup

```bash
# Clone the repository
git clone https://github.com/kvpratama/sanctuary-ai.git
cd sanctuary-ai

# Install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys and Supabase credentials
```

### Running

```bash
# Start the development server
vercel dev
```

### Testing

```bash
# Run unit tests
uv run pytest

# Run all tests including integration (requires live Supabase, LLM, and LangSmith)
uv run pytest -m ""
```

### Linting & Formatting

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```


## Project Structure

```
src/
├── app.py                  # FastAPI application entry point
├── auth.py                 # Authentication (Supabase JWT)
├── config.py               # Settings (Pydantic BaseSettings)
├── db/
│   ├── client.py           # Supabase client singleton
│   ├── database_types.py   # Auto-generated DB types (do not edit)
│   └── schema.sql          # Auto-generated schema (do not edit)
├── eval/                   # LangSmith evaluation pipeline
│   ├── dataset.py          # Dataset management
│   ├── evaluators.py       # LLM-as-Judge evaluators
│   ├── jury.py             # Jury-of-judges evaluation
│   ├── run.py              # Evaluation runner
│   └── target.py           # Target functions for evaluation
├── prompts/                # Prompt management
│   ├── manager.py          # Prompt loading
│   └── push.py             # Push prompts to LangSmith
├── routers/
│   ├── chat.py             # /chat endpoint (SSE streaming)
│   └── ingest.py           # /ingest endpoint
├── schemas/
│   └── chat.py             # Pydantic request/response models
└── services/
    ├── exceptions.py       # Custom exception types
    ├── ingestion.py        # PDF download, chunking, embedding
    ├── retrieval.py        # RAG pipeline orchestration
    └── strategies/         # Pluggable RAG strategies
        ├── core.py         # Base strategy interface
        ├── registry.py     # Strategy registry
        ├── naive_rag.py
        ├── query_rewrite.py
        ├── multi_query.py
        ├── self_correcting.py
        └── agentic_rag.py
```


## License

[MIT](LICENSE)
