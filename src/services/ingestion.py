import asyncio
import hashlib
import io
import logging
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIMENSIONS, get_settings
from src.db.client import get_supabase_client
from src.services.exceptions import DocumentNotFoundError, DownloadError
from supabase import AsyncClient

logger = logging.getLogger(__name__)

TOKENS_PER_MINUTE = 30_000
REQUESTS_PER_MINUTE = 100


async def get_document(
    document_id: str, user_id: str, *, client: AsyncClient | None = None
) -> dict[str, Any]:
    """Fetch a document from the documents table.

    Args:
        document_id: UUID of the document.
        user_id: UUID of the owning user.

    Returns:
        Document row as a dict.

    Raises:
        DocumentNotFoundError: If the document does not exist.
    """
    if client is None:
        client = await get_supabase_client()
    result = await (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if result is None or result.data is None:
        raise DocumentNotFoundError(document_id)
    data: dict[str, Any] = result.data  # ty: ignore[invalid-assignment]
    return data


async def download_pdf(blob_url: str) -> bytes:
    """Download a PDF from a public blob URL.

    Args:
        blob_url: Public URL of the PDF file.

    Returns:
        Raw PDF bytes.

    Raises:
        DownloadError: If the HTTP response is not 2xx.
    """
    settings = get_settings()

    parsed = urlparse(blob_url)
    blob_url_origin = f"{parsed.scheme}://{parsed.netloc}"
    if blob_url_origin != settings.blob_storage_origin:
        raise DownloadError(blob_url, 403)

    token = settings.bookified_blob_read_write_token.get_secret_value()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                blob_url,
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=True,
            )
        if response.status_code >= 300:
            raise DownloadError(blob_url, response.status_code)
    except (httpx.RequestError, httpx.HTTPStatusError) as err:
        raise DownloadError(blob_url, err) from err
    return response.content


def chunk_pdf(pdf_bytes: bytes) -> list[Document]:
    """Extract text from PDF bytes and split into chunks.

    Args:
        pdf_bytes: Raw PDF file content.

    Returns:
        List of LangChain Document objects with page metadata.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    documents: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_num}))

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def estimate_tokens(texts):
    # Rough estimate: 1 token ≈ 4 characters
    return sum(len(t) for t in texts) // 4


async def embed_with_retry(embeddings_model, batch, max_retries=5):
    wait = 2
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(embeddings_model.embed_documents, batch)
        except GoogleGenerativeAIError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Rate limited. Waiting {wait}s before retry...")
            logger.warning(f"Rate limited. Waiting {wait}s before retry...")
            await asyncio.sleep(wait)
            wait *= 2
        except Exception:
            raise


async def count_stored_chunks(
    document_id: str, user_id: str, *, client: AsyncClient | None = None
) -> int:
    """Return the number of embeddings already persisted for a document.

    Used by embed_and_store to resume an interrupted ingestion run without
    re-embedding chunks that were already saved to Supabase.

    Args:
        document_id: UUID of the parent document.
        user_id: UUID of the owning user.

    Returns:
        Number of rows currently in document_embeddings for this document.
    """
    if client is None:
        client = await get_supabase_client()
    result = (
        await client.table("document_embeddings")
        .select("id", count="exact")  # ty: ignore[invalid-argument-type]
        .eq("document_id", document_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.count or 0


async def embed_and_store(
    chunks: list[Document],
    document_id: str,
    user_id: str,
    batch_size: int = 20,
    *,
    client: AsyncClient | None = None,
) -> None:
    """Generate embeddings for chunks and store them in Supabase batch-by-batch.

    Each batch is persisted to the database immediately after it is embedded,
    so a partial run can be resumed without re-embedding already-stored chunks.
    On restart, the function queries how many chunks are already saved for this
    document and skips that many from the front of *chunks* before continuing.

    Args:
        chunks: List of LangChain Document objects to embed.
        document_id: UUID of the parent document.
        user_id: UUID of the owning user.
        batch_size: Number of chunks to embed and insert per round-trip.
    """
    if client is None:
        client = await get_supabase_client()

    already_stored = await count_stored_chunks(document_id, user_id, client=client)
    if already_stored:
        print(
            f"Resuming: skipping {already_stored} already-stored chunks out of {len(chunks)}."
        )
    else:
        print(f"Starting ingestion for {len(chunks)} chunks.")
    remaining_chunks = chunks[already_stored:]

    if not remaining_chunks:
        print("All chunks already stored — nothing to do.")
        return

    settings = get_settings()
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.gemini_api_key,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )

    requests_this_minute = 0
    tokens_this_minute = 0
    window_start = time.time()

    for i in range(0, len(remaining_chunks), batch_size):
        batch_chunks = remaining_chunks[i : i + batch_size]
        batch_texts = [chunk.page_content for chunk in batch_chunks]
        batch_tokens = estimate_tokens(batch_texts)

        # Reset rate-limit window if 60 s have elapsed.
        elapsed = time.time() - window_start
        if elapsed >= 60:
            requests_this_minute = 0
            tokens_this_minute = 0
            window_start = time.time()

        # Pause when approaching API rate limits.
        if (
            requests_this_minute >= REQUESTS_PER_MINUTE // 2
            or tokens_this_minute + batch_tokens > TOKENS_PER_MINUTE // 2
        ):
            wait = 60 - elapsed
            print(
                f"Approaching limit (reqs={requests_this_minute}, "
                f"tokens={tokens_this_minute}). Waiting {wait:.1f}s..."
            )
            await asyncio.sleep(max(wait, 0))
            requests_this_minute = 0
            tokens_this_minute = 0
            window_start = time.time()

        batch_num = already_stored // batch_size + i // batch_size + 1
        print(f"Embedding batch {batch_num} (~{batch_tokens} tokens)...")
        batch_embeddings = await embed_with_retry(embeddings_model, batch_texts)

        rows = [
            {
                "document_id": document_id,
                "user_id": user_id,
                "content": chunk.page_content,
                "embedding": embedding,
                "metadata": chunk.metadata,
                "chunk_key": hashlib.sha256(
                    f"{document_id}:{already_stored + i + j}".encode()
                ).hexdigest(),
            }
            for j, (chunk, embedding) in enumerate(
                zip(batch_chunks, batch_embeddings, strict=True)
            )
        ]

        # Persist this batch immediately so progress is never lost.
        # Upsert ensures idempotency if a batch is partially re-processed.
        await (
            client.table("document_embeddings")
            .upsert(rows, on_conflict="document_id,chunk_key")
            .execute()
        )
        print(f"Batch {batch_num} stored ({len(rows)} chunks).")

        requests_this_minute += 1
        tokens_this_minute += batch_tokens
        await asyncio.sleep(5)


async def mark_ingested(document_id: str, *, client: AsyncClient | None = None) -> None:
    """Mark a document as ingested by setting ingested_at to now.

    Args:
        document_id: UUID of the document to mark.
        client: Optional authenticated Supabase client for RLS enforcement.
    """
    if client is None:
        client = await get_supabase_client()
    now = datetime.now(UTC).isoformat()
    await (
        client.table("documents")
        .update({"ingested_at": now})
        .eq("id", document_id)
        .execute()
    )
