from fastapi import APIRouter

from src.services.ingestion import (
    chunk_pdf,
    download_pdf,
    embed_and_store,
    get_document,
    mark_ingested,
)

router = APIRouter()


@router.post("/ingest/{document_id}")
async def ingest_document(document_id: str):
    """Ingest a document by downloading, chunking, and embedding it.

    Orchestrates the full ingestion pipeline for a document:
    1. Fetch document metadata from the database
    2. Check if already ingested (skip if so)
    3. Download PDF from blob storage
    4. Split PDF into text chunks
    5. Generate embeddings and store in vector database
    6. Mark document as ingested

    Args:
        document_id: UUID of the document to ingest.

    Returns:
        Dict with status and chunk count on success, or status on skip.
    """
    # Hardcode a test user_id for now
    user_id = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"

    doc = await get_document(document_id, user_id)

    if doc["ingested_at"]:
        return {"status": "already_ingested"}

    pdf_bytes = await download_pdf(doc["blob_url"])
    chunks = chunk_pdf(pdf_bytes)

    await embed_and_store(chunks, document_id, user_id)
    await mark_ingested(document_id)

    return {"status": "ingested", "chunk_count": len(chunks)}
