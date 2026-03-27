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
