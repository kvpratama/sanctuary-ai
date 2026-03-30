import logging

from fastapi import APIRouter, Depends, HTTPException

from src.auth import AuthenticatedUser, get_authenticated_user
from src.services.exceptions import DocumentNotFoundError, DownloadError
from src.services.ingestion import (
    chunk_pdf,
    download_pdf,
    embed_and_store,
    get_document,
    mark_ingested,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest/{document_id}")
async def ingest_document(
    document_id: str,
    user: AuthenticatedUser = Depends(get_authenticated_user),
) -> dict[str, str | int]:
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
    try:
        doc = await get_document(document_id, user.id, client=user.client)

        if doc["ingested_at"]:
            return {"status": "already_ingested"}

        pdf_bytes = await download_pdf(doc["blob_url"])
        chunks = chunk_pdf(pdf_bytes)

        await embed_and_store(chunks, document_id, user.id, client=user.client)
        await mark_ingested(document_id, client=user.client)

        return {"status": "ingested", "chunk_count": len(chunks)}

    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except DownloadError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("Ingestion failed for document %s", document_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
