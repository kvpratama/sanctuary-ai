import io
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader

from src.config import settings, EMBEDDING_DIMENSIONS, CHUNK_SIZE, CHUNK_OVERLAP
from src.db.client import get_supabase_client
from src.services.exceptions import DocumentNotFoundError, DownloadError


async def get_document(document_id: str, user_id: str) -> dict[str, Any]:
    """Fetch a document from the documents table.

    Args:
        document_id: UUID of the document.
        user_id: UUID of the owning user.

    Returns:
        Document row as a dict.

    Raises:
        DocumentNotFoundError: If the document does not exist.
    """
    client = await get_supabase_client()
    result = await (
        client.table("documents")
        .select("*")
        .eq("id", document_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    if result.data is None:
        raise DocumentNotFoundError(document_id)
    return result.data  # ty: ignore[invalid-return-type]


async def download_pdf(blob_url: str) -> bytes:
    """Download a PDF from a public blob URL.

    Args:
        blob_url: Public URL of the PDF file.

    Returns:
        Raw PDF bytes.

    Raises:
        DownloadError: If the HTTP response is not 2xx.
    """
    token = settings.bookified_blob_read_write_token.get_secret_value()
    async with httpx.AsyncClient() as client:
        response = await client.get(
            blob_url,
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code >= 400:
        raise DownloadError(blob_url, response.status_code)
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
    for page_num, page in enumerate(reader.pages):
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


async def embed_and_store(
    chunks: list[Document], document_id: str, user_id: str
) -> None:
    """Generate embeddings for chunks and store in Supabase.

    Args:
        chunks: List of LangChain Document objects to embed.
        document_id: UUID of the parent document.
        user_id: UUID of the owning user.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.gemini_api_key,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)

    rows = [
        {
            "document_id": document_id,
            "user_id": user_id,
            "content": chunk.page_content,
            "embedding": embedding,
            "metadata": chunk.metadata,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client = await get_supabase_client()
    await client.table("document_embeddings").insert(rows).execute()


async def mark_ingested(document_id: str) -> None:
    """Mark a document as ingested by setting ingested_at to now.

    Args:
        document_id: UUID of the document to mark.
    """
    client = await get_supabase_client()
    now = datetime.now(timezone.utc).isoformat()
    await (
        client.table("documents")
        .update({"ingested_at": now})
        .eq("id", document_id)
        .execute()
    )
