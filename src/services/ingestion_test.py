import io
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from src.services.exceptions import DocumentNotFoundError, DownloadError


class TestGetDocument:
    """Tests for get_document()."""

    async def test_returns_document_when_found(self):
        """Returns document data when it exists and belongs to user."""
        mock_client = MagicMock()
        fake_row = {"id": "doc-1", "user_id": "user-1", "name": "test.pdf"}
        mock_result = MagicMock()
        mock_result.data = fake_row
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from services.ingestion import get_document

            result = await get_document("doc-1", "user-1")

        assert result == fake_row
        mock_client.table.assert_called_once_with("documents")

    async def test_raises_not_found_when_no_data(self):
        """Raises DocumentNotFoundError when document does not exist."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = None
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.single.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from services.ingestion import get_document

            with pytest.raises(DocumentNotFoundError):
                await get_document("nonexistent", "user-1")


class TestDownloadPdf:
    """Tests for download_pdf()."""

    async def test_returns_bytes_on_success(self):
        """Returns PDF bytes when download succeeds."""
        mock_response = httpx.Response(200, content=b"%PDF-1.4 fake content")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("services.ingestion.settings") as mock_settings,
        ):
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "test-token"
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from services.ingestion import download_pdf

            result = await download_pdf("https://example.com/test.pdf")

        assert result == b"%PDF-1.4 fake content"
        mock_client.get.assert_called_once_with(
            "https://example.com/test.pdf",
            headers={"Authorization": "Bearer test-token"},
        )

    async def test_raises_download_error_on_failure(self):
        """Raises DownloadError on non-2xx status."""
        mock_response = httpx.Response(403, content=b"Forbidden")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("services.ingestion.settings") as mock_settings,
        ):
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "test-token"
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from services.ingestion import download_pdf

            with pytest.raises(DownloadError) as exc_info:
                await download_pdf("https://example.com/test.pdf")
            assert exc_info.value.status_code == 403


def _make_pdf_bytes_with_text(pages: list[str]) -> bytes:
    """Helper: create a minimal PDF with text on each page."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()
    return buf.getvalue()


class TestChunkPdf:
    """Tests for chunk_pdf()."""

    def test_returns_documents_with_page_metadata(self):
        """Chunks contain page number in metadata."""
        pdf_bytes = _make_pdf_bytes_with_text(["Page one content", "Page two content"])

        from services.ingestion import chunk_pdf

        chunks = chunk_pdf(pdf_bytes)

        assert isinstance(chunks, list)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert "page" in chunk.metadata

    def test_returns_empty_list_for_blank_pdf(self):
        """A PDF with no text content returns an empty list."""
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        from services.ingestion import chunk_pdf

        chunks = chunk_pdf(pdf_bytes)
        assert chunks == []


class TestEmbedAndStore:
    """Tests for embed_and_store()."""

    async def test_embeds_chunks_and_inserts_rows(self):
        """Generates embeddings for each chunk and inserts into document_embeddings."""
        chunks = [
            Document(page_content="Hello world", metadata={"page": 0}),
            Document(page_content="Goodbye world", metadata={"page": 1}),
        ]
        fake_embeddings = [[0.1] * 768, [0.2] * 768]

        mock_embeddings_model = MagicMock()
        mock_embeddings_model.embed_documents = MagicMock(return_value=fake_embeddings)

        mock_client = MagicMock()
        mock_client.table.return_value.insert.return_value.execute = AsyncMock(
            return_value=None
        )

        with (
            patch(
                "services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "services.ingestion.GoogleGenerativeAIEmbeddings",
                return_value=mock_embeddings_model,
            ),
        ):
            from services.ingestion import embed_and_store

            await embed_and_store(chunks, "doc-1", "user-1")

        mock_embeddings_model.embed_documents.assert_called_once_with(
            ["Hello world", "Goodbye world"]
        )
        mock_client.table.assert_called_once_with("document_embeddings")
        inserted_rows = mock_client.table.return_value.insert.call_args[0][0]
        assert len(inserted_rows) == 2
        assert inserted_rows[0]["document_id"] == "doc-1"
        assert inserted_rows[0]["user_id"] == "user-1"
        assert inserted_rows[0]["content"] == "Hello world"
        assert inserted_rows[0]["embedding"] == fake_embeddings[0]
        assert inserted_rows[0]["metadata"] == {"page": 0}
        assert inserted_rows[1]["metadata"] == {"page": 1}


class TestMarkIngested:
    """Tests for mark_ingested()."""

    async def test_updates_ingested_at(self):
        """Sets ingested_at to current UTC time."""
        mock_client = MagicMock()
        mock_client.table.return_value.update.return_value.eq.return_value.execute = (
            AsyncMock(return_value=None)
        )

        with (
            patch(
                "services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch("services.ingestion.datetime") as mock_dt,
        ):
            fake_now = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)
            mock_dt.now.return_value = fake_now

            from services.ingestion import mark_ingested

            await mark_ingested("doc-1")

        mock_client.table.assert_called_once_with("documents")
        update_arg = mock_client.table.return_value.update.call_args[0][0]
        assert update_arg["ingested_at"] == fake_now.isoformat()
        mock_client.table.return_value.update.return_value.eq.assert_called_once_with(
            "id", "doc-1"
        )
