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
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "src.services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from src.services.ingestion import get_document

            result = await get_document("doc-1", "user-1")

        assert result == fake_row
        mock_client.table.assert_called_once_with("documents")

    async def test_raises_not_found_when_no_data(self):
        """Raises DocumentNotFoundError when document does not exist."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = None
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "src.services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from src.services.ingestion import get_document

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
            patch("src.services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("src.services.ingestion.get_settings") as mock_get_settings,
        ):
            mock_settings = MagicMock()
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "test-token"
            mock_settings.blob_storage_origin = "https://example.com"
            mock_get_settings.return_value = mock_settings
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from src.services.ingestion import download_pdf

            result = await download_pdf("https://example.com/test.pdf")

        assert result == b"%PDF-1.4 fake content"
        mock_client.get.assert_called_once_with(
            "https://example.com/test.pdf",
            headers={"Authorization": "Bearer test-token"},
            follow_redirects=True,
        )

    async def test_raises_download_error_on_failure(self):
        """Raises DownloadError on non-2xx status."""
        mock_response = httpx.Response(403, content=b"Forbidden")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("src.services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("src.services.ingestion.get_settings") as mock_get_settings,
        ):
            mock_settings = MagicMock()
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "test-token"
            mock_settings.blob_storage_origin = "https://example.com"
            mock_get_settings.return_value = mock_settings
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from src.services.ingestion import download_pdf

            with pytest.raises(DownloadError) as exc_info:
                await download_pdf("https://example.com/test.pdf")
            assert exc_info.value.status_code == 403

    async def test_rejects_blob_url_with_wrong_origin(self):
        """Raises DownloadError when blob_url origin doesn't match allowlist."""
        with patch("src.services.ingestion.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.blob_storage_origin = (
                "https://trusted.blob.vercel-storage.com"
            )
            mock_get_settings.return_value = mock_settings

            from src.services.ingestion import download_pdf

            with pytest.raises(DownloadError) as exc_info:
                await download_pdf("https://evil.com/steal-token.pdf")
            assert exc_info.value.status_code == 403

    async def test_follows_redirects(self):
        """Specifically check that follow_redirects=True is passed to httpx."""
        mock_response = httpx.Response(200, content=b"content")

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("src.services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("src.services.ingestion.get_settings") as mock_get_settings,
        ):
            mock_settings = MagicMock()
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "token"
            mock_settings.blob_storage_origin = "https://example.com"
            mock_get_settings.return_value = mock_settings
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from src.services.ingestion import download_pdf

            await download_pdf("https://example.com/test.pdf")

        mock_client.get.assert_called_once_with(
            "https://example.com/test.pdf",
            headers={"Authorization": "Bearer token"},
            follow_redirects=True,
        )

    async def test_raises_download_error_on_network_exception(self):
        """Raises DownloadError when httpx raises a RequestError."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))

        with (
            patch("src.services.ingestion.httpx.AsyncClient") as mock_client_cls,
            patch("src.services.ingestion.get_settings") as mock_get_settings,
        ):
            mock_settings = MagicMock()
            mock_settings.bookified_blob_read_write_token.get_secret_value.return_value = "token"
            mock_settings.blob_storage_origin = "https://example.com"
            mock_get_settings.return_value = mock_settings
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            from src.services.ingestion import download_pdf

            with pytest.raises(DownloadError) as exc_info:
                await download_pdf("https://example.com/test.pdf")

            assert "Connection failed" in str(exc_info.value)
            assert isinstance(exc_info.value.status_code, httpx.ConnectError)


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

        from src.services.ingestion import chunk_pdf

        chunks = chunk_pdf(pdf_bytes)

        assert isinstance(chunks, list)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert isinstance(chunk, Document)
            assert "page" in chunk.metadata

        # Pages are 1-indexed (first page = 1, second page = 2)
        page_numbers = [int(chunk.metadata["page"]) for chunk in chunks]
        assert page_numbers == [1, 2]

    def test_returns_empty_list_for_blank_pdf(self):
        """A PDF with no text content returns an empty list."""
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        buf = io.BytesIO()
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        from src.services.ingestion import chunk_pdf

        chunks = chunk_pdf(pdf_bytes)
        assert chunks == []


class TestCountStoredChunks:
    """Tests for count_stored_chunks()."""

    async def test_returns_count_of_existing_rows(self):
        """Returns the number of rows already in document_embeddings for a document."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = 5
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "src.services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from src.services.ingestion import count_stored_chunks

            result = await count_stored_chunks("doc-1", "user-1")

        assert result == 5
        mock_client.table.assert_called_once_with("document_embeddings")

    async def test_returns_zero_when_count_is_none(self):
        """Returns 0 when result.count is None (Supabase returns None for empty tables)."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.count = None
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(
            return_value=mock_result
        )

        with patch(
            "src.services.ingestion.get_supabase_client",
            new_callable=AsyncMock,
            return_value=mock_client,
        ):
            from src.services.ingestion import count_stored_chunks

            result = await count_stored_chunks("doc-1", "user-1")

        assert result == 0


def _make_embed_and_store_mocks(
    already_stored: int, fake_embeddings: list[list[float]]
) -> tuple[MagicMock, MagicMock]:
    """Return a pre-wired (mock_client, mock_embeddings_model) pair.

    Args:
        already_stored: Value that the count query should return.
        fake_embeddings: Embeddings the model will return per embed_documents call.

    Returns:
        Tuple of (mock_client, mock_embeddings_model).
    """
    mock_embeddings_model = MagicMock()
    mock_embeddings_model.embed_documents = MagicMock(return_value=fake_embeddings)

    mock_client = MagicMock()
    count_result = MagicMock()
    count_result.count = already_stored
    # count query chain: .select().eq().eq().execute()
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(
        return_value=count_result
    )
    # upsert chain: .upsert().execute()
    mock_client.table.return_value.upsert.return_value.execute = AsyncMock(
        return_value=None
    )
    return mock_client, mock_embeddings_model


class TestEmbedAndStore:
    """Tests for embed_and_store()."""

    async def test_embeds_chunks_and_inserts_rows(self):
        """Generates embeddings for each chunk and inserts into document_embeddings."""
        chunks = [
            Document(page_content="Hello world", metadata={"page": 0}),
            Document(page_content="Goodbye world", metadata={"page": 1}),
        ]
        fake_embeddings = [[0.1] * 768, [0.2] * 768]
        mock_client, mock_embeddings_model = _make_embed_and_store_mocks(
            already_stored=0, fake_embeddings=fake_embeddings
        )

        with (
            patch(
                "src.services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "src.services.ingestion.GoogleGenerativeAIEmbeddings",
                return_value=mock_embeddings_model,
            ),
            patch("src.services.ingestion.asyncio.sleep", new_callable=AsyncMock),
        ):
            from src.services.ingestion import embed_and_store

            await embed_and_store(chunks, "doc-1", "user-1")

        # Both chunks fit in one batch (default batch_size=20).
        mock_embeddings_model.embed_documents.assert_called_once_with(
            ["Hello world", "Goodbye world"]
        )
        upserted_rows = mock_client.table.return_value.upsert.call_args[0][0]
        assert len(upserted_rows) == 2
        assert upserted_rows[0]["document_id"] == "doc-1"
        assert upserted_rows[0]["user_id"] == "user-1"
        assert upserted_rows[0]["content"] == "Hello world"
        assert upserted_rows[0]["embedding"] == fake_embeddings[0]
        assert upserted_rows[0]["metadata"] == {"page": 0}
        assert "chunk_key" in upserted_rows[0]
        assert upserted_rows[1]["metadata"] == {"page": 1}

    async def test_resumes_from_already_stored_chunks(self):
        """Skips already-stored chunks and only embeds and inserts the remainder."""
        chunks = [
            Document(page_content="Hello world", metadata={"page": 0}),
            Document(page_content="Goodbye world", metadata={"page": 1}),
        ]
        # First chunk already persisted; only second needs embedding.
        fake_embeddings = [[0.2] * 768]
        mock_client, mock_embeddings_model = _make_embed_and_store_mocks(
            already_stored=1, fake_embeddings=fake_embeddings
        )

        with (
            patch(
                "src.services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "src.services.ingestion.GoogleGenerativeAIEmbeddings",
                return_value=mock_embeddings_model,
            ),
            patch("src.services.ingestion.asyncio.sleep", new_callable=AsyncMock),
        ):
            from src.services.ingestion import embed_and_store

            await embed_and_store(chunks, "doc-1", "user-1")

        mock_embeddings_model.embed_documents.assert_called_once_with(["Goodbye world"])
        upserted_rows = mock_client.table.return_value.upsert.call_args[0][0]
        assert len(upserted_rows) == 1
        assert upserted_rows[0]["content"] == "Goodbye world"
        assert upserted_rows[0]["embedding"] == fake_embeddings[0]

    async def test_does_nothing_when_all_chunks_already_stored(self):
        """Returns early without any embedding or insert when all chunks are stored."""
        chunks = [Document(page_content="Hello world", metadata={"page": 0})]
        mock_client, mock_embeddings_model = _make_embed_and_store_mocks(
            already_stored=1, fake_embeddings=[]
        )

        with (
            patch(
                "src.services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch(
                "src.services.ingestion.GoogleGenerativeAIEmbeddings",
                return_value=mock_embeddings_model,
            ),
        ):
            from src.services.ingestion import embed_and_store

            await embed_and_store(chunks, "doc-1", "user-1")

        mock_embeddings_model.embed_documents.assert_not_called()
        mock_client.table.return_value.upsert.assert_not_called()


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
                "src.services.ingestion.get_supabase_client",
                new_callable=AsyncMock,
                return_value=mock_client,
            ),
            patch("src.services.ingestion.datetime") as mock_dt,
        ):
            fake_now = datetime(2026, 3, 27, 12, 0, 0, tzinfo=UTC)
            mock_dt.now.return_value = fake_now

            from src.services.ingestion import mark_ingested

            await mark_ingested("doc-1")

        mock_client.table.assert_called_once_with("documents")
        update_arg = mock_client.table.return_value.update.call_args[0][0]
        assert update_arg["ingested_at"] == fake_now.isoformat()
        mock_client.table.return_value.update.return_value.eq.assert_called_once_with(
            "id", "doc-1"
        )
