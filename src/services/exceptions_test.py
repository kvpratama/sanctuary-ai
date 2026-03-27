from services.exceptions import DocumentNotFoundError, DownloadError


def test_document_not_found_error_has_document_id():
    """DocumentNotFoundError stores document_id and has useful message."""
    err = DocumentNotFoundError("abc-123")
    assert err.document_id == "abc-123"
    assert "abc-123" in str(err)


def test_download_error_has_url_and_status():
    """DownloadError stores url and status_code."""
    err = DownloadError("https://example.com/file.pdf", 403)
    assert err.url == "https://example.com/file.pdf"
    assert err.status_code == 403
    assert "403" in str(err)
