class DocumentNotFoundError(Exception):
    """Raised when a document is not found in the database."""

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")


class DownloadError(Exception):
    """Raised when downloading a file fails."""

    def __init__(self, url: str, status_code: int | Exception) -> None:
        self.url = url
        self.status_code = status_code
        if isinstance(status_code, int):
            msg = f"HTTP {status_code}"
        else:
            msg = str(status_code)
        super().__init__(f"Failed to download {url}: {msg}")
