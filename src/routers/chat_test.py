import json
from typing import AsyncGenerator, Union
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.schemas.chat import Citation, CitationsEvent, TokenEvent


def parse_sse_events(lines: list[str]) -> list[dict[str, str]]:
    """Parse incremental SSE lines into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            current_data = []
        elif line.startswith("data:"):
            current_data.append(line[len("data:") :].strip())
        elif line == "" and current_event is not None:
            events.append(
                {
                    "event": current_event,
                    "data": "\n".join(current_data) if current_data else "",
                }
            )
            current_event = None
            current_data = []
    if current_event is not None:
        events.append(
            {
                "event": current_event,
                "data": "\n".join(current_data) if current_data else "",
            }
        )
    return events


@pytest.mark.asyncio
async def test_chat_stream_emits_token_citations_done() -> None:
    """Test SSE stream emits token events, then citations, then done."""
    document_id = "test-doc-123"

    async def fake_stream(
        query: str, chunks: int
    ) -> AsyncGenerator[Union[TokenEvent, CitationsEvent], None]:
        """Fake streaming generator that yields token and citation events.

        Args:
            query: The user's query string (unused in fake implementation).
            chunks: Number of chunks to retrieve (unused in fake implementation).

        Yields:
            TokenEvent instances for each token, followed by a CitationsEvent.
        """
        yield TokenEvent(token="The author ")
        yield TokenEvent(token="argues that [p. 12].")
        yield CitationsEvent(citations=[Citation(page=12)])

    with (
        patch(
            "src.routers.chat.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.routers.chat.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            lines: list[str] = []
            async with client.stream(
                "POST",
                f"/chat/{document_id}",
                json={"message": "What is the main argument?"},
            ) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]
                async for line in response.aiter_lines():
                    lines.append(line)

        events = parse_sse_events(lines)

        # Assert ordered sequence of events
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "The author "),
            ("token", "argues that [p. 12]."),
            ("citations", [{"page": 12}]),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence


@pytest.mark.asyncio
async def test_chat_stream_with_no_results() -> None:
    """Test SSE stream when no relevant chunks found."""
    document_id = "test-doc-456"

    async def fake_stream(
        query: str, chunks: int
    ) -> AsyncGenerator[Union[TokenEvent, CitationsEvent], None]:
        """Fake streaming generator that yields token and empty citation events.

        Args:
            query: The user's query string (unused in fake implementation).
            chunks: Number of chunks to retrieve (unused in fake implementation).

        Yields:
            TokenEvent instance followed by a CitationsEvent with empty citations.
        """
        yield TokenEvent(token="I don't have enough information.")
        yield CitationsEvent(citations=[])

    with (
        patch(
            "src.routers.chat.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.routers.chat.stream_answer_with_citations",
            side_effect=fake_stream,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            lines: list[str] = []
            async with client.stream(
                "POST",
                f"/chat/{document_id}",
                json={"message": "Unknown topic"},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    lines.append(line)

        events = parse_sse_events(lines)

        # Assert ordered sequence of events
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "I don't have enough information."),
            ("citations", []),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence


@pytest.mark.asyncio
async def test_chat_stream_error_handling() -> None:
    """Test SSE endpoint error handling."""
    document_id = "test-doc-789"

    with patch(
        "src.routers.chat.retrieve_chunks",
        new_callable=AsyncMock,
        side_effect=Exception("Database error"),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/chat/{document_id}",
                json={"message": "Test question"},
            )

        assert response.status_code == 500


@pytest.mark.asyncio
async def test_chat_stream_error_during_streaming() -> None:
    """Test SSE endpoint emits error event when streaming fails mid-stream."""
    document_id = "test-doc-stream-err"

    async def failing_stream(
        query: str, chunks: int
    ) -> AsyncGenerator[TokenEvent, None]:
        """Fake streaming generator that yields a token then raises an error.

        Args:
            query: The user's query string (unused in fake implementation).
            chunks: Number of chunks to retrieve (unused in fake implementation).

        Yields:
            A single TokenEvent before raising RuntimeError.

        Raises:
            RuntimeError: Simulates LLM connection failure mid-stream.
        """
        yield TokenEvent(token="partial token")
        raise RuntimeError("LLM connection lost")

    with (
        patch(
            "src.routers.chat.retrieve_chunks",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.routers.chat.stream_answer_with_citations",
            side_effect=failing_stream,
        ),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            lines: list[str] = []
            async with client.stream(
                "POST",
                f"/chat/{document_id}",
                json={"message": "Test question"},
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    lines.append(line)

        events = parse_sse_events(lines)

        # Assert ordered sequence of events
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "partial token"),
            ("error", {"detail": "An error occurred while generating the answer."}),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence
