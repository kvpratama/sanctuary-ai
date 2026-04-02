"""Tests for the chat router endpoints."""

import json
from collections.abc import AsyncGenerator
from typing import Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.auth import get_authenticated_user
from src.schemas.chat import Citation, CitationsEvent, TokenEvent

USER_ID = "f9937aab-6c97-4c3e-a6f8-38f4a1676200"


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
        yield TokenEvent(token="The author ")
        yield TokenEvent(token="argues that [p. 12].")
        yield CitationsEvent(citations=[Citation(page=12)])

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.chat.retrieve_chunks",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_retrieve,
            patch(
                "src.routers.chat.stream_answer_with_citations", side_effect=fake_stream
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                lines: list[str] = []
                async with client.stream(
                    "POST",
                    f"/chat/{document_id}",
                    json={"message": "What is the main argument?"},
                    headers={"Authorization": "Bearer fake-token"},
                ) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers["content-type"]
                    async for line in response.aiter_lines():
                        lines.append(line)

        events = parse_sse_events(lines)
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "The author "),
            ("token", "argues that [p. 12]."),
            ("citations", [{"page": 12}]),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence
        mock_retrieve.assert_called_once_with(
            query="What is the main argument?",
            document_id=document_id,
            user_id=USER_ID,
            k=5,
            client=mock_auth_user.client,
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_stream_with_no_results() -> None:
    """Test SSE stream when no relevant chunks found."""
    document_id = "test-doc-456"

    async def fake_stream(
        query: str, chunks: int
    ) -> AsyncGenerator[Union[TokenEvent, CitationsEvent], None]:
        yield TokenEvent(token="I don't have enough information.")
        yield CitationsEvent(citations=[])

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.chat.retrieve_chunks",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_retrieve,
            patch(
                "src.routers.chat.stream_answer_with_citations", side_effect=fake_stream
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                lines: list[str] = []
                async with client.stream(
                    "POST",
                    f"/chat/{document_id}",
                    json={"message": "Unknown topic"},
                    headers={"Authorization": "Bearer fake-token"},
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        lines.append(line)

        events = parse_sse_events(lines)
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "I don't have enough information."),
            ("citations", []),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence
        mock_retrieve.assert_called_once_with(
            query="Unknown topic",
            document_id=document_id,
            user_id=USER_ID,
            k=5,
            client=mock_auth_user.client,
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_stream_error_handling() -> None:
    """Test SSE endpoint error handling."""
    document_id = "test-doc-789"
    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.chat.retrieve_chunks",
                new_callable=AsyncMock,
                side_effect=Exception("Database error"),
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    f"/chat/{document_id}",
                    json={"message": "Test question"},
                    headers={"Authorization": "Bearer fake-token"},
                )

            assert response.status_code == 500
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_stream_error_during_streaming() -> None:
    """Test SSE endpoint emits error event when streaming fails mid-stream."""
    document_id = "test-doc-stream-err"

    async def failing_stream(
        query: str, chunks: int
    ) -> AsyncGenerator[TokenEvent, None]:
        yield TokenEvent(token="partial token")
        raise RuntimeError("LLM connection lost")

    mock_auth_user = MagicMock()
    mock_auth_user.id = USER_ID
    mock_auth_user.client = AsyncMock()

    app.dependency_overrides[get_authenticated_user] = lambda: mock_auth_user

    try:
        with (
            patch(
                "src.routers.chat.retrieve_chunks",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_retrieve,
            patch(
                "src.routers.chat.stream_answer_with_citations",
                side_effect=failing_stream,
            ),
        ):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                lines: list[str] = []
                async with client.stream(
                    "POST",
                    f"/chat/{document_id}",
                    json={"message": "Test question"},
                    headers={"Authorization": "Bearer fake-token"},
                ) as response:
                    assert response.status_code == 200
                    async for line in response.aiter_lines():
                        lines.append(line)

        events = parse_sse_events(lines)
        event_sequence = [(e["event"], json.loads(e["data"])) for e in events]
        expected_sequence = [
            ("token", "partial token"),
            ("error", {"detail": "An error occurred while generating the answer."}),
            ("done", {}),
        ]
        assert event_sequence == expected_sequence
        mock_retrieve.assert_called_once_with(
            query="Test question",
            document_id=document_id,
            user_id=USER_ID,
            k=5,
            client=mock_auth_user.client,
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_without_auth_returns_401() -> None:
    """Chat endpoint without auth header returns 401."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/chat/some-doc-id", json={"message": "hello"})
    assert response.status_code == 401
