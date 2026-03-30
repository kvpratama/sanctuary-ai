import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.schemas.chat import Citation


def parse_sse_events(body: str) -> list[dict[str, str]]:
    """Parse SSE formatted text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = None
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current_data = line[len("data:") :].strip()
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": current_data or ""})
            current_event = None
            current_data = None
    if current_event is not None:
        events.append({"event": current_event, "data": current_data or ""})
    return events


@pytest.mark.asyncio
async def test_chat_stream_emits_token_citations_done():
    """Test SSE stream emits token events, then citations, then done."""
    document_id = "test-doc-123"

    async def fake_stream(query, chunks):
        yield "The author "
        yield "argues that [p. 12]."
        yield ([Citation(page=12)],)

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
            response = await client.post(
                f"/chat/{document_id}",
                json={"message": "What is the main argument?"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        events = parse_sse_events(response.text)

        # Collect token events
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 2
        assert json.loads(token_events[0]["data"]) == "The author "
        assert json.loads(token_events[1]["data"]) == "argues that [p. 12]."

        # Citations event
        citation_events = [e for e in events if e["event"] == "citations"]
        assert len(citation_events) == 1
        assert json.loads(citation_events[0]["data"]) == [{"page": 12}]

        # Done event
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert json.loads(done_events[0]["data"]) == {}


@pytest.mark.asyncio
async def test_chat_stream_with_no_results():
    """Test SSE stream when no relevant chunks found."""
    document_id = "test-doc-456"

    async def fake_stream(query, chunks):
        yield "I don't have enough information."
        yield ([],)

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
            response = await client.post(
                f"/chat/{document_id}",
                json={"message": "Unknown topic"},
            )

        assert response.status_code == 200
        events = parse_sse_events(response.text)

        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 1

        citation_events = [e for e in events if e["event"] == "citations"]
        assert json.loads(citation_events[0]["data"]) == []

        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1


@pytest.mark.asyncio
async def test_chat_stream_error_handling():
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
