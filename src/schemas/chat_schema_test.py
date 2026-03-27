from src.schemas.chat import ChatRequest, ChatResponse, Citation


def test_chat_request_accepts_message():
    """Test ChatRequest accepts message field."""
    request = ChatRequest(message="What is this about?")
    assert request.message == "What is this about?"


def test_chat_response_with_citations():
    """Test ChatResponse with citations."""
    response = ChatResponse(
        answer="The author argues this [p. 12].", citations=[Citation(page=12)]
    )
    assert response.answer == "The author argues this [p. 12]."
    assert len(response.citations) == 1
    assert response.citations[0].page == 12


def test_chat_response_empty_citations():
    """Test ChatResponse with empty citations."""
    response = ChatResponse(answer="No info available.")
    assert response.answer == "No info available."
    assert response.citations == []
