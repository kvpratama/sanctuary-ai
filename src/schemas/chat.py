from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(..., description="User's question")


class Citation(BaseModel):
    """A citation referencing a page number."""

    page: int = Field(..., description="Page number from the document")


class TokenEvent(BaseModel):
    """A streaming event carrying a single token."""

    type: Literal["token"] = "token"
    token: str


class ChunksEvent(BaseModel):
    """A streaming event carrying the retrieved document chunks."""

    type: Literal["chunks"] = "chunks"
    chunks: list[dict] = Field(
        ..., description="Retrieved chunks with page_content and metadata"
    )


class CitationsEvent(BaseModel):
    """A streaming event carrying the final citations list."""

    type: Literal["citations"] = "citations"
    citations: list[Citation]


StreamEvent = TokenEvent | CitationsEvent | ChunksEvent


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    answer: str = Field(..., description="The generated answer with citations")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of page citations used in the answer",
    )
