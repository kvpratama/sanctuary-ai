from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(..., description="User's question")


class Citation(BaseModel):
    """A citation referencing a page number."""

    page: int = Field(..., description="Page number from the document")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    answer: str = Field(..., description="The generated answer with citations")
    citations: list[Citation] = Field(
        default_factory=list,
        description="List of page citations used in the answer",
    )
