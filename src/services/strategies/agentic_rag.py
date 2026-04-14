"""Agentic RAG strategy -- LLM agent with search tool.

Uses LangChain's create_agent() with a search_docs tool backed by
retrieve_chunks. The agent autonomously searches, gathers context,
and decides when to answer.  The agent's own streamed answer is used
directly — no second LLM call is needed.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_core.tools import tool
from langsmith import traceable

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import (
    ChunksEvent,
    CitationsEvent,
    RetrievedChunk,
    StreamEvent,
    TokenEvent,
)
from src.services.strategies.core import extract_citations, retrieve_chunks
from supabase import AsyncClient

logger = logging.getLogger(__name__)


def _make_search_tool(
    *,
    document_id: str,
    user_id: str,
    k: int,
    client: AsyncClient | None,
    accumulated_chunks: list[Document],
):
    """Create a search tool bound to the current retrieval context.

    The tool accumulates retrieved chunks (deduplicated) into the
    provided list so they are available after the agent finishes.

    Args:
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks per search.
        client: Optional Supabase client.
        accumulated_chunks: Mutable list that collected chunks are appended to.

    Returns:
        A LangChain tool that searches the document's embeddings.
    """

    @tool
    async def search_docs(query: str) -> str:
        """Search the document for relevant information.

        Use this tool to find specific facts, concepts, or details
        within the document. Be specific with your search queries.

        Args:
            query: The specific search query (3-8 words recommended)
        """
        chunks = await retrieve_chunks(
            query=query,  # ty: ignore[invalid-argument-type]
            document_id=document_id,  # ty: ignore[invalid-argument-type]
            user_id=user_id,  # ty: ignore[invalid-argument-type]
            k=k,  # ty: ignore[invalid-argument-type]
            client=client,  # ty: ignore[invalid-argument-type]
        )

        if not chunks:
            return "No relevant information found for this query."

        existing_contents = {c.page_content for c in accumulated_chunks}
        new_chunks = [c for c in chunks if c.page_content not in existing_contents]
        accumulated_chunks.extend(new_chunks)

        if not new_chunks:
            return "No additional information found for this query."

        context = "\n\n---\n\n".join(
            f"[Page {c.metadata.get('page', 'unknown')}]: {c.page_content}"
            for c in new_chunks
        )
        return context

    return search_docs


async def _build_agent(
    *,
    query: str,
    document_id: str,
    user_id: str,
    k: int,
    client: AsyncClient | None,
    accumulated_chunks: list[Document],
):
    """Build the agentic RAG agent with a search tool.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user.
        k: Number of chunks per search.
        client: Optional Supabase client.
        accumulated_chunks: Mutable list for chunk accumulation.

    Returns:
        A compiled agent ready for invocation.
    """
    settings = get_settings()
    llm = init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_provider_base_url,
        temperature=0,
    )

    search_tool = _make_search_tool(
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
        accumulated_chunks=accumulated_chunks,
    )

    prompt = await pull_eval_prompt("sanctuary-agentic-rag")
    system_text = prompt.messages[0].prompt.template  # ty: ignore[unresolved-attribute]

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_text,
    )

    return agent


@traceable(metadata={"rag_strategy": "agentic_rag"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the agentic RAG strategy.

    Builds and streams an LLM agent that autonomously searches the
    document via a tool, gathers context, and produces a final answer.
    The agent's own streamed response is forwarded as ``TokenEvent`` objects
    so only a single LLM generation is performed.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve per search.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with final documents, then TokenEvent and CitationsEvent.
    """
    accumulated: list[Document] = []

    agent = await _build_agent(
        query=query,
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
        accumulated_chunks=accumulated,
    )

    settings = get_settings()
    config: dict[str, Any] = {
        "configurable": {"thread_id": f"agentic-rag-{document_id}-{user_id}"},
        "recursion_limit": settings.agentic_rag_max_iterations,
    }

    chunks_yielded = False
    full_answer = ""

    async for msg, metadata in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="messages",
    ):
        if not isinstance(msg, AIMessageChunk):
            continue

        # Skip tool-call chunks (intermediate reasoning)
        if msg.tool_call_chunks:
            continue

        # Yield chunks once, right before the first final-answer token
        if not chunks_yielded:
            chunks_yielded = True
            yield ChunksEvent(
                chunks=[
                    RetrievedChunk(
                        page_content=c.page_content, page=c.metadata.get("page")
                    )
                    for c in accumulated
                ]
            )

        token = msg.content if isinstance(msg.content, str) else str(msg.content)
        if token:
            full_answer += token
            yield TokenEvent(token=token)

    # If the agent never produced a final answer (e.g. no chunks found)
    if not chunks_yielded:
        yield ChunksEvent(chunks=[])

    if not full_answer:
        yield TokenEvent(
            token="The document doesn't contain enough information to answer "
            "this question. Try rephrasing or checking other documents."
        )

    citations = extract_citations(full_answer, accumulated)
    yield CitationsEvent(citations=citations)
