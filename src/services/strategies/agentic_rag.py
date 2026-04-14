"""Agentic RAG strategy -- LLM agent with search tool.

Uses LangChain's create_agent() with a search_docs tool backed by
retrieve_chunks. The agent autonomously searches, gathers context,
and decides when to answer.  The agent's own streamed answer is used
directly — no second LLM call is needed.

Iteration control is handled by ToolCallLimitMiddleware, which gracefully
stops the agent when limits are reached rather than abruptly cutting off
execution like recursion_limit.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable

from src.config import get_settings
from src.prompts.manager import FALLBACK_PROMPTS, pull_eval_prompt
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


def _extract_system_text(prompt: ChatPromptTemplate) -> str | None:
    """Extract the system message template text from a ``ChatPromptTemplate``.

    Iterates over ``prompt.messages`` looking for the first
    ``SystemMessagePromptTemplate`` and returns its ``.prompt.template``
    attribute.  Also handles raw ``SystemMessage`` objects (which store
    text in ``.content``).

    Args:
        prompt: The prompt template to inspect.

    Returns:
        The system message text, or ``None`` if no system message is found.
    """
    from langchain_core.messages import SystemMessage

    for message in prompt.messages:
        if isinstance(message, SystemMessagePromptTemplate):
            template = getattr(message.prompt, "template", None)
            if template:
                return template
        elif isinstance(message, SystemMessage):
            content = message.content
            if isinstance(content, str):
                return content
    return None


def _make_search_tool(
    *,
    document_id: str,
    user_id: str,
    k: int,
    client: AsyncClient | None,
    accumulated_chunks: list[Document],
) -> Any:
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
    async def search_docs(query: str = "") -> str:
        """Search the document for relevant information.

        Use this tool to find specific facts, concepts, or details
        within the document. Be specific with your search queries.

        Args:
            query: The specific search query (3-8 words recommended)
        """

        if not query.strip():
            return (
                "Please provide a specific search query to find relevant information."
            )

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
    max_tool_calls: int,
) -> CompiledStateGraph:
    """Build the agentic RAG agent with a search tool.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user.
        k: Number of chunks per search.
        client: Optional Supabase client.
        accumulated_chunks: Mutable list for chunk accumulation.
        max_tool_calls: Maximum number of tool calls allowed per run.

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
        streaming=True,
    )

    search_tool = _make_search_tool(
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
        accumulated_chunks=accumulated_chunks,
    )

    prompt = await pull_eval_prompt("sanctuary-agentic-rag")
    system_text = _extract_system_text(prompt)

    if system_text is None:
        fallback = FALLBACK_PROMPTS.get("sanctuary-agentic-rag")
        if fallback:
            logger.warning(
                "Pulled prompt 'sanctuary-agentic-rag' has no system message. "
                "Using fallback prompt.",
            )
            system_text = _extract_system_text(fallback)

        if system_text is None:
            raise ValueError(
                "No system message found in pulled prompt or fallback for "
                "'sanctuary-agentic-rag'. The prompt must contain a system message."
            )

    # Use ToolCallLimitMiddleware for graceful iteration control
    iteration_limiter = ToolCallLimitMiddleware(
        run_limit=max_tool_calls,
        exit_behavior="continue",
    )

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=system_text,
        middleware=[iteration_limiter],
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

    Iteration control is managed by ToolCallLimitMiddleware which gracefully
    terminates the agent when the limit is reached, allowing it to produce
    a final answer based on gathered context.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve per search.
        client: Optional Supabase client to reuse.

    Yields:
        TokenEvent for streamed answer tokens, then ChunksEvent with final
        documents, and CitationsEvent.
    """
    settings = get_settings()
    accumulated: list[Document] = []

    agent = await _build_agent(
        query=query,
        document_id=document_id,
        user_id=user_id,
        k=k,
        client=client,
        accumulated_chunks=accumulated,
        max_tool_calls=settings.max_tool_calls,
    )

    config: RunnableConfig = {
        "configurable": {"thread_id": f"agentic-rag-{document_id}-{user_id}"},
    }

    full_answer = ""

    async for msg, metadata in agent.astream(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(msg, (AIMessageChunk, AIMessage)) and not (
            isinstance(msg, AIMessageChunk) and msg.tool_call_chunks
        ):
            token = msg.content if isinstance(msg.content, str) else str(msg.content)
            if token:
                full_answer += token
                yield TokenEvent(token=token)

    # Yield ChunksEvent after streaming completes, so all accumulated
    # chunks are available and the event ordering is deterministic.
    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in accumulated
        ]
    )

    if not full_answer:
        yield TokenEvent(
            token="The document doesn't contain enough information to answer "
            "this question. Try rephrasing or checking other documents."
        )

    citations = extract_citations(full_answer, accumulated)
    yield CitationsEvent(citations=citations)
