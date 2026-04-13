"""Self-correcting RAG strategy — LangGraph-based retrieve, grade, optionally rewrite.

Uses a StateGraph to:
1. Retrieve documents
2. Grade their relevance
3. Optionally rewrite the query and re-retrieve (max 1 retry)
4. Generate the final answer
"""

import logging
from collections.abc import AsyncGenerator
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field

from src.config import get_settings
from src.prompts.manager import pull_eval_prompt
from src.schemas.chat import ChunksEvent, RetrievedChunk, StreamEvent
from src.services.strategies.core import retrieve_chunks, stream_answer_with_citations
from supabase import AsyncClient

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
MIN_RELEVANT_CHUNKS = 3


class RelevanceGrade(BaseModel):
    """Structured output for a single chunk relevance judgment."""

    relevant: bool = Field(
        description="Whether the document is relevant to the question."
    )


class SelfCorrectingState(TypedDict, total=False):
    """State for the self-correcting RAG graph.

    Attributes:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user.
        k: Number of chunks to retrieve.
        client: Optional Supabase client.
        retrieval_query: The query used for retrieval (may be rewritten).
        chunks: Retrieved document chunks (filtered to relevant only after grading).
        retry_count: Number of retrieval retries performed.
        answer: The generated answer text.
    """

    query: str
    document_id: str
    user_id: str
    k: int
    client: AsyncClient | None
    retrieval_queries: list[str]
    chunks: list[Document]
    retry_count: int
    answer: str


async def retrieve_node(state: SelfCorrectingState) -> dict:
    """Retrieve chunks using the current retrieval query.

    Uses ``retrieval_query`` if set, otherwise falls back to the original ``query``.

    Args:
        state: The current graph state.

    Returns:
        Dict with updated ``chunks``.
    """
    queries = state.get("retrieval_queries", [])
    search_query = queries[-1] if queries else state["query"]

    chunks = await retrieve_chunks(
        query=search_query,  # ty: ignore[invalid-argument-type]
        document_id=state["document_id"],  # ty: ignore[invalid-argument-type]
        user_id=state["user_id"],  # ty: ignore[invalid-argument-type]
        k=state.get("k", 5),  # ty: ignore[invalid-argument-type]
        client=state.get("client"),  # ty: ignore[invalid-argument-type]
    )

    return {"chunks": chunks}


async def grade_relevance_node(state: SelfCorrectingState) -> dict:
    """Grade whether the retrieved chunks are relevant to the query.

    Sends each chunk to an LLM for a structured relevance judgment and
    filters out irrelevant ones. ``chunks`` in the returned dict contains
    only the documents that passed grading, so downstream nodes never see
    noise. On exception, ``chunks`` is cleared to trigger a rewrite if retries
    remain. If at the retry limit, existing chunks are preserved to avoid
    generating an answer with zero context.

    Args:
        state: The current graph state.

    Returns:
        Dict with updated ``chunks`` (relevant only).
    """
    chunks = state.get("chunks", [])
    if not chunks:
        return {"chunks": []}

    try:
        settings = get_settings()
        llm = init_chat_model(
            model=settings.grading_llm_model,
            model_provider=settings.llm_provider,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_provider_base_url,
            temperature=0,
        )

        structured_llm = llm.with_structured_output(RelevanceGrade)
        prompt = await pull_eval_prompt("sanctuary-retrieval-grading")
        chain = prompt | structured_llm

        relevant_chunks: list[Document] = []
        for chunk in chunks:
            result = await chain.ainvoke(
                {
                    "query": state["query"],
                    "document": chunk.page_content,
                }
            )
            if isinstance(result, RelevanceGrade) and result.relevant:
                relevant_chunks.append(chunk)

        logger.info(
            "Relevance grading: %d/%d relevant (threshold: %d)",
            len(relevant_chunks),
            len(chunks),
            MIN_RELEVANT_CHUNKS,
        )
        return {"chunks": relevant_chunks}
    except Exception:
        if state.get("retry_count", 0) < MAX_RETRIES:
            logger.warning(
                "Relevance grading failed, clearing chunks to trigger rewrite (%d/%d)",
                state.get("retry_count", 0),
                MAX_RETRIES,
                exc_info=True,
            )
            return {"chunks": []}

        logger.warning(
            "Relevance grading failed on final attempt, preserving raw chunks",
            exc_info=True,
        )
        return {}


async def rewrite_node(state: SelfCorrectingState) -> dict:
    """Rewrite the retrieval query using an LLM and increment retry count.

    Falls back to the original query if the LLM call fails.

    Args:
        state: The current graph state.

    Returns:
        Dict with updated ``retrieval_query`` and incremented ``retry_count``.
    """
    try:
        settings = get_settings()
        llm = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=settings.llm_provider_base_url,
            temperature=0,
        )

        prompt = await pull_eval_prompt("sanctuary-query-rewrite")
        chain = prompt | llm

        query_input = state["query"]
        queries = state.get("retrieval_queries", [])
        if queries:
            query_input += (
                "\n\nPast rewrites you already tried (do not repeat these):\n"
            )
            query_input += "\n".join(f"- {q}" for q in queries)

        response = await chain.ainvoke({"query": query_input})

        content = response.content if hasattr(response, "content") else ""
        rewritten = content.strip() if isinstance(content, str) else ""
        if not rewritten:
            logger.warning(
                "Self-correcting rewrite returned empty, falling back to original query"
            )
            rewritten = state["query"]

        logger.info("Self-correcting rewrite: '%s' -> '%s'", state["query"], rewritten)
    except Exception:
        logger.warning(
            "Self-correcting rewrite failed, falling back to original query",
            exc_info=True,
        )
        rewritten = state["query"]

    queries = state.get("retrieval_queries", [])
    new_queries = list(queries)
    if rewritten and (not queries or rewritten != queries[-1]):
        new_queries.append(rewritten)

    return {
        "retrieval_queries": new_queries,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def should_retry(state: SelfCorrectingState) -> Literal["rewrite"] | str:
    """Decide whether to rewrite and re-retrieve or proceed to generate.

    Relevance is inferred from chunk count: fewer than ``MIN_RELEVANT_CHUNKS``
    means grading did not find enough signal.

    Args:
        state: The current graph state.

    Returns:
        ``"rewrite"`` if fewer than ``MIN_RELEVANT_CHUNKS`` chunks remain
        and retries remain, otherwise ``END``.
    """
    has_enough = len(state.get("chunks", [])) >= MIN_RELEVANT_CHUNKS
    if not has_enough and state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"
    return END


def build_graph() -> StateGraph:
    """Build and compile the self-correcting RAG graph.

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(SelfCorrectingState)  # ty: ignore[invalid-argument-type]

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_relevance_node)
    graph.add_node("rewrite", rewrite_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", should_retry, ["rewrite", END])
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()  # ty:ignore[invalid-return-type]


@traceable(metadata={"rag_strategy": "self_correcting"})
async def execute(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
    *,
    client: AsyncClient | None = None,
) -> AsyncGenerator[StreamEvent, None]:
    """Run the self-correcting RAG strategy.

    Builds and invokes the LangGraph graph for retrieval with relevance
    grading and optional query rewriting, then streams the answer.

    Args:
        query: The user's original question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.
        client: Optional Supabase client to reuse.

    Yields:
        ChunksEvent with final documents, then TokenEvent and CitationsEvent.
    """
    graph = build_graph()

    initial_state: SelfCorrectingState = {
        "query": query,
        "document_id": document_id,
        "user_id": user_id,
        "k": k,
        "client": client,
        "retrieval_queries": [query],
        "chunks": [],
        "retry_count": 0,
        "answer": "",
    }

    final_state = await graph.ainvoke(initial_state)  # ty:ignore[unresolved-attribute]
    chunks = final_state.get("chunks", [])

    yield ChunksEvent(
        chunks=[
            RetrievedChunk(page_content=c.page_content, page=c.metadata.get("page"))
            for c in chunks
        ]
    )

    async for event in stream_answer_with_citations(query=query, chunks=chunks):  # ty:ignore[invalid-argument-type]
        yield event
