import re

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import EMBEDDING_DIMENSIONS, settings
from src.db.client import get_supabase_client
from src.schemas.chat import Citation


async def retrieve_chunks(
    query: str,
    document_id: str,
    user_id: str,
    k: int = 5,
) -> list[Document]:
    """Retrieve relevant chunks from Supabase vector store.

    Args:
        query: The user's question.
        document_id: UUID of the document to search within.
        user_id: UUID of the user (for authorization).
        k: Number of chunks to retrieve.

    Returns:
        List of LangChain Document objects with page metadata.
    """
    # Generate embedding for the query
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=f"models/{settings.embedding_model}",
        google_api_key=settings.gemini_api_key,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    query_embedding = embeddings_model.embed_query(query)

    # Query Supabase for similar chunks with filters
    client = await get_supabase_client()

    # Use RPC function for similarity search with filter parameter
    result = await client.rpc(
        "match_document_embeddings",
        {
            "query_embedding": query_embedding,
            "filter": {
                "document_id": document_id,
                "user_id": user_id,
            },
            "match_count": k,
        },
    ).execute()

    if not result.data:
        print("No results found")
        return []

    # Convert to LangChain Document format
    chunks: list[Document] = []
    for row in result.data:  # ty: ignore[not-iterable]
        chunks.append(
            Document(
                page_content=row["content"],  # ty: ignore[invalid-argument-type, not-subscriptable]
                metadata={"page": row["metadata"].get("page", 0)},  # ty: ignore[invalid-argument-type, not-subscriptable]
            )
        )

    return chunks


def extract_citations(answer: str) -> list[Citation]:
    """Extract page citations from the answer text.

    Looks for patterns like [p. X], [page X], (p. X), etc.

    Args:
        answer: The LLM-generated answer text.

    Returns:
        List of Citation objects with page numbers.
    """
    citations = []
    # Match patterns like [p. 12], [page 12], (p. 12)
    pattern = r"\[?(?:p\.?|page)\s*(\d+)\]?"
    matches = re.findall(pattern, answer, re.IGNORECASE)

    for page_num in matches:
        citations.append(Citation(page=int(page_num)))

    return citations


async def generate_answer_with_citations(
    query: str,
    chunks: list[Document],
) -> tuple[str, list[Citation]]:
    """Generate an answer based on retrieved chunks with citations.

    Args:
        query: The user's question.
        chunks: List of retrieved Document objects.

    Returns:
        Tuple of (answer_text, citations_list).
    """
    if not chunks:
        return "I don't have enough information to answer that question.", []

    # Build context from chunks with page references
    context_parts = []
    for chunk in chunks:
        page = chunk.metadata.get("page", "unknown")
        context_parts.append(f"[Page {page}]: {chunk.page_content}")

    context = "\n\n".join(context_parts)

    # Create prompt that constrains answer to retrieved content
    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided document content. 
Do not use any outside knowledge. If the answer cannot be found in the content, say so.

Always cite your sources using [p. X] format where X is the page number.

Question: {query}

Context:
{context}

Answer:"""

    # Initialize LLM using LangChain's init_chat_model
    llm = init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_provider_base_url,
        temperature=0,
        streaming=False,
    )

    response = llm.invoke(prompt)
    answer_content = response.content if hasattr(response, "content") else str(response)
    answer: str = (
        answer_content if isinstance(answer_content, str) else str(answer_content)
    )

    # Extract citations (answer is str from above)
    citations = extract_citations(answer)

    return answer, citations
