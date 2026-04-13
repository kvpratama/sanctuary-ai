"""Strategy registry for RAG pipeline variants."""

from collections.abc import AsyncGenerator, Callable

from src.schemas.chat import StreamEvent

# Type alias for a strategy execute function
StrategyFn = Callable[
    ...,
    AsyncGenerator[StreamEvent, None],
]


def get_strategy(name: str) -> StrategyFn:
    """Return the execute function for the given strategy name.

    Args:
        name: Strategy identifier (e.g. 'naive_rag', 'query_rewrite').

    Returns:
        The async generator function for the named strategy.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    from src.services.strategies.multi_query import execute as multi_query_execute
    from src.services.strategies.naive_rag import execute as naive_rag_execute
    from src.services.strategies.query_rewrite import execute as query_rewrite_execute
    from src.services.strategies.self_correcting import (
        execute as self_correcting_execute,
    )

    registry: dict[str, StrategyFn] = {
        "naive_rag": naive_rag_execute,
        "query_rewrite": query_rewrite_execute,
        "multi_query": multi_query_execute,
        "self_correcting": self_correcting_execute,
    }

    if name not in registry:
        raise ValueError(f"Unknown RAG strategy: '{name}'")

    return registry[name]
