import pytest

from src.services.strategies.registry import get_strategy


def test_get_strategy_returns_naive_rag() -> None:
    """get_strategy('naive_rag') returns the naive_rag execute function."""
    strategy = get_strategy("naive_rag")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.naive_rag"


def test_get_strategy_returns_query_rewrite() -> None:
    """get_strategy('query_rewrite') returns the query_rewrite execute function."""
    strategy = get_strategy("query_rewrite")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.query_rewrite"


def test_get_strategy_returns_multi_query() -> None:
    """get_strategy('multi_query') returns the multi_query execute function."""
    strategy = get_strategy("multi_query")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.multi_query"


def test_get_strategy_returns_self_correcting() -> None:
    """get_strategy('self_correcting') returns the self_correcting execute function."""
    strategy = get_strategy("self_correcting")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.self_correcting"


def test_get_strategy_unknown_raises_value_error() -> None:
    """get_strategy raises ValueError for unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown RAG strategy: 'nonexistent'"):
        get_strategy("nonexistent")
