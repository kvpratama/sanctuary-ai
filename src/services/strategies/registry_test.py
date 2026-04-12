import pytest

from src.services.strategies.registry import get_strategy


def test_get_strategy_returns_naive_rag():
    """get_strategy('naive_rag') returns the naive_rag execute function."""
    strategy = get_strategy("naive_rag")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.naive_rag"


def test_get_strategy_returns_query_rewrite():
    """get_strategy('query_rewrite') returns the query_rewrite execute function."""
    strategy = get_strategy("query_rewrite")
    assert callable(strategy)
    assert strategy.__module__ == "src.services.strategies.query_rewrite"


def test_get_strategy_unknown_raises_value_error():
    """get_strategy raises ValueError for unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown RAG strategy: 'nonexistent'"):
        get_strategy("nonexistent")
