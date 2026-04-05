"""Tests for the FastAPI application entry point."""

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app.

    Returns:
        TestClient: A test client instance for making HTTP requests.
    """
    return TestClient(app)


def test_read_root(client):
    """Test the root endpoint returns expected response.

    Args:
        client: TestClient instance for making requests.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sanctuary AI"}


def test_health(client):
    """Test the health endpoint returns healthy status.

    Args:
        client: TestClient instance for making requests.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
