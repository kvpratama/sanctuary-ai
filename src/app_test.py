import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_read_root(client):
    """Test the root endpoint returns expected response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bookified API"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
