import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check_api():
    """Verify the health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'ok'
    assert 'db' in data

def test_api_stocks_list():
    """Verify the stock list endpoint."""
    response = client.get("/api/stocks")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_top_picks():
    """Verify the top picks endpoint."""
    response = client.get("/api/top_picks?sort=score")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_sync_status():
    """Verify the sync status endpoint."""
    response = client.get("/api/sync/status")
    assert response.status_code == 200
    data = response.json()
    assert 'is_syncing' in data
