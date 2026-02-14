import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_api_v4_candidates_basic():
    """Test the V4 specific sniper candidates endpoint."""
    response = client.get("/api/v4/sniper/candidates")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        candidate = data[0]
        assert "ticker" in candidate
        assert "rise_score" in candidate
        assert "ai_prob" in candidate

from unittest.mock import patch

def test_api_backtest_with_version():
    """Test backtest endpoint with a specific model version using mocks."""
    mock_data = {
        "days_ago": 5,
        "model_version": "latest",
        "simulated_date": "2026-02-10",
        "candidate_pool_size": 1,
        "top_picks": [{"ticker": "2330.TW", "actual_return": 0.02}],
        "summary": {"avg_return": 0.02, "win_rate": 100}
    }
    with patch("backend.main.run_time_machine", return_value=mock_data):
        response = client.get("/api/backtest?days=5&version=latest")
        assert response.status_code == 200
        data = response.json()
        assert "top_picks" in data
        assert data["model_version"] == "latest"

def test_api_models_list():
    """Verify the models listing endpoint."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
