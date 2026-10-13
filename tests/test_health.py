"""Test health and service status endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["server"] == "online"
    assert "python_version" in data
    assert "ollama" in data
    assert "stt" in data
    assert "emotion" in data
    assert "tts" in data
    assert "ready" in data["ollama"]
