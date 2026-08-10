import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["knowledge_base_chunks"] > 0

def test_route_classification_endpoint():
    response = client.post("/api/route", json={"query": "How to treat yellow rust in wheat?"})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "disease"

def test_advisory_query_endpoint():
    response = client.post("/api/query", json={"query": "What is the dosage of Mancozeb for tomato early blight?"})
    assert response.status_code == 200
    data = response.json()
    assert "domain" in data
    assert "response" in data
    assert len(data["response"]) > 0

def test_domains_list_endpoint():
    response = client.get("/api/domains")
    assert response.status_code == 200
    data = response.json()
    assert "domains" in data
    assert len(data["domains"]) >= 4

def test_knowledge_endpoint():
    response = client.get("/api/knowledge?domain=disease")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] > 0
