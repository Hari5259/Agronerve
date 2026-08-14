import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image
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
    response = client.post(
        "/api/route", json={"query": "How to treat yellow rust in wheat?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("primary_domain") == "disease"


def test_advisory_query_endpoint():
    response = client.post(
        "/api/query",
        json={"query": "What is the dosage of Mancozeb for tomato early blight?"},
    )
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


def test_sensor_telemetry_endpoint():
    response = client.get("/api/sensor/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "soil" in data
    assert "environment" in data


def test_voice_clean_endpoint():
    response = client.post(
        "/api/voice/clean", json={"text": "### Treatment\n* Spray Tricyclazole 75 WP."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "#" not in data["cleaned_speech_text"]


def test_scan_leaf_endpoint():
    img = Image.new("RGB", (60, 60), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("leaf.jpg", buf, "image/jpeg")}
    response = client.post("/api/scan-leaf", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "predicted_disease" in data


def test_update_sensor_telemetry_endpoint():
    response = client.post(
        "/api/sensor/telemetry",
        json={
            "soil_moisture": 22.5,
            "ambient_temp": 32.0,
            "humidity": 45.0,
            "rain": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert abs(data["telemetry"]["soil"]["moisture_vwc_pct"] - 22.5) <= 0.5
    assert (
        "CRITICAL WATER STRESS"
        in data["telemetry"]["agronomic_advisories"]["irrigation_action"]
    )
