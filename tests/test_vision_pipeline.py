import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from core.vision_quality import ImageQualityChecker, quality_checker
from core.vision_analyzer import LeafVisionAnalyzer
from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_quality_checker_valid_image():
    # Green leaf-like image with texture
    img = Image.new("RGB", (200, 200), color=(45, 130, 45))
    for x in range(40, 160, 4):
        for y in range(40, 160, 4):
            img.putpixel((x, y), (170, 150, 30))  # lesion spots for texture

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = quality_checker.evaluate_quality(buf.getvalue())
    assert res["is_usable"] is True
    assert res["foliage_ratio"] > 0.08
    assert res["blur_score"] > 0


def test_quality_checker_too_small():
    img = Image.new("RGB", (32, 32), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = quality_checker.evaluate_quality(buf.getvalue())
    assert res["is_usable"] is False
    assert "too small" in res["reason"]


def test_quality_checker_underexposed():
    img = Image.new("RGB", (100, 100), color=(5, 5, 5))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = quality_checker.evaluate_quality(buf.getvalue())
    assert res["is_usable"] is False
    assert "dark" in res["reason"].lower()


def test_quality_checker_insufficient_foliage():
    # Blue background without green or brown leaf
    img = Image.new("RGB", (100, 100), color=(20, 40, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = quality_checker.evaluate_quality(buf.getvalue())
    assert res["is_usable"] is False
    assert "foliage" in res["reason"].lower() or "leaf" in res["reason"].lower()


def test_leaf_vision_pipeline_diagnosis_and_knowledge():
    analyzer = LeafVisionAnalyzer()
    img = Image.new("RGB", (120, 120), color=(40, 140, 40))
    for x in range(30, 90, 2):
        for y in range(30, 90, 2):
            img.putpixel((x, y), (180, 150, 25))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    result = analyzer.analyze_image_bytes(img_bytes, crop_hint="Tomato")
    assert result["status"] in ["success", "uncertain"]
    if result["status"] == "success":
        assert result["crop"] == "Tomato"
        assert result["confidence_pct"] > 0
        assert "verified_protocol" in result
        assert "knowledge" in result


def test_api_scan_leaf_endpoint_schema():
    img = Image.new("RGB", (120, 120), color=(40, 140, 40))
    for x in range(30, 90, 2):
        for y in range(30, 90, 2):
            img.putpixel((x, y), (180, 150, 25))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    response = client.post("/api/scan-leaf", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "crop" in data
    assert "disease" in data
    assert "quality_metrics" in data
