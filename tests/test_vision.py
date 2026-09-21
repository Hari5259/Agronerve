import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image
from core.vision_analyzer import LeafVisionAnalyzer


def test_leaf_vision_synthetic_image_analysis():
    analyzer = LeafVisionAnalyzer()
    # Create a synthetic test RGB image (green background with yellow/brown spot)
    img = Image.new("RGB", (120, 120), color=(40, 140, 40))
    for x in range(30, 90):
        for y in range(30, 90):
            img.putpixel((x, y), (180, 160, 20))  # yellow/chlorotic spot

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = analyzer.analyze_image_bytes(img_bytes, crop_hint="Tomato")
    assert result["status"] in ["success", "uncertain"]
    assert result["crop"] == "Tomato"
    assert "Paddy" not in result["crop"]
    assert result["affected_leaf_area_pct"] > 0
    assert "confidence_pct" in result


def test_leaf_vision_query_crop_inference():
    analyzer = LeafVisionAnalyzer()
    img = Image.new("RGB", (120, 120), color=(50, 150, 50))
    for x in range(30, 90):
        for y in range(30, 90):
            img.putpixel((x, y), (180, 160, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = analyzer.analyze_image_bytes(
        img_bytes, crop_hint="auto", user_query="What is wrong with my tomato leaves?"
    )
    assert result["status"] in ["success", "uncertain"]
    assert result["crop"] == "Tomato"
    assert "Paddy" not in result["crop"]
