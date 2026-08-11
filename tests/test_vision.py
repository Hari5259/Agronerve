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
    img = Image.new("RGB", (100, 100), color=(40, 140, 40))
    for x in range(30, 70):
        for y in range(30, 70):
            img.putpixel((x, y), (180, 160, 20)) # yellow/chlorotic spot
            
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    result = analyzer.analyze_image_bytes(img_bytes, crop_hint="Tomato")
    assert result["status"] == "success"
    assert result["affected_leaf_area_pct"] > 0
    assert "confidence_pct" in result
    assert "predicted_disease" in result
