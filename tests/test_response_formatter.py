"""Tests for AgroNerve Response Formatter."""

import pytest
from core.response_formatter import ResponseFormatter, response_formatter


def test_format_structured_advisory_basic():
    raw = "Apply Mancozeb 75 WP at 2.5 g/L.\n\nEnsure good spray coverage."
    result = response_formatter.format_structured_advisory(
        raw_text=raw,
        domain="disease",
        crop="tomato",
    )
    assert "`🔬 Crop Pathology Advisory`" in result
    assert "`🌱 Target Crop: Tomato`" in result
    assert "Apply Mancozeb 75 WP at 2.5 g/L." in result


def test_format_structured_advisory_with_safety_alerts():
    raw = "Spray pesticide in field."
    alerts = ["Wear protective mask and nitrile gloves."]
    result = response_formatter.format_structured_advisory(
        raw_text=raw,
        domain="pesticide",
        crop="cotton",
        safety_alerts=alerts,
    )
    assert "CRITICAL SAFETY ADVISORY" in result
    assert "Wear protective mask and nitrile gloves." in result


def test_create_dosage_table():
    treatments = [
        {"name": "Coragen 18.5 SC", "dosage": "0.4 ml/L", "method": "Foliar Spray", "phi_days": 14},
        {"name": "Neem Oil 10000 ppm", "dosage": "4.0 ml/L", "method": "Foliar Spray", "phi_days": 0},
    ]
    table = response_formatter.create_dosage_table(treatments)
    assert "| Chemical / Bio-Agent | Dilution Rate |" in table
    assert "| **Coragen 18.5 SC** | `0.4 ml/L` | Foliar Spray | 14 days |" in table
    assert "| **Neem Oil 10000 ppm** | `4.0 ml/L` | Foliar Spray | 0 days |" in table


def test_format_water_budget_card():
    card = response_formatter.format_water_budget_card(
        crop="Tomato",
        stage="flowering",
        water_req_mm=5.2,
        interval_days=4,
        soil_type="Sandy Loam",
    )
    assert "Irrigation Water Budget Summary" in card
    assert "5.2 mm/day" in card
    assert "Every `4` days" in card
