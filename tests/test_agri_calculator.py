"""Tests for AgroNerve AgriCalculator Engine."""

import pytest
from core.agri_calculator import AgriCalculator, agri_calculator


def test_calculate_spray_dosage():
    res = agri_calculator.calculate_spray_dosage(
        area_acres=2.0,
        dose_per_liter=0.4,
        unit="ml",
        tank_capacity_liters=16.0,
        water_volume_per_acre=200.0,
    )
    assert res["area_acres"] == 2.0
    assert res["total_water_volume_liters"] == 400.0
    assert res["total_chemical_required"] == 160.0  # 400 * 0.4
    assert res["chemical_per_tank"] == 6.4          # 16 * 0.4
    assert res["total_knapsack_tanks"] == 25        # 400 / 16


def test_calculate_water_requirement():
    res = agri_calculator.calculate_water_requirement(
        crop="tomato",
        stage="mid",
        area_acres=1.0,
        eto_mm_day=4.0,
        irrigation_efficiency=0.85,
    )
    assert res["crop"] == "Tomato"
    assert res["stage"] == "mid"
    assert res["etc_mm_day"] == 4.6  # 4.0 * 1.15 = 4.6
    assert res["daily_gross_liters"] > 0
    assert res["weekly_gross_liters"] == res["daily_gross_liters"] * 7


def test_calculate_fertilizer_npk_sources():
    # 40 kg N, 20 kg P2O5, 20 kg K2O
    res = agri_calculator.calculate_fertilizer_npk_sources(
        n_kg=40.0,
        p_kg=20.0,
        k_kg=20.0,
    )
    # DAP required: 20 / 0.46 = ~43.5 kg
    assert res["dap_kg"] == 43.5
    # N from DAP: 43.478 * 0.18 = ~7.82 kg
    # Remaining N: 40 - 7.82 = 32.18 kg
    # Urea required: 32.18 / 0.46 = ~69.95 -> 70.0 kg
    assert 69.0 <= res["urea_kg"] <= 71.0
    # MOP required: 20 / 0.60 = ~33.3 kg
    assert res["mop_kg"] == 33.3
