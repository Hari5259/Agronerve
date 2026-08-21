import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.sensor_telemetry import SensorTelemetryManager


def test_sensor_telemetry_reading():
    manager = SensorTelemetryManager()
    t = manager.get_telemetry()
    assert "soil" in t
    assert "environment" in t
    assert "agronomic_advisories" in t
    assert "moisture_vwc_pct" in t["soil"]


def test_sensor_critical_dry_alert():
    manager = SensorTelemetryManager()
    manager.set_manual_telemetry(
        soil_moisture=18.0, ambient_temp=30.0, humidity=50.0, rain=False
    )
    t = manager.get_telemetry()
    assert "CRITICAL WATER STRESS" in t["agronomic_advisories"]["irrigation_action"]


def test_sensor_input_clamping():
    manager = SensorTelemetryManager()
    
    # Test values above maximum physical bounds
    manager.set_manual_telemetry(soil_moisture=150.0, ambient_temp=100.0, humidity=120.0, rain=False)
    t1 = manager.get_telemetry()
    assert t1["soil"]["moisture_vwc_pct"] == 100.0
    assert t1["environment"]["ambient_temp_celsius"] == 60.0
    assert t1["environment"]["relative_humidity_pct"] == 100.0

    # Test values below minimum physical bounds
    manager.set_manual_telemetry(soil_moisture=-20.0, ambient_temp=-80.0, humidity=-10.0, rain=False)
    t2 = manager.get_telemetry()
    assert t2["soil"]["moisture_vwc_pct"] == 0.0
    assert t2["environment"]["ambient_temp_celsius"] == -50.0
    assert t2["environment"]["relative_humidity_pct"] == 0.0


def test_sensor_frost_warning():
    manager = SensorTelemetryManager()
    manager.set_manual_telemetry(soil_moisture=40.0, ambient_temp=2.0, humidity=60.0, rain=False)
    t = manager.get_telemetry()
    assert "FROST WARNING" in t["agronomic_advisories"]["pathogen_infection_risk"]


def test_sensor_spray_warnings():
    manager = SensorTelemetryManager()
    
    # Test low temperature spray window advisory
    manager.set_manual_telemetry(soil_moisture=40.0, ambient_temp=8.0, humidity=60.0, rain=False)
    t1 = manager.get_telemetry()
    assert "Low temperature reduces systemic chemical uptake" in t1["agronomic_advisories"]["chemical_spray_window"]

    # Test high humidity spray window advisory
    manager.set_manual_telemetry(soil_moisture=40.0, ambient_temp=22.0, humidity=95.0, rain=False)
    t2 = manager.get_telemetry()
    assert "High humidity delays spray drying" in t2["agronomic_advisories"]["chemical_spray_window"]

