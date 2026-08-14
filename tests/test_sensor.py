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
