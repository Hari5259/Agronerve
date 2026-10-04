"""Tests for AgroNerve Response Quality, Sensor Alerts & Consultation Summaries."""

import pytest
from core.orchestrator import AgentOrchestrator
from core.sensor_telemetry import sensor_manager
from core.session_manager import session_manager


@pytest.fixture
def orchestrator():
    return AgentOrchestrator()


def test_process_query_structured_response(orchestrator):
    res = orchestrator.process_query(
        query="What is the dosage of chlorantraniliprole for stem borer in paddy?",
        session_id="test_quality_sess",
        language="en",
    )
    assert res["domain"] in ["pesticide", "disease"]
    assert "safety_audit" in res
    assert "response" in res
    assert len(res["response"]) > 20
    # Check that safety guard / badges are present
    assert "`" in res["response"]


def test_proactive_sensor_alerts():
    # Set dry soil condition
    sensor_manager.set_manual_telemetry(
        soil_moisture=15.0,
        ambient_temp=30.0,
        humidity=60.0,
        rain=False,
    )
    alerts = sensor_manager.get_proactive_field_alerts()
    assert len(alerts) >= 1
    assert any(a["category"] == "IRRIGATION" for a in alerts)

    duration = sensor_manager.calculate_irrigation_duration_minutes(target_moisture=60.0)
    assert duration > 0.0


def test_session_summary_generation():
    sess_id = "test_summary_sess"
    sess = session_manager.get_or_create_session(sess_id)
    sess.add_message("user", "My tomato leaves have yellow spots")
    sess.add_message("assistant", "It appears to be Early Blight.")
    sess.current_crop = "Tomato"
    sess.current_diagnosed_disease = "Early Blight"

    summary = session_manager.get_session_summary(sess_id)
    assert summary["session_id"] == sess_id
    assert summary["target_crop"] == "Tomato"
    assert summary["diagnosed_disease"] == "Early Blight"
    assert summary["total_turns"] == 1
