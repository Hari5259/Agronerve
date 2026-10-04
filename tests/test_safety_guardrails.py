"""Tests for AgroNerve Safety Guardrails & Agrochemical Validator."""

import pytest
from core.safety_guardrails import SafetyGuardrails, safety_guardrails


def test_banned_chemical_detection():
    unsafe_text = "You should spray Endosulfan 35 EC on your cotton crop to control bollworms."
    audit = safety_guardrails.audit_advisory(unsafe_text)
    assert not audit["is_safe"]
    assert audit["has_warnings"]
    assert any("Endosulfan" in v for v in audit["violations"])


def test_safe_chemical_with_ppe_and_phi():
    safe_text = (
        "Spray Chlorantraniliprole 18.5 SC at 0.4 ml/L for stem borer. "
        "Wear protective gloves and mask during spray. The pre-harvest interval (PHI) is 14 days."
    )
    audit = safety_guardrails.audit_advisory(safe_text)
    assert audit["is_safe"]
    assert len(audit["violations"]) == 0
    assert len(audit["alerts"]) == 0


def test_missing_ppe_warning():
    text_without_ppe = "Spray Imidacloprid 17.8 SL at 0.4 ml/L for aphid control. Observe 21 days PHI."
    audit = safety_guardrails.audit_advisory(text_without_ppe)
    assert any("Personal Protective Equipment" in a for a in audit["alerts"])


def test_sanitize_and_guard_appends_warnings():
    unsafe_text = "Apply Monocrotophos 36 SL to kill insects."
    guarded = safety_guardrails.sanitize_and_guard(unsafe_text)
    assert "REGULATORY CHEMICAL WARNING" in guarded
    assert "Monocrotophos" in guarded
