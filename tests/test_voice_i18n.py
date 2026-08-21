import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.voice_engine import VoiceEngine
from core.translator import LanguageManager


def test_voice_markdown_cleaning():
    markdown = "### Disease Title\n* **Pesticide**: Mancozeb 75 WP @ 2.5 g/L.\n> Caution: Wear PPE."
    clean = VoiceEngine.clean_text_for_speech(markdown)
    assert "#" not in clean
    assert "*" not in clean
    assert "Mancozeb 75 WP" in clean


def test_voice_technical_unit_expansion():
    # Test chemical dosage unit expansion
    clean_dosage = VoiceEngine.clean_text_for_speech("Use 2 ml/L or 3 g/L of Mancozeb")
    assert "milliliters per liter" in clean_dosage
    assert "grams per liter" in clean_dosage

    # Test soil telemetry unit expansion
    clean_telemetry = VoiceEngine.clean_text_for_speech("Soil moisture is 34.5% VWC, air humidity is 60% RH at 28°C")
    assert "percent Volumetric Water Content" in clean_telemetry
    assert "percent Relative Humidity" in clean_telemetry
    assert "28 degrees Celsius" in clean_telemetry

    # Test regulatory/safety abbreviation expansion
    clean_safety = VoiceEngine.clean_text_for_speech("Follow PPE and PHI rules")
    assert "P P E" in clean_safety
    assert "P H I" in clean_safety


def test_voice_html_script_generation():
    script = VoiceEngine.generate_html5_audio_speech_script("Hello farmers", "en")
    assert "speechRate" in script
    assert "document.getElementById(\"speechRate\")" in script
    assert "type=\"range\"" in script


def test_language_manager_translations():
    mgr = LanguageManager()
    assert mgr.get_text("app_title", "en") != ""
    assert mgr.get_text("app_title", "ta") != ""
    assert mgr.get_text("app_title", "hi") != ""
