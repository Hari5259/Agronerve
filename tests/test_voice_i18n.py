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


def test_language_manager_translations():
    mgr = LanguageManager()
    assert mgr.get_text("app_title", "en") != ""
    assert mgr.get_text("app_title", "ta") != ""
    assert mgr.get_text("app_title", "hi") != ""
