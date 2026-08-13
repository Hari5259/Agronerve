import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch, MagicMock
from core.orchestrator import AgentOrchestrator
from core.translator import language_manager

def test_i18n_prompt_instruction_generation():
    assert "Tamil" in language_manager.get_language_prompt_instruction("ta")
    assert "Hindi" in language_manager.get_language_prompt_instruction("hi")
    assert "Telugu" in language_manager.get_language_prompt_instruction("te")
    assert "Kannada" in language_manager.get_language_prompt_instruction("kn")
    assert language_manager.get_language_prompt_instruction("en") == ""

def test_process_query_propagates_language_prompt():
    orchestrator = AgentOrchestrator()
    
    # Mock _call_ollama to check system_prompt contents
    with patch.object(orchestrator, "_call_ollama") as mock_call:
        mock_call.return_value = "Mocked Response"
        
        # Test query in Tamil
        orchestrator.process_query("pesticide spray dosage", session_id="i18n_test_ta", language="ta")
        
        # Verify language instructions are present in the mock call system_prompt arg
        called_args = mock_call.call_args[1]
        assert "Tamil" in called_args["system_prompt"]
        
        # Test query in Hindi
        orchestrator.process_query("pesticide spray dosage", session_id="i18n_test_hi", language="hi")
        called_args_hi = mock_call.call_args[1]
        assert "Hindi" in called_args_hi["system_prompt"]

def test_fallback_offline_synthesizer_i18n():
    orchestrator = AgentOrchestrator()
    
    # Run fallback synthesizer with non-English language (Tamil)
    fallback_text = orchestrator._fallback_offline_synthesizer(
        active_domains=["disease"],
        chunks=[{"title": "Test Blast", "text": "Blast treatment guidelines."}],
        query="Test query",
        visual_context=None,
        language="ta"
    )
    
    # The title suffix is in Tamil
    assert "ஆலோசனைக்கான" in fallback_text or "அக்ரோநெர்வின்" in fallback_text
