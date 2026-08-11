import sys
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from PIL import Image
from core.orchestrator import AgentOrchestrator
from core.session_manager import session_manager

def test_multimodal_turn_and_follow_up_chat():
    orchestrator = AgentOrchestrator()
    session_id = "test_user_session_42"
    
    # 1. Simulate user sending a tomato leaf image with early blight symptoms
    img = Image.new("RGB", (80, 80), color=(50, 140, 50))
    for x in range(25, 55):
        for y in range(25, 55):
            img.putpixel((x, y), (180, 160, 20)) # yellow/brown patch
            
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Step 1: Multimodal image turn
    turn1_res = orchestrator.process_multimodal_turn(
        image_bytes=img_bytes,
        user_text="What is this disease on my tomato crop?",
        session_id=session_id
    )
    assert turn1_res["domain"] == "disease"
    assert "response" in turn1_res
    
    # Check session memory state
    session = session_manager.get_or_create_session(session_id)
    assert session.current_crop is not None
    assert len(session.messages) == 2

    # Step 2: Follow-up question without repeating the crop name
    turn2_res = orchestrator.process_query(
        query="What is the exact chemical spray dosage for this?",
        session_id=session_id
    )
    assert turn2_res["domain"] == "pesticide"
    assert len(session.messages) == 4
    assert len(turn2_res["response"]) > 0

    # Step 3: Second follow-up regarding weather forecast
    turn3_res = orchestrator.process_query(
        query="Should I spray if it rains tomorrow?",
        session_id=session_id
    )
    assert turn3_res["domain"] == "weather"
    assert "weather" in turn3_res["active_domains"]
    assert len(session.messages) == 6
