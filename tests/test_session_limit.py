import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.session_manager import SessionManager, ChatSession


def test_session_cleanup():
    manager = SessionManager()
    s1 = manager.get_or_create_session("sess_1")
    s2 = manager.get_or_create_session("sess_2")
    
    # Manually backdate s1's last active time
    s1.last_active = time.time() - 5000
    
    # Prune sessions older than 3600 seconds
    pruned = manager.cleanup_expired_sessions(3600)
    assert pruned == 1
    assert "sess_1" not in manager.sessions
    assert "sess_2" in manager.sessions


def test_chat_session_message_limiting():
    session = ChatSession("limit_test")
    # Add 60 messages
    for i in range(60):
        session.add_message("user", f"message {i}", max_messages=50)
        
    assert len(session.messages) == 50
    assert session.messages[0]["content"] == "message 10"
    assert session.messages[-1]["content"] == "message 59"
