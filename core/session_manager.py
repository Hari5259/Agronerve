import time
from typing import Dict, Any, List, Optional

class ChatSession:
    """Represents a multi-turn agricultural advisory conversation with context memory."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.messages: List[Dict[str, Any]] = []
        # Persistent agronomic context across turns
        self.current_crop: Optional[str] = None
        self.current_diagnosed_disease: Optional[str] = None
        self.last_visual_diagnosis: Optional[Dict[str, Any]] = None
        self.active_domains: List[str] = ["general"]

    def add_message(self, role: str, content: str, meta: Optional[Dict[str, Any]] = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "meta": meta or {}
        })
        self.last_active = time.time()

    def update_visual_context(self, vision_result: Dict[str, Any]):
        self.last_visual_diagnosis = vision_result
        if "crop" in vision_result:
            self.current_crop = vision_result["crop"]
        if "predicted_disease" in vision_result:
            self.current_diagnosed_disease = vision_result["predicted_disease"]

    def get_conversation_history_prompt(self, max_turns: int = 4) -> str:
        """Formats the most recent dialogue turns for LLM prompt context."""
        recent = self.messages[-max_turns*2:] if len(self.messages) > max_turns*2 else self.messages
        if not recent:
            return ""
        
        history_lines = ["### RECENT CONVERSATION HISTORY:"]
        if self.current_crop or self.current_diagnosed_disease:
            history_lines.append(f"Ongoing Context: Crop={self.current_crop or 'Not specified'}, Diagnosed Disease={self.current_diagnosed_disease or 'None'}")
        
        for msg in recent:
            prefix = "Farmer" if msg["role"] == "user" else "AgroNerve AI"
            history_lines.append(f"{prefix}: {msg['content']}")
        
        return "\n".join(history_lines)

class SessionManager:
    """Manages active chat sessions for multi-turn advisory continuity."""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def get_or_create_session(self, session_id: str) -> ChatSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

session_manager = SessionManager()
