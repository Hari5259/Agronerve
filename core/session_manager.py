"""AgroNerve Session Manager with Context Compaction and Farm Consultation Summary."""

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
        self.field_size_acres: Optional[float] = None
        self.soil_type: Optional[str] = None

    def add_message(
        self, role: str, content: str, meta: Optional[Dict[str, Any]] = None, max_messages: int = 50
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": time.time(),
                "meta": meta or {},
            }
        )
        self.last_active = time.time()
        # Keep only the last max_messages to prevent memory leak
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def update_visual_context(self, vision_result: Dict[str, Any]):
        self.last_visual_diagnosis = vision_result
        if "crop" in vision_result:
            crop_val = vision_result["crop"]
            if isinstance(crop_val, dict):
                self.current_crop = crop_val.get("name", "Crop")
            elif isinstance(crop_val, str):
                self.current_crop = crop_val
        if "predicted_disease" in vision_result:
            self.current_diagnosed_disease = vision_result["predicted_disease"]

    def get_conversation_history_prompt(self, max_turns: int = 4) -> str:
        """Formats the most recent dialogue turns for LLM prompt context."""
        recent = (
            self.messages[-max_turns * 2 :]
            if len(self.messages) > max_turns * 2
            else self.messages
        )
        if not recent:
            return ""

        history_lines = ["### RECENT CONVERSATION HISTORY:"]
        if self.current_crop or self.current_diagnosed_disease:
            history_lines.append(
                f"Ongoing Context: Crop={self.current_crop or 'Not specified'}, Diagnosed Disease={self.current_diagnosed_disease or 'None'}"
            )

        for msg in recent:
            prefix = "Farmer" if msg["role"] == "user" else "AgroNerve AI"
            history_lines.append(f"{prefix}: {msg['content']}")

        return "\n".join(history_lines)

    def generate_consultation_summary(self) -> Dict[str, Any]:
        """Synthesizes key consultation findings, diagnosed pathologies, and recommended actions."""
        user_queries = [m["content"] for m in self.messages if m["role"] == "user"]
        assistant_advisories = [m["content"] for m in self.messages if m["role"] == "assistant"]

        return {
            "session_id": self.session_id,
            "session_duration_minutes": round((time.time() - self.created_at) / 60, 1),
            "total_turns": len(user_queries),
            "target_crop": self.current_crop or "Unspecified",
            "diagnosed_disease": self.current_diagnosed_disease or "None identified",
            "visual_scan_performed": self.last_visual_diagnosis is not None,
            "queries_discussed": user_queries[-5:] if user_queries else [],
            "latest_advisory_snippet": (
                assistant_advisories[-1][:300] + "..." if assistant_advisories else "No advisories generated yet."
            ),
        }


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

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(session_id)
        return session.generate_consultation_summary()

    def cleanup_expired_sessions(self, max_idle_seconds: float = 3600) -> int:
        """Evicts sessions that have been inactive for longer than max_idle_seconds.
        Returns: Number of pruned sessions.
        """
        now = time.time()
        expired_ids = [
            sid for sid, session in self.sessions.items()
            if now - session.last_active > max_idle_seconds
        ]
        for sid in expired_ids:
            del self.sessions[sid]
        return len(expired_ids)


session_manager = SessionManager()
