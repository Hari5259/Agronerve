import re
from typing import Dict, Any


class VoiceEngine:
    """Processes agricultural advisories for offline speech synthesis and voice UI."""

    @staticmethod
    def clean_text_for_speech(markdown_text: str) -> str:
        """Strips markdown headers, asterisks, bullet formatting, and disclaimers for smooth natural voice readout."""
        # Remove markdown headers and URLs
        text = re.sub(r"#+\s*", "", markdown_text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove bold, italics, bullets, blockquotes
        text = re.sub(r"[*_`>~]", "", text)
        text = re.sub(r"--+", "", text)
        # Clean multiple newlines and extra spaces
        text = re.sub(r"\n+", ". ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def generate_html5_audio_speech_script(spoken_text: str, lang: str = "en") -> str:
        """Returns browser Web Speech API JavaScript snippet for local offline audio playback."""
        escaped = spoken_text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
        lang_code = {
            "en": "en-IN",
            "ta": "ta-IN",
            "hi": "hi-IN",
            "te": "te-IN",
            "kn": "kn-IN",
        }.get(lang, "en-IN")

        return f"""
        <div style="margin: 0.8rem 0;">
            <button onclick="speakText()" style="background-color: #1b4332; color: #d8f3dc; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600;">
                🔊 Read Advisory Aloud
            </button>
            <button onclick="window.speechSynthesis.cancel()" style="background-color: #7f1d1d; color: #fecaca; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; margin-left: 6px;">
                ⏹ Stop
            </button>
            <script>
                function speakText() {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{escaped}");
                    msg.lang = "{lang_code}";
                    msg.rate = 0.95;
                    window.speechSynthesis.speak(msg);
                }}
            </script>
        </div>
        """


voice_engine = VoiceEngine()
