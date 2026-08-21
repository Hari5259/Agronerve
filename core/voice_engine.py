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
        
        # Translate technical units / abbreviations to readable spoken words
        text = re.sub(r"\bml/L\b", "milliliters per liter", text, flags=re.IGNORECASE)
        text = re.sub(r"\bg/L\b", "grams per liter", text, flags=re.IGNORECASE)
        text = re.sub(r"\bVWC\b", "Volumetric Water Content", text, flags=re.IGNORECASE)
        text = re.sub(r"\bRH\b", "Relative Humidity", text, flags=re.IGNORECASE)
        text = re.sub(r"°C\b", " degrees Celsius", text)
        text = re.sub(r"%", " percent", text)
        text = re.sub(r"\bPPE\b", "P P E", text)
        text = re.sub(r"\bPHI\b", "P H I", text)

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
        <div style="margin: 0.8rem 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <button onclick="speakText()" style="background-color: #1b4332; color: #d8f3dc; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600;">
                🔊 Read Advisory Aloud
            </button>
            <button onclick="window.speechSynthesis.cancel()" style="background-color: #7f1d1d; color: #fecaca; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600;">
                ⏹ Stop
            </button>
            <span style="font-size: 0.85rem; color: #d8f3dc; display: flex; align-items: center; gap: 4px; background-color: #1b4332; padding: 4px 8px; border-radius: 6px;">
                Speed: 
                <input type="range" id="speechRate" min="0.5" max="2.0" value="0.95" step="0.05" style="width: 70px; accent-color: #d8f3dc; cursor: pointer; vertical-align: middle;">
            </span>
            <script>
                function speakText() {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance("{escaped}");
                    msg.lang = "{lang_code}";
                    var speedEl = document.getElementById("speechRate");
                    msg.rate = speedEl ? parseFloat(speedEl.value) : 0.95;
                    window.speechSynthesis.speak(msg);
                }}
            </script>
        </div>
        """


voice_engine = VoiceEngine()
