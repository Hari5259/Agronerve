"""AgroNerve Voice Engine with Agricultural Phonetics and TTS Normalization."""

import re
from typing import Dict, Any


class VoiceEngine:
    """Processes agricultural advisories for offline speech synthesis and voice UI."""

    @staticmethod
    def clean_text_for_speech(markdown_text: str) -> str:
        """Strips markdown formatting and expands agricultural abbreviations for smooth natural voice readout."""
        text = markdown_text

        # Remove emojis and symbols
        text = re.sub(r"[📸🔬💧🌦️⚠️🛡️🌿⏱️☀️🚫📍⚡]", "", text)

        # Remove markdown headers and URLs
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove bold, italics, bullets, blockquotes
        text = re.sub(r"[*_`>~]", "", text)
        text = re.sub(r"--+", "", text)
        text = re.sub(r"\|", " ", text)

        # Expand agricultural formulations & technical units
        abbreviations = [
            (r"\bml/L\b", "milliliters per liter"),
            (r"\bg/L\b", "grams per liter"),
            (r"\bkg/ha\b", "kilograms per hectare"),
            (r"\bkg/acre\b", "kilograms per acre"),
            (r"\bkm/h\b", "kilometers per hour"),
            (r"\bLPH\b", "liters per hour"),
            (r"\bDAS\b", "days after sowing"),
            (r"\bVWC\b", "Volumetric Water Content"),
            (r"\bRH\b", "Relative Humidity"),
            (r"\bETc\b", "crop evapotranspiration"),
            (r"\bETo\b", "reference evapotranspiration"),
            (r"\bppm\b", "parts per million"),
            (r"\bNPK\b", "N P K"),
            (r"\bFYM\b", "farmyard manure"),
            (r"\bAWD\b", "alternate wetting and drying"),
            (r"\bWP\b", "wettable powder"),
            (r"\bSC\b", "suspension concentrate"),
            (r"\bSL\b", "soluble liquid"),
            (r"\bEC\b", "emulsifiable concentrate"),
            (r"\bWG\b", "water dispersible granule"),
            (r"°C\b", " degrees Celsius"),
            (r"%", " percent"),
            (r"\bPPE\b", "personal protective equipment"),
            (r"\bPHI\b", "pre-harvest interval"),
            (r"\bCIBRC\b", "Central Insecticides Board"),
            (r"\bICAR\b", "Indian Council of Agricultural Research"),
            (r"\bKVK\b", "Krishi Vigyan Kendra"),
        ]

        for pattern, replacement in abbreviations:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Clean multiple newlines and extra spaces
        text = re.sub(r"\n+", ". ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def generate_html5_audio_speech_script(spoken_text: str, lang: str = "en") -> str:
        """Returns browser Web Speech API JavaScript snippet for local offline audio playback."""
        clean = VoiceEngine.clean_text_for_speech(spoken_text)
        escaped = clean.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
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
