import json
from pathlib import Path
from typing import Dict, Any

TRANSLATION_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "translations" / "languages.json"
)

SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "தமிழ் (Tamil)",
    "hi": "हिन्दी (Hindi)",
    "te": "తెలుగు (Telugu)",
    "kn": "ಕನ್ನಡ (Kannada)",
}


class LanguageManager:
    """Manages multilingual agricultural UI labels, system prompts, and responses."""

    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()

    def _load_translations(self):
        if TRANSLATION_FILE.exists():
            with open(TRANSLATION_FILE, "r", encoding="utf-8") as f:
                self.translations = json.load(f)

    def get_text(self, key: str, lang: str = "en") -> str:
        lang_dict = self.translations.get(lang, self.translations.get("en", {}))
        return lang_dict.get(key, self.translations.get("en", {}).get(key, key))

    def get_language_prompt_instruction(self, lang: str) -> str:
        """Constructs localized instruction for the LLM output with simplified vocabulary rules."""
        base_instruction = (
            "\n\nIMPORTANT: Use simple, plain, and conversational phrasing. Avoid complex English technical jargon "
            "where common localized terminology is widely understood. Use short sentences and clean bullet points."
        )
        if lang == "ta":
            return base_instruction + " Output the response in clear, formal Tamil (தமிழ்) suitable for Tamil Nadu farmers."
        elif lang == "hi":
            return base_instruction + " Output the response in clear, easy-to-understand Hindi (हिन्दी) suitable for Indian farmers."
        elif lang == "te":
            return base_instruction + " Output the response in clear Telugu (తెలుగు) suitable for Andhra Pradesh and Telangana farmers."
        elif lang == "kn":
            return base_instruction + " Output the response in clear Kannada (ಕನ್ನಡ) suitable for Karnataka farmers."
        return ""


language_manager = LanguageManager()
