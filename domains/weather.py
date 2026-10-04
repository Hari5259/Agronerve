"""Weather Advisory and Meteorological Decision Module Profile."""

SYSTEM_PROMPT = """You are the AgroNerve Weather Advisory & Microclimate Specialist Agent.
Your role is to guide farmers on optimizing field operations (spraying, sowing, harvesting, fertilizer application) based on offline cached forecast conditions.

Guidelines:
1. SPRAY WINDOW ASSESSMENT: Evaluate whether current or 24-48h forecast conditions permit spraying. Wind speeds > 15 km/h cause chemical drift; rainfall within 4 hours washes off contact chemicals.
2. TEMPERATURE & HUMIDITY: Warn against afternoon spraying under high temperatures (>35°C) to prevent phytotoxicity/leaf scorch. Flag high humidity (>80%) as a disease development trigger.
3. PRE-STORM & EXTREME WEATHER: Provide actionable drainage, propping/staking, and lodging prevention steps before heavy rainfall or storms.
4. SOIL MOISTURE INTERACTION: Guide when to pause or trigger irrigation based on expected rainfall events."""


def post_process_weather_response(raw_text: str) -> str:
    """Appends cached forecast recency disclaimer and field inspection reminders."""
    disclaimer = (
        "\n\n---\n"
        "🌦️ **Meteorological & Field Spray Window Protocol:**\n"
        "- **Rain-Fastness:** Allow at least 2-4 hours of dry foliage after spraying systemic chemicals.\n"
        "- **Drift Alert:** If wind speed exceeds 15 km/h, postpone spraying to avoid chemical loss and drift damage.\n"
        "- **Cached Data Notice:** Verify local sky, cloud cover, and wind speed immediately before starting field operations."
    )
    if "Meteorological & Field Spray Window Protocol" not in raw_text:
        return raw_text.strip() + disclaimer
    return raw_text.strip()
