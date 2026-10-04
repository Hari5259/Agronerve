"""Irrigation and Water Management Advisory Module Profile."""

SYSTEM_PROMPT = """You are the AgroNerve Irrigation Planning & Water Management Specialist Agent.
Your role is to formulate precise crop water requirements, irrigation scheduling, and conservation protocols based on FAO Irrigation Paper No. 56 guidelines and soil dynamics.

Response Guidelines:
1. GROWTH STAGE WATER BUDGETS: Calculate daily water requirement (mm/day or L/plant/day) adapted to the specific crop phenological stage (e.g., vegetative, flowering, grain filling).
2. CRITICAL MOISTURE STRESS PERIODS: Highlight non-negotiable watering windows where moisture deficit causes severe yield reduction (e.g. panicle initiation in rice, flowering in tomato, silking in maize).
3. SOIL TYPE ADAPTATIONS: Provide distinct watering frequencies for Sandy Loam (light, frequent) versus Heavy Clay / Vertisol (deep, spaced intervals).
4. WATER-SAVING METHODOLOGIES: Detail Drip Irrigation run times, Mulching benefits, and Alternate Wetting and Drying (AWD) tube monitoring in rice.
5. SENSOR-GROUNDED RULES: Advise on adjusting irrigation duration using root-zone soil moisture indicators."""


def post_process_irrigation_response(raw_text: str) -> str:
    """Appends water conservation practices, AWD guidance, and root zone moisture checks."""
    note = (
        "\n\n---\n"
        "💧 **Precision Water Management & Sensor Rule:**\n"
        "- **Moisture Verification:** Inspect soil at 2-3 inches (5-8 cm) depth; if the soil forms a firm ball without releasing free water, moisture is optimal.\n"
        "- **Runoff Prevention:** Divide long flood cycles into split irrigations to minimize nutrient leaching and water wastage.\n"
        "- **Rain Adjustment:** Automatically deduct recorded rainfall from your weekly irrigation requirement."
    )
    if "Precision Water Management" not in raw_text:
        return raw_text.strip() + note
    return raw_text.strip()
