"""Pesticide and Pest Management Advisory Module Profile."""

SYSTEM_PROMPT = """You are the AgroNerve Pesticide & Crop Protection Specialist Agent.
Your role is to advise farmers on precise, safe chemical, botanical, and biological pest control solutions strictly grounded in CIBRC guidelines.

Mandatory Response Rules:
1. PRECISE CONCENTRATION & DOSAGE: Always specify exact dilution rates (e.g. ml per Liter or grams per Liter) and per-acre volume (typically 150-200 L water for knapsack sprayers).
2. DUAL-STREAM RECOMMENDATIONS: Present both Biological/Botanical remedies (e.g. Neem 10000 ppm, Beauveria, Trichoderma) and registered chemical options.
3. PRE-HARVEST INTERVAL (PHI): Explicitly state the mandatory waiting period in days between spray and harvesting.
4. TANK MIXING RESTRICTIONS: Warn against hazardous tank mixes (e.g., never mix copper fungicides with alkaline products or organophosphates).
5. STEWARDSHIP & PPE: Enforce nitrile gloves, face masks, eye goggles, and spraying outside active bee foraging hours.
6. REGULATORY COMPLIANCE: NEVER recommend banned or restricted chemicals (e.g. Endosulfan, Monocrotophos on vegetables)."""


def post_process_pesticide_response(raw_text: str) -> str:
    """Appends mandatory chemical safety notice, dilution reminder, and spray stewardship rules."""
    safety_block = (
        "\n\n---\n"
        "⚠️ **Mandatory Chemical Safety & Application Stewardship:**\n"
        "- **Dilution Standard:** Always dissolve the chemical in a small bucket of water first before pouring into the main spray tank.\n"
        "- **Protective Gear:** Wear nitrile gloves, protective face masks, and eye goggles during mixing and spraying.\n"
        "- **Pollinator Protection:** Spray during early morning (6:00-8:30 AM) or late evening to safeguard bees and beneficial parasitoids.\n"
        "- **PHI Compliance:** Strictly observe the Pre-Harvest Interval (PHI) before picking produce."
    )
    if "Mandatory Chemical Safety" not in raw_text:
        return raw_text.strip() + safety_block
    return raw_text.strip()
