"""AgroNerve Structured Response Formatter.

Transforms raw agronomic responses and knowledge synthesis into beautifully structured,
farmer-friendly markdown advisory cards with clear visual hierarchy, action badges, and
follow-up recommendations.
"""

from typing import Dict, Any, List, Optional
import re


class ResponseFormatter:
    """Formats agricultural AI responses into structured, standardized advisories."""

    @staticmethod
    def format_structured_advisory(
        raw_text: str,
        domain: str = "general",
        crop: Optional[str] = None,
        language: str = "en",
        safety_alerts: Optional[List[str]] = None,
        action_badges: Optional[List[str]] = None,
    ) -> str:
        """Enriches raw advisory text with clean sections, badge highlights, and safety notices."""
        lines = []

        # 1. Action Badges Header
        badges = action_badges or []
        if domain == "disease":
            badges.append("🔬 Crop Pathology Advisory")
        elif domain == "pesticide":
            badges.append("🛡️ Precision Crop Protection")
        elif domain == "weather":
            badges.append("🌦️ Meteorological Guidance")
        elif domain == "irrigation":
            badges.append("💧 Water & Moisture Advisory")

        if crop:
            badges.append(f"🌱 Target Crop: {crop.capitalize()}")

        if badges:
            badge_line = "  ".join([f"`{b}`" for b in badges])
            lines.append(f"{badge_line}\n")

        # 2. Critical Safety Alerts (if any)
        if safety_alerts:
            lines.append("> ⚠️ **CRITICAL SAFETY ADVISORY:**")
            for alert in safety_alerts:
                lines.append(f"> - {alert}")
            lines.append("")

        # 3. Clean and normalize main content
        cleaned_text = ResponseFormatter._clean_text(raw_text)
        lines.append(cleaned_text)

        # 4. Standardized footer formatting
        return "\n".join(lines).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Removes excessive newlines and standardizes list item markers."""
        # Replace 3 or more consecutive newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def create_dosage_table(
        treatments: List[Dict[str, Any]]
    ) -> str:
        """Generates a GitHub-flavored markdown table for chemical & bio-treatments."""
        if not treatments:
            return ""

        headers = ["Chemical / Bio-Agent", "Dilution Rate", "Application Method", "PHI (Days)"]
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = [header_row, separator_row]
        for t in treatments:
            name = t.get("name", "N/A")
            dose = t.get("dosage", "N/A")
            method = t.get("method", "Foliar Spray")
            phi = str(t.get("phi_days", "N/A"))
            rows.append(f"| **{name}** | `{dose}` | {method} | {phi} days |")

        return "\n".join(rows)

    @staticmethod
    def format_water_budget_card(
        crop: str,
        stage: str,
        water_req_mm: float,
        interval_days: int,
        soil_type: str = "Loam",
    ) -> str:
        """Generates a structured irrigation summary card."""
        return (
            f"### 💧 Irrigation Water Budget Summary\n"
            f"- **Target Crop:** {crop} ({stage})\n"
            f"- **Estimated Water Requirement:** `{water_req_mm} mm/day`\n"
            f"- **Recommended Irrigation Cycle:** Every `{interval_days}` days\n"
            f"- **Soil Adaptation ({soil_type}):** Adjust duration based on field moisture sensors."
        )


response_formatter = ResponseFormatter()
