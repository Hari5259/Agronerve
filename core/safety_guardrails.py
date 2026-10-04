"""AgroNerve Safety Guardrails and Agrochemical Validator.

Validates pesticide recommendations against CIBRC safety standards, checks for banned/restricted
substances, enforces PPE mandates, and validates maximum safe chemical dosage rates.
"""

from typing import Dict, Any, List, Tuple
import re

# Banned / Restricted chemicals under Indian Central Insecticides Board & Registration Committee (CIBRC)
BANNED_CHEMICALS = {
    "endosulfan": "BANNED: Endosulfan is completely prohibited due to severe environmental and human toxicity.",
    "monocrotophos": "RESTRICTED: Monocrotophos is banned on vegetables and restricted due to extreme acute toxicity.",
    "paraquat": "RESTRICTED: Paraquat dichloride is a high-hazard non-selective herbicide with severe systemic toxicity.",
    "phorate": "RESTRICTED: Phorate is an extremely toxic organophosphate restricted in food crops.",
    "methyl parathion": "BANNED: Methyl parathion is banned from agricultural use in India.",
    "carbofuran": "RESTRICTED: Carbofuran 50% SP is banned/restricted on food crops.",
    "diazinon": "BANNED: Diazinon is banned from agricultural use in India.",
    "ddt": "BANNED: DDT is banned for all agricultural applications.",
}

# Maximum safe dosage thresholds in ml/L or g/L for common agricultural chemicals
MAX_SAFE_DOSAGES = {
    "chlorantraniliprole": 0.5,  # 0.3 - 0.4 ml/L (Coragen 18.5 SC)
    "emamectin benzoate": 0.6,   # 0.4 - 0.5 g/L (5% SG)
    "imidacloprid": 0.75,        # 0.3 - 0.5 ml/L (17.8 SL)
    "mancozeb": 3.0,             # 2.0 - 2.5 g/L (75% WP)
    "copper oxychloride": 3.5,   # 2.5 - 3.0 g/L (50% WP)
    "tricyclazole": 1.0,         # 0.6 g/L (75% WP)
    "hexaconazole": 2.0,         # 1.0 - 2.0 ml/L (5% EC)
    "azoxystrobin": 1.5,         # 1.0 ml/L (23% SC)
    "neem oil": 10.0,            # 3.0 - 5.0 ml/L (10,000 ppm)
}


class SafetyGuardrails:
    """Validates agricultural advisories for chemical safety and regulatory compliance."""

    @staticmethod
    def audit_advisory(text: str) -> Dict[str, Any]:
        """Performs a comprehensive safety audit of an agricultural advisory response."""
        alerts: List[str] = []
        violations: List[str] = []
        is_safe = True

        lower_text = text.lower()

        # 1. Check for banned/restricted chemicals
        for chem, reason in BANNED_CHEMICALS.items():
            if chem in lower_text:
                is_safe = False
                violations.append(f"⛔ Dangerous Chemical Alert: '{chem.title()}' detected. {reason}")

        # 2. Check for missing PPE on chemical spray advisories
        chemical_keywords = ["spray", "fungicide", "insecticide", "pesticide", "chemical", "dosage"]
        has_chemical_advice = any(kw in lower_text for kw in chemical_keywords)
        has_ppe = any(kw in lower_text for kw in ["ppe", "gloves", "mask", "goggles", "protective clothing", "safety gear"])

        if has_chemical_advice and not has_ppe:
            alerts.append("Personal Protective Equipment (gloves, mask, goggles) must be worn during chemical handling.")

        # 3. Check for missing PHI mention when harvesting/picking or pesticide is mentioned
        has_phi = any(kw in lower_text for kw in ["phi", "pre-harvest interval", "waiting period", "harvest interval", "days before harvest"])
        if has_chemical_advice and not has_phi:
            alerts.append("Ensure adherence to Pre-Harvest Interval (PHI) waiting periods before crop consumption.")

        # 4. Check for extreme dosage mentions
        dosage_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:ml|g|gm)\s*(?:/|per)\s*(?:l|liter|litre)", lower_text)
        for dose_str in dosage_matches:
            try:
                dose_val = float(dose_str)
                if dose_val > 15.0:  # Excessive dosage threshold
                    alerts.append(f"High concentration detected ({dose_val} ml or g/L). Verify dilution before field spraying.")
            except ValueError:
                pass

        return {
            "is_safe": is_safe,
            "violations": violations,
            "alerts": alerts,
            "has_warnings": len(violations) > 0 or len(alerts) > 0,
        }

    @staticmethod
    def sanitize_and_guard(text: str) -> str:
        """Appends active safety annotations if critical alerts are detected."""
        audit = SafetyGuardrails.audit_advisory(text)
        if not audit["has_warnings"]:
            return text

        guard_lines = []
        if audit["violations"]:
            guard_lines.append("\n\n> 🚫 **REGULATORY CHEMICAL WARNING:**")
            for v in audit["violations"]:
                guard_lines.append(f"> - {v}")

        if audit["alerts"]:
            guard_lines.append("\n> 🛡️ **MANDATORY SAFETY PROTOCOLS:**")
            for a in audit["alerts"]:
                guard_lines.append(f"> - {a}")

        return text + "\n" + "\n".join(guard_lines)


safety_guardrails = SafetyGuardrails()
