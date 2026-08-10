import re
from typing import Literal

DomainType = Literal["disease", "pesticide", "weather", "irrigation", "general"]

KEYWORDS = {
    "disease": [
        "spot", "spots", "yellow", "yellowing", "wilt", "wilting", "fungus", 
        "blight", "rot", "mildew", "rust", "leaf", "disease", "infection", "symptom"
    ],
    "pesticide": [
        "pesticide", "insecticide", "fungicide", "spray", "dosage", "chemical", 
        "pest", "insect", "worm", "caterpillar", "aphid", "control", "ml/l", "dose"
    ],
    "weather": [
        "weather", "rain", "rainfall", "temperature", "forecast", "humidity", 
        "wind", "climate", "monsoon", "frost", "heat"
    ],
    "irrigation": [
        "irrigation", "water", "watering", "drip", "moisture", "dry", 
        "soil", "sprinkler", "interval", "flood", "liter"
    ]
}

class IntentRouter:
    """Two-stage intent classification for domain-specific agent assembly."""

    def route(self, query: str) -> DomainType:
        q_lower = query.lower()
        scores = {domain: 0 for domain in KEYWORDS}

        for domain, words in KEYWORDS.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', q_lower):
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        if scores[best_domain] > 0:
            return best_domain

        return "general"
