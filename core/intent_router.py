import re
from typing import Literal, Dict, Any, Tuple
from config import settings

DomainType = Literal["disease", "pesticide", "weather", "irrigation", "general"]

# Discriminative weighted keywords based on AgroNerve agricultural vocabulary
DOMAIN_WEIGHTS: Dict[str, Dict[str, float]] = {
    "disease": {
        "spot": 2.5, "spots": 2.5, "yellowing": 2.0, "yellow": 1.5, "wilt": 2.5, "wilting": 2.5,
        "fungus": 3.0, "blight": 3.0, "rot": 2.5, "mildew": 3.0, "rust": 3.0, "leaf": 1.0,
        "disease": 3.0, "infection": 2.5, "symptom": 2.0, "pustule": 3.0, "blast": 3.0,
        "canker": 3.0, "anthracnose": 3.0, "lesion": 2.5, "lesions": 2.5, "curling": 2.0,
        "ooze": 2.5, "neck": 3.0, "sheath": 3.0, "chaffy": 3.0, "stripe": 2.5, "stripes": 2.5,
        "drying": 2.0, "rings": 2.0
    },
    "pesticide": {
        "pesticide": 3.5, "insecticide": 3.5, "fungicide": 3.5, "spray": 2.2, "spraying": 2.2,
        "dosage": 3.0, "dose": 3.0, "chemical": 2.5, "pest": 2.5, "insect": 2.5, "worm": 2.5,
        "caterpillar": 3.0, "aphid": 3.0, "thrips": 3.0, "whitefly": 3.0, "control": 1.5,
        "ml/l": 3.5, "g/l": 3.5, "neem": 3.0, "coragen": 3.5, "mancozeb": 3.5, "imidacloprid": 3.5,
        "phi": 3.0, "pre-harvest": 3.5, "toxic": 2.0, "ppe": 2.5, "organic": 2.8, "diluted": 2.5,
        "chlorantraniliprole": 3.5, "tricyclazole": 3.5, "emamectin": 3.5, "copper": 2.5, "mix": 2.0
    },
    "weather": {
        "weather": 3.5, "rain": 3.5, "rains": 3.5, "raining": 3.5, "rainfall": 3.5, "temperature": 3.0, "forecast": 3.5,
        "humidity": 3.0, "wind": 3.0, "climate": 2.5, "monsoon": 3.0, "frost": 3.0,
        "heat": 2.0, "heatwave": 3.5, "drizzle": 3.0, "storm": 3.0, "cloudy": 2.5, "clouds": 2.5,
        "overcast": 3.0, "dew": 2.5, "cached": 2.5, "speed": 2.0, "degrees": 2.5
    },
    "irrigation": {
        "irrigation": 3.5, "water": 1.8, "watering": 3.0, "drip": 3.5, "moisture": 2.5,
        "dry": 1.5, "soil": 2.0, "sprinkler": 3.5, "interval": 2.5, "intervals": 2.5, "flood": 2.5,
        "liter": 2.0, "awd": 3.5, "evapotranspiration": 3.5, "fao": 2.5, "tillering": 2.0,
        "clay": 2.0, "loam": 2.0, "schedule": 2.5, "schedules": 2.5, "stages": 2.5, "root": 2.0,
        "flowering": 1.5
    }
}

class IntentRouter:
    """Two-stage hybrid intent classification engine with 1.5x margin verification."""

    def __init__(self):
        self.margin_threshold = 1.5

    def route_stage1(self, query: str) -> Tuple[DomainType, float, bool]:
        """Stage 1: Fast rule-based weighted keyword matching.
        Returns: (domain, top_score, is_confident)
        """
        q_lower = query.lower()
        scores: Dict[str, float] = {domain: 0.0 for domain in DOMAIN_WEIGHTS}

        for domain, weights in DOMAIN_WEIGHTS.items():
            for word, weight in weights.items():
                # Word boundary match
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = len(re.findall(pattern, q_lower))
                if matches > 0:
                    scores[domain] += weight * matches

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_domain, top_score = sorted_scores[0]
        second_domain, second_score = sorted_scores[1]

        if top_score == 0:
            return "general", 0.0, False

        # Confident if top score >= 1.5x second score (or second score is 0 and top >= 2.0)
        is_confident = (top_score >= self.margin_threshold * second_score) if second_score > 0 else (top_score >= 2.0)
        return top_domain, top_score, is_confident

    def route_stage2_llm(self, query: str) -> DomainType:
        """Stage 2: Lightweight fast LLM fallback for ambiguous queries."""
        try:
            import requests
            prompt = (
                "You are an intent classification router for an agricultural advisory system. "
                "Classify the following farmer query into EXACTLY one of these domains: "
                "disease, pesticide, weather, irrigation, or general.\n\n"
                f"Query: \"{query}\"\n\n"
                "Respond with ONLY the single domain word in lowercase."
            )
            response = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.DEFAULT_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 10}
                },
                timeout=0.8
            )
            if response.status_code == 200:
                raw_label = response.json().get("response", "").strip().lower()
                for d in ["disease", "pesticide", "weather", "irrigation", "general"]:
                    if d in raw_label:
                        return d
        except Exception:
            pass

        return "general"

    def route(self, query: str) -> Dict[str, Any]:
        """Runs the full 2-stage routing pipeline with diagnostic metadata."""
        domain, score, is_confident = self.route_stage1(query)
        stage_used = 1

        if not is_confident and domain != "general":
            # Attempt Stage 2 LLM fallback if keyword pass was not decisively confident
            llm_domain = self.route_stage2_llm(query)
            if llm_domain != "general":
                domain = llm_domain
                stage_used = 2

        return {
            "domain": domain,
            "score": score,
            "is_confident": is_confident,
            "stage_used": stage_used
        }
