import io
import re
import base64
import requests
from typing import Dict, Any, List, Optional
from config import settings
from core.knowledge_seeder import KnowledgeChunker

CROP_KEYWORDS = {
    "Tomato": ["tomato", "thakkali", "tamatar"],
    "Paddy (Rice)": ["paddy", "rice", "nellu", "dhan", "chawal"],
    "Cotton": ["cotton", "kapas", "paruthi"],
    "Wheat": ["wheat", "gehun", "godhumai"],
    "Chilli": ["chilli", "chili", "pepper", "mirchi", "milagai"]
}

class LeafVisionAnalyzer:
    """Multimodal vision analyzer supporting local Ollama Vision models & robust offline crop-aware foliar symptom extraction."""

    def __init__(self):
        self.disease_data = KnowledgeChunker.load_disease_chunks()

    def _extract_crop_from_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()
        for crop_name, aliases in CROP_KEYWORDS.items():
            for alias in aliases:
                if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
                    return crop_name
        return None

    def _call_ollama_vision(self, image_bytes: bytes, user_query: Optional[str] = None) -> Optional[str]:
        """Attempts inference with local multimodal vision models (e.g. llava, moondream)."""
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                user_query or 
                "You are an expert plant pathologist AI. Analyze this crop leaf photograph. "
                "Identify the crop (e.g. Tomato, Rice, Cotton, Wheat), describe visual symptoms (lesions, chlorosis, concentric rings, spots), "
                "diagnose the disease accurately, and state immediate ICAR management protocols."
            )
            res = requests.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": "llava",
                    "prompt": prompt,
                    "images": [b64_image],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 400}
                },
                timeout=15.0
            )
            if res.status_code == 200:
                return res.json().get("response", "").strip()
        except Exception:
            pass
        return None

    def analyze_image_bytes(self, image_bytes: bytes, crop_hint: str = "auto", user_query: Optional[str] = None) -> Dict[str, Any]:
        """Analyzes leaf image byte data, extracts lesion & chlorosis metrics, and maps to verified crop disease knowledge."""
        # Check if user query contains crop name (e.g. "my tomato has spots")
        inferred_crop = self._extract_crop_from_text(user_query)
        effective_crop_hint = crop_hint
        if crop_hint.lower() == "auto" or not crop_hint:
            effective_crop_hint = inferred_crop or "auto"

        # Try local Ollama vision if available
        ollama_vision_resp = self._call_ollama_vision(image_bytes, user_query)

        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = img.size
            total_pixels = width * height
            
            # Fast offline pixel color distribution analysis
            sample_step = max(1, int((total_pixels / 10000) ** 0.5))
            pixels = [img.getpixel((x, y)) for y in range(0, height, sample_step) for x in range(0, width, sample_step)]
            sample_count = len(pixels) or 1

            yellow_count = 0
            brown_necrotic_count = 0
            green_healthy_count = 0
            dark_spot_count = 0

            for r, g, b in pixels:
                # Yellow / chlorotic
                if r > 130 and g > 130 and b < 100 and (r + g) > (2.2 * b):
                    yellow_count += 1
                # Brown / necrotic
                elif r > 80 and g > 40 and b < 50 and r > (1.3 * g):
                    brown_necrotic_count += 1
                # Dark spot / black blight
                elif r < 60 and g < 60 and b < 60:
                    dark_spot_count += 1
                # Healthy green
                elif g > r and g > b and g > 70:
                    green_healthy_count += 1

            chlorosis_pct = round((yellow_count / sample_count) * 100, 1)
            necrotic_pct = round((brown_necrotic_count / sample_count) * 100, 1)
            dark_lesion_pct = round((dark_spot_count / sample_count) * 100, 1)
            healthy_pct = round((green_healthy_count / sample_count) * 100, 1)

            total_damage_pct = min(100.0, round(chlorosis_pct + necrotic_pct + dark_lesion_pct, 1))

            # Diagnostic symptom inference
            detected_features = []
            if chlorosis_pct > 15.0:
                detected_features.append("Marked leaf yellowing / chlorosis")
            if necrotic_pct > 10.0:
                detected_features.append("Brown necrotic target-like lesions")
            if dark_lesion_pct > 5.0:
                detected_features.append("Dark concentrated necrotic spots")
            if not detected_features:
                detected_features.append("Mild localized foliage discoloration")

            # Match against curated disease database with strict crop-filtering
            match_candidates = []
            for item in self.disease_data:
                crop_name = item.get("crop", "")
                text = item.get("text", "").lower()
                title = item.get("title", "").lower()
                score = 0.0

                # Crop match weighting
                if effective_crop_hint.lower() != "auto":
                    hint_lower = effective_crop_hint.lower()
                    if hint_lower in crop_name.lower() or any(alias in hint_lower for alias in CROP_KEYWORDS.get(crop_name, [])):
                        score += 150.0  # Dominant crop boost
                    else:
                        score -= 50.0   # Penalize mismatched crops

                # Specific symptom matching
                if necrotic_pct > 8.0:
                    if "early blight" in title or "target" in text or "concentric" in text:
                        score += 40.0
                    elif "spot" in text or "brown" in text or "lesion" in text:
                        score += 20.0

                if chlorosis_pct > 15.0:
                    if "curl" in title or "yellow" in title or "chlorosis" in text:
                        score += 35.0
                    elif "yellowing" in text:
                        score += 15.0

                if dark_lesion_pct > 5.0:
                    if "late blight" in title or "blast" in title or "black" in text:
                        score += 30.0

                match_candidates.append((score, item))

            match_candidates.sort(key=lambda x: x[0], reverse=True)
            top_match = match_candidates[0][1] if match_candidates else {}
            
            detected_crop_final = top_match.get("crop", effective_crop_hint if effective_crop_hint != "auto" else "Crop")
            confidence = min(96.0, max(72.0, round(70.0 + (total_damage_pct * 0.25), 1)))

            # If Ollama vision gave a response, use it
            ai_description = ollama_vision_resp if ollama_vision_resp else (
                f"Visual scan identifies symptoms consistent with **{top_match.get('title', 'Foliar Disease')}** on **{detected_crop_final}** "
                f"with approximately **{total_damage_pct}%** affected leaf surface area. "
                f"Observed patterns: {', '.join(detected_features)}."
            )

            return {
                "status": "success",
                "predicted_disease": top_match.get("title", "Suspected Foliar Blight"),
                "crop": detected_crop_final,
                "confidence_pct": confidence,
                "affected_leaf_area_pct": total_damage_pct,
                "ai_description": ai_description,
                "metrics": {
                    "chlorosis_yellow_pct": chlorosis_pct,
                    "necrotic_brown_pct": necrotic_pct,
                    "dark_lesion_pct": dark_lesion_pct,
                    "healthy_green_pct": healthy_pct
                },
                "detected_symptoms": detected_features,
                "verified_protocol": top_match.get("text", "")
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to process leaf image: {str(e)}"
            }

leaf_vision_scanner = LeafVisionAnalyzer()
