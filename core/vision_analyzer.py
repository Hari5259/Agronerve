import io
import re
import base64
import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from config import settings
from core.knowledge_seeder import KnowledgeChunker

logger = logging.getLogger(__name__)

CROP_KEYWORDS = {
    "Tomato": ["tomato", "thakkali", "tamatar"],
    "Paddy (Rice)": ["paddy", "rice", "nellu", "dhan", "chawal"],
    "Cotton": ["cotton", "kapas", "paruthi"],
    "Wheat": ["wheat", "gehun", "godhumai"],
    "Chilli": ["chilli", "chili", "pepper", "mirchi", "milagai"],
    "Maize (Corn)": ["maize", "corn", "cholam", "makka", "makai"],
    "Sugarcane": ["sugarcane", "karumbu", "ganna"],
    "Groundnut": ["groundnut", "peanut", "verkadalai", "moongfali"],
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
                if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                    return crop_name
        return None

    def _parse_vision_response(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parses the crop and disease name from the Ollama Vision response text."""
        text_lower = text.lower()
        detected_crop = None
        for crop_name, aliases in CROP_KEYWORDS.items():
            for alias in aliases:
                if re.search(r"\b" + re.escape(alias) + r"\b", text_lower):
                    detected_crop = crop_name
                    break
            if detected_crop:
                break

        detected_disease = None
        for item in self.disease_data:
            disease_title = item.get("title", "")
            clean_title = re.sub(r"\(.*?\)", "", disease_title).strip().lower()
            if clean_title in text_lower or disease_title.lower() in text_lower:
                detected_disease = disease_title
                break
            parts = clean_title.split()
            if len(parts) >= 2 and all(p in text_lower for p in parts):
                detected_disease = disease_title
                break

        return detected_crop, detected_disease

    def _call_ollama_vision(
        self, image_bytes: bytes, user_query: Optional[str] = None
    ) -> Optional[str]:
        """Attempts inference with local multimodal vision models (e.g. llava, moondream)."""
        logger.info("Attempting local Ollama vision inference.")
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                user_query
                or "You are an expert plant pathologist AI. Analyze this crop leaf photograph. "
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
                    "options": {"temperature": 0.2, "num_predict": 400},
                },
                timeout=15.0,
            )
            if res.status_code == 200:
                response_text = res.json().get("response", "").strip()
                logger.info("Local Ollama vision inference successful.")
                return response_text
            else:
                logger.warning(
                    f"Ollama vision API returned status code {res.status_code}"
                )
        except Exception as e:
            logger.warning(f"Failed to run Ollama vision inference: {str(e)}")
            pass
        return None

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        crop_hint: str = "auto",
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyzes leaf image byte data, extracts lesion & chlorosis metrics, and maps to verified crop disease knowledge."""
        logger.info(
            f"Analyzing leaf image bytes with crop_hint: '{crop_hint}', user_query: '{user_query}'"
        )
        if not image_bytes:
            return {
                "status": "error",
                "message": "Empty image bytes provided.",
            }

        # Check if user query contains crop name (e.g. "my tomato has spots")
        inferred_crop = self._extract_crop_from_text(user_query)
        effective_crop_hint = crop_hint
        if crop_hint.lower() == "auto" or not crop_hint:
            effective_crop_hint = inferred_crop or "auto"
        logger.info(
            f"Inferred crop: '{inferred_crop}', effective crop hint: '{effective_crop_hint}'"
        )

        # Try local Ollama vision if available
        ollama_vision_resp = self._call_ollama_vision(image_bytes, user_query)

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = img.size
            if width < 10 or height < 10:
                return {
                    "status": "error",
                    "message": f"Image dimensions {width}x{height} are too small (minimum 10x10 required).",
                }
            if width > 8000 or height > 8000:
                return {
                    "status": "error",
                    "message": f"Image dimensions {width}x{height} exceed maximum limits (8000x8000).",
                }
            total_pixels = width * height

            # Fast offline pixel color distribution analysis
            sample_step = max(1, int((total_pixels / 10000) ** 0.5))
            pixels = [
                img.getpixel((x, y))
                for y in range(0, height, sample_step)
                for x in range(0, width, sample_step)
            ]
            sample_count = len(pixels) or 1

            yellow_count = 0
            brown_necrotic_count = 0
            green_healthy_count = 0
            dark_spot_count = 0

            for r, g, b in pixels:
                # Yellow / chlorotic
                if r > 110 and g > 110 and b < 110 and (r + g) > (2.0 * b):
                    yellow_count += 1
                # Brown / necrotic
                elif r > 60 and g > 30 and b < 80 and r > (1.2 * g):
                    brown_necrotic_count += 1
                # Dark spot / black blight
                elif r < 75 and g < 75 and b < 75:
                    dark_spot_count += 1
                # Healthy green
                elif g > r and g > b and g > 55:
                    green_healthy_count += 1

            leaf_pixel_count = (
                yellow_count
                + brown_necrotic_count
                + dark_spot_count
                + green_healthy_count
            )
            if leaf_pixel_count == 0:
                leaf_pixel_count = sample_count

            chlorosis_pct = round((yellow_count / leaf_pixel_count) * 100, 1)
            necrotic_pct = round((brown_necrotic_count / leaf_pixel_count) * 100, 1)
            dark_lesion_pct = round((dark_spot_count / leaf_pixel_count) * 100, 1)
            healthy_pct = round((green_healthy_count / leaf_pixel_count) * 100, 1)

            total_damage_pct = min(
                100.0, round(chlorosis_pct + necrotic_pct + dark_lesion_pct, 1)
            )
            logger.info(
                f"Offline leaf color analysis metrics: chlorosis={chlorosis_pct}%, "
                f"necrotic={necrotic_pct}%, dark_lesion={dark_lesion_pct}%, total_damage={total_damage_pct}%"
            )

            # Diagnostic symptom inference
            detected_features = []
            if chlorosis_pct > 5.0:
                detected_features.append("Marked leaf yellowing / chlorosis")
            if necrotic_pct > 3.0:
                detected_features.append("Brown necrotic target-like lesions")
            if dark_lesion_pct > 2.0:
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
                    if hint_lower in crop_name.lower() or any(
                        alias in hint_lower
                        for alias in CROP_KEYWORDS.get(crop_name, [])
                    ):
                        score += 150.0  # Dominant crop boost
                    else:
                        score -= 50.0  # Penalize mismatched crops

                # Specific symptom matching
                if necrotic_pct > 3.0:
                    if (
                        "early blight" in title
                        or "target" in text
                        or "concentric" in text
                    ):
                        score += 40.0
                    elif "spot" in text or "brown" in text or "lesion" in text:
                        score += 20.0

                if chlorosis_pct > 5.0:
                    if "curl" in title or "yellow" in title or "chlorosis" in text:
                        score += 35.0
                    elif "yellowing" in text or "yellow" in text:
                        score += 15.0

                if dark_lesion_pct > 2.0:
                    if "late blight" in title or "blast" in title or "black" in text:
                        score += 30.0

                match_candidates.append((score, item))

            match_candidates.sort(key=lambda x: x[0], reverse=True)
            top_match = match_candidates[0][1] if match_candidates else {}
            logger.info(
                f"Top disease match: '{top_match.get('title', 'None')}' (score={match_candidates[0][0] if match_candidates else 0.0})"
            )

            detected_crop_final = top_match.get(
                "crop", effective_crop_hint if effective_crop_hint != "auto" else "Crop"
            )

            # If Ollama vision returned a response, use its diagnosis to override if parsed
            if ollama_vision_resp:
                inferred_crop_ollama, inferred_disease_ollama = (
                    self._parse_vision_response(ollama_vision_resp)
                )
                if inferred_crop_ollama:
                    detected_crop_final = inferred_crop_ollama
                if inferred_disease_ollama:
                    for item in self.disease_data:
                        if item.get("title") == inferred_disease_ollama:
                            top_match = item
                            break

            confidence = min(
                96.0, max(72.0, round(70.0 + (total_damage_pct * 0.25), 1))
            )

            # If Ollama vision gave a response, use it
            ai_description = (
                ollama_vision_resp
                if ollama_vision_resp
                else (
                    f"Visual scan identifies symptoms consistent with **{top_match.get('title', 'Foliar Disease')}** on **{detected_crop_final}** "
                    f"with approximately **{total_damage_pct}%** affected leaf surface area. "
                    f"Observed patterns: {', '.join(detected_features)}."
                )
            )

            result = {
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
                    "healthy_green_pct": healthy_pct,
                },
                "detected_symptoms": detected_features,
                "verified_protocol": top_match.get("text", ""),
            }
            logger.info(
                f"Foliar diagnostic analysis succeeded: predicted_disease='{result['predicted_disease']}', crop='{result['crop']}', confidence={result['confidence_pct']}%"
            )
            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to process leaf image: {str(e)}",
            }


leaf_vision_scanner = LeafVisionAnalyzer()
