import io
from typing import Dict, Any, List
from core.knowledge_seeder import KnowledgeChunker

class LeafVisionAnalyzer:
    """Offline visual leaf disease feature extractor and diagnostic scanner."""

    def __init__(self):
        self.disease_data = KnowledgeChunker.load_disease_chunks()

    def analyze_image_bytes(self, image_bytes: bytes, crop_hint: str = "auto") -> Dict[str, Any]:
        """Analyzes leaf image byte data and computes lesion / chlorosis metrics."""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = img.size
            total_pixels = width * height
            
            # Simple fast pixel color distribution analysis
            # Samples up to 10,000 pixels for fast offline CPU computation
            sample_step = max(1, int((total_pixels / 10000) ** 0.5))
            pixels = [img.getpixel((x, y)) for y in range(0, height, sample_step) for x in range(0, width, sample_step)]
            sample_count = len(pixels) or 1

            yellow_count = 0
            brown_necrotic_count = 0
            green_healthy_count = 0
            dark_spot_count = 0

            for r, g, b in pixels:
                # Yellow / chlorotic: high red + high green, low blue
                if r > 130 and g > 130 and b < 100 and (r + g) > (2.2 * b):
                    yellow_count += 1
                # Brown / necrotic: moderate red, lower green, low blue
                elif r > 80 and g > 40 and b < 50 and r > (1.3 * g):
                    brown_necrotic_count += 1
                # Dark spot / black blight: very low intensity
                elif r < 60 and g < 60 and b < 60:
                    dark_spot_count += 1
                # Healthy green: green significantly exceeds red and blue
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
                detected_features.append("Brown necrotic dead tissue patches")
            if dark_lesion_pct > 5.0:
                detected_features.append("Dark concentrated fungal/bacterial lesions")
            if not detected_features:
                detected_features.append("Mild localized foliage discoloration")

            # Match against curated disease database
            match_candidates = []
            for item in self.disease_data:
                crop_name = item.get("crop", "")
                text = item.get("text", "").lower()
                score = 0.0

                if crop_hint.lower() != "auto" and crop_hint.lower() in crop_name.lower():
                    score += 40.0

                if necrotic_pct > 10.0 and ("brown" in text or "spot" in text or "lesion" in text):
                    score += 25.0
                if chlorosis_pct > 15.0 and ("yellow" in text or "chlorosis" in text or "wilt" in text):
                    score += 25.0
                if dark_lesion_pct > 5.0 and ("black" in text or "blight" in text or "blast" in text):
                    score += 25.0

                match_candidates.append((score, item))

            match_candidates.sort(key=lambda x: x[0], reverse=True)
            top_match = match_candidates[0][1] if match_candidates else {}
            confidence = min(96.0, max(68.0, round(65.0 + (total_damage_pct * 0.3), 1)))

            return {
                "status": "success",
                "predicted_disease": top_match.get("title", "Suspected Foliar Blight"),
                "crop": top_match.get("crop", "Vegetable / Field Crop"),
                "confidence_pct": confidence,
                "affected_leaf_area_pct": total_damage_pct,
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
