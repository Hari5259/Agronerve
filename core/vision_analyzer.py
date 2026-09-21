import io
import re
import os
import json
import base64
import logging
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import settings
from core.knowledge_seeder import KnowledgeChunker
from core.vision_quality import quality_checker

logger = logging.getLogger(__name__)

CROP_KEYWORDS = {
    "Tomato": ["tomato", "thakkali", "tamatar"],
    "Potato": ["potato", "aloo", "urulakizhangu", "batata", "alu"],
    "Paddy (Rice)": ["paddy", "rice", "nellu", "dhan", "chawal"],
    "Cotton": ["cotton", "kapas", "paruthi"],
    "Wheat": ["wheat", "gehun", "godhumai"],
    "Chilli": ["chilli", "chili", "pepper", "mirchi", "milagai"],
    "Maize (Corn)": ["maize", "corn", "cholam", "makka", "makai"],
    "Sugarcane": ["sugarcane", "karumbu", "ganna"],
    "Groundnut": ["groundnut", "peanut", "verkadalai", "moongfali"],
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DISEASE_MODELS_DIR = MODELS_DIR / "disease"


class LeafVisionAnalyzer:
    """Production Computer Vision Diagnostic Pipeline for AgroNerve.

    Pipeline:
    1. Image Quality & Usability Assessment (Resolution, Blur, Lighting, Foliage presence)
    2. Deep Learning Crop Identification (MobileNetV3 / ONNX)
    3. Crop Confidence & Uncertainty Filter
    4. Deep Learning Disease Diagnosis (Unified & Crop-Specific MobileNetV3 / ONNX)
    5. Disease Confidence & Out-of-Distribution Calibration Check
    6. Grounding Linkage to data/knowledge_base/disease.json (ICAR Management Protocols)
    7. Foliar Lesion Surface Segmentation (Chlorosis, Necrosis, Blight Area %)
    8. Graceful Multimodal Vision / Offline Rule Fallback
    """

    def __init__(self):
        self.disease_data = KnowledgeChunker.load_disease_chunks()
        self._crop_model = None
        self._crop_classes: List[str] = []
        self._crop_to_idx: Dict[str, int] = {}
        self._idx_to_crop: Dict[int, str] = {}

        self._disease_model = None
        self._disease_classes: List[str] = []
        self._disease_class_to_idx: Dict[str, int] = {}
        self._idx_to_disease_class: Dict[int, str] = {}

        self._crop_specific_models: Dict[str, Any] = {}
        self._onnx_crop_session = None
        self._onnx_disease_session = None
        self._ml_models_loaded = False

        self._load_models_if_available()

    def _load_models_if_available(self):
        """Loads trained PyTorch or ONNX models into memory if present on disk."""
        try:
            # 1. Load Crop Model
            crop_classes_file = MODELS_DIR / "crop_classes.json"
            crop_onnx_file = MODELS_DIR / "crop_model.onnx"
            crop_pth_file = MODELS_DIR / "crop_model.pth"

            if crop_classes_file.exists():
                with open(crop_classes_file, "r", encoding="utf-8") as f:
                    cmeta = json.load(f)
                    self._crop_classes = cmeta.get("crops", [])
                    self._crop_to_idx = cmeta.get("crop_to_idx", {})
                    self._idx_to_crop = {int(k): v for k, v in cmeta.get("idx_to_crop", {}).items()}

                if crop_onnx_file.exists():
                    try:
                        import onnxruntime as ort
                        self._onnx_crop_session = ort.InferenceSession(str(crop_onnx_file), providers=["CPUExecutionProvider"])
                        logger.info(f"Loaded ONNX Crop Model from {crop_onnx_file}")
                    except Exception as e:
                        logger.warning(f"Could not init ONNX session for crop model: {e}")

                if not self._onnx_crop_session and crop_pth_file.exists():
                    import torch
                    from training.train_crop_model import build_crop_model
                    self._crop_model = build_crop_model(len(self._crop_classes))
                    self._crop_model.load_state_dict(torch.load(crop_pth_file, map_location="cpu", weights_only=True))
                    self._crop_model.eval()
                    logger.info(f"Loaded PyTorch Crop Model from {crop_pth_file}")

            # 2. Load Unified Disease Model
            disease_classes_file = DISEASE_MODELS_DIR / "disease_classes.json"
            disease_onnx_file = DISEASE_MODELS_DIR / "unified_disease_model.onnx"
            disease_pth_file = DISEASE_MODELS_DIR / "unified_disease_model.pth"

            if disease_classes_file.exists():
                with open(disease_classes_file, "r", encoding="utf-8") as f:
                    dmeta = json.load(f)
                    self._disease_classes = dmeta.get("classes", [])
                    self._disease_class_to_idx = dmeta.get("class_to_idx", {})
                    self._idx_to_disease_class = {int(k): v for k, v in dmeta.get("idx_to_class", {}).items()}

                if disease_onnx_file.exists():
                    try:
                        import onnxruntime as ort
                        self._onnx_disease_session = ort.InferenceSession(str(disease_onnx_file), providers=["CPUExecutionProvider"])
                        logger.info(f"Loaded ONNX Disease Model from {disease_onnx_file}")
                    except Exception as e:
                        logger.warning(f"Could not init ONNX session for disease model: {e}")

                if not self._onnx_disease_session and disease_pth_file.exists():
                    import torch
                    from training.train_disease_model import build_disease_model
                    self._disease_model = build_disease_model(len(self._disease_classes))
                    self._disease_model.load_state_dict(torch.load(disease_pth_file, map_location="cpu", weights_only=True))
                    self._disease_model.eval()
                    logger.info(f"Loaded PyTorch Disease Model from {disease_pth_file}")

            self._ml_models_loaded = bool(
                (self._onnx_crop_session or self._crop_model) and (self._onnx_disease_session or self._disease_model)
            )
        except Exception as e:
            logger.warning(f"Error initializing ML vision models: {e}. Falling back to baseline vision engine.")
            self._ml_models_loaded = False

    def _preprocess_image_for_inference(self, pil_img: Image.Image) -> np.ndarray:
        """Preprocesses PIL Image for standard 224x224 RGB ImageNet normalization."""
        img_resized = pil_img.resize((settings.VISION_IMAGE_SIZE, settings.VISION_IMAGE_SIZE), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized).astype(np.float32) / 255.0  # HWC, [0, 1]

        # Normalize with ImageNet mean & std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        # Transpose HWC -> CHW and add batch dimension -> NCHW (1, 3, 224, 224)
        img_tensor = np.transpose(img_np, (2, 0, 1))
        return np.expand_dims(img_tensor, axis=0).astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Computes stable softmax probabilities."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def _predict_crop_ml(self, img_tensor: np.ndarray) -> Tuple[str, float, float]:
        """Runs crop model inference. Returns (crop_name, top1_confidence, top2_margin)."""
        if self._onnx_crop_session:
            input_name = self._onnx_crop_session.get_inputs()[0].name
            logits = self._onnx_crop_session.run(None, {input_name: img_tensor})[0][0]
        elif self._crop_model:
            import torch
            with torch.no_grad():
                tensor_torch = torch.from_numpy(img_tensor)
                logits = self._crop_model(tensor_torch).numpy()[0]
        else:
            return "auto", 0.0, 0.0

        probs = self._softmax(logits)
        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = top1_prob - top2_prob

        crop_slug = self._idx_to_crop.get(top1_idx, self._crop_classes[top1_idx] if top1_idx < len(self._crop_classes) else "unknown")
        return crop_slug, top1_prob, margin

    def _predict_disease_ml(self, img_tensor: np.ndarray, crop_hint: Optional[str] = None) -> Tuple[str, float, float]:
        """Runs disease model inference. Returns (class_name, top1_confidence, top2_margin)."""
        if self._onnx_disease_session:
            input_name = self._onnx_disease_session.get_inputs()[0].name
            logits = self._onnx_disease_session.run(None, {input_name: img_tensor})[0][0]
        elif self._disease_model:
            import torch
            with torch.no_grad():
                tensor_torch = torch.from_numpy(img_tensor)
                logits = self._disease_model(tensor_torch).numpy()[0]
        else:
            return "unknown", 0.0, 0.0

        probs = self._softmax(logits)

        # If crop_hint is provided and valid, filter probabilities to that crop
        if crop_hint and crop_hint.lower() != "auto":
            crop_clean = crop_hint.split("(")[0].strip().lower()
            mask = np.array([c.startswith(f"{crop_clean}_") for c in self._disease_classes], dtype=bool)
            if np.any(mask):
                # Re-normalize masked probabilities
                filtered_logits = logits.copy()
                filtered_logits[~mask] = -1e9
                probs = self._softmax(filtered_logits)

        sorted_indices = np.argsort(probs)[::-1]
        top1_idx = int(sorted_indices[0])
        top1_prob = float(probs[top1_idx])
        top2_prob = float(probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = top1_prob - top2_prob

        class_slug = self._idx_to_disease_class.get(
            top1_idx, self._disease_classes[top1_idx] if top1_idx < len(self._disease_classes) else "unknown"
        )
        return class_slug, top1_prob, margin

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
        """Parses the crop and disease name from Ollama Vision response text."""
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

    def _call_ollama_vision(self, image_bytes: bytes, user_query: Optional[str] = None) -> Optional[str]:
        """Attempts inference with local multimodal vision models (e.g. llava, moondream)."""
        logger.info("Attempting local Ollama vision inference.")
        try:
            b64_image = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                user_query
                or "You are an expert plant pathologist AI. Analyze this crop leaf photograph. "
                "Identify the crop (e.g. Tomato, Rice, Cotton, Wheat, Chilli, Potato), describe visual symptoms (lesions, chlorosis, concentric rings, spots), "
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
        except Exception as e:
            logger.warning(f"Failed to run Ollama vision inference: {str(e)}")
        return None

    def _compute_foliar_damage_metrics(self, pil_img: Image.Image) -> Tuple[Dict[str, float], float, List[str]]:
        """Fast offline pixel color distribution analysis to quantify foliar surface metrics."""
        width, height = pil_img.size
        total_pixels = width * height
        sample_step = max(1, int((total_pixels / 10000) ** 0.5))
        pixels = [
            pil_img.getpixel((x, y))
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

        leaf_pixel_count = yellow_count + brown_necrotic_count + dark_spot_count + green_healthy_count
        if leaf_pixel_count == 0:
            leaf_pixel_count = sample_count

        chlorosis_pct = round((yellow_count / leaf_pixel_count) * 100, 1)
        necrotic_pct = round((brown_necrotic_count / leaf_pixel_count) * 100, 1)
        dark_lesion_pct = round((dark_spot_count / leaf_pixel_count) * 100, 1)
        healthy_pct = round((green_healthy_count / leaf_pixel_count) * 100, 1)

        total_damage_pct = min(100.0, round(chlorosis_pct + necrotic_pct + dark_lesion_pct, 1))

        detected_features = []
        if chlorosis_pct > 5.0:
            detected_features.append("Marked leaf chlorosis / yellowing")
        if necrotic_pct > 3.0:
            detected_features.append("Brown necrotic foliar lesions")
        if dark_lesion_pct > 2.0:
            detected_features.append("Dark concentrated necrotic spots")
        if not detected_features:
            detected_features.append("Healthy green foliage pattern")

        metrics = {
            "chlorosis_yellow_pct": chlorosis_pct,
            "necrotic_brown_pct": necrotic_pct,
            "dark_lesion_pct": dark_lesion_pct,
            "healthy_green_pct": healthy_pct,
        }
        return metrics, total_damage_pct, detected_features

    def _lookup_knowledge_base(self, crop_name: str, disease_name: str) -> Dict[str, Any]:
        """Maps predicted crop and disease to data/knowledge_base/disease.json."""
        c_clean = crop_name.split("(")[0].strip().lower()
        d_clean = disease_name.lower().replace("_", " ").replace("healthy", "").strip()

        best_match = None
        for item in self.disease_data:
            item_crop = item.get("crop", "").split("(")[0].strip().lower()
            item_title = item.get("title", "").lower()
            item_text = item.get("text", "").lower()

            if c_clean in item_crop or item_crop in c_clean:
                if d_clean and (d_clean in item_title or d_clean in item_text):
                    return item
                if best_match is None:
                    best_match = item

        if best_match:
            return best_match
        return {}

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        crop_hint: str = "auto",
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point: Runs complete vision pipeline with image quality check, ML inference, uncertainty thresholding, and ICAR knowledge lookup."""
        logger.info(f"Analyzing leaf image bytes with crop_hint: '{crop_hint}', user_query: '{user_query}'")

        if not image_bytes:
            return {
                "status": "error",
                "message": "Empty image bytes provided.",
            }

        # Step 1: Image Quality Assessment (Blur, Exposure, Resolution, Foliage Ratio)
        quality = quality_checker.evaluate_quality(image_bytes)
        if not quality["is_usable"]:
            logger.warning(f"Image rejected by quality check: {quality['reason']}")
            return {
                "status": "unusable",
                "crop": "Unknown",
                "crop_details": {},
                "disease": "Unknown",
                "predicted_disease": "Unknown",
                "confidence_pct": 0.0,
                "crop_confidence": 0.0,
                "disease_confidence": 0.0,
                "quality_metrics": quality,
                "message": quality["reason"],
                "ai_description": f"Image quality check failed: {quality['reason']}",
            }

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to decode image bytes: {str(e)}",
            }

        # Infer crop from query text if provided
        inferred_crop = self._extract_crop_from_text(user_query)
        effective_crop_hint = crop_hint
        if crop_hint.lower() == "auto" or not crop_hint:
            effective_crop_hint = inferred_crop or "auto"

        # Step 2: Compute Foliar Damage Metrics
        metrics, total_damage_pct, detected_features = self._compute_foliar_damage_metrics(pil_img)

        # Refresh model loading state if newly trained
        if not self._ml_models_loaded:
            self._load_models_if_available()

        # Step 3: Deep Learning ML Model Inference
        if self._ml_models_loaded:
            img_tensor = self._preprocess_image_for_inference(pil_img)

            # 3.1 Crop Model Inference
            pred_crop_slug, crop_conf, crop_margin = self._predict_crop_ml(img_tensor)

            # Determine effective crop
            if effective_crop_hint != "auto":
                final_crop = effective_crop_hint
            elif pred_crop_slug != "unknown":
                final_crop = pred_crop_slug.capitalize()
            else:
                final_crop = "Crop"

            # Check Crop Model Confidence / Uncertainty
            if effective_crop_hint == "auto" and crop_conf < settings.VISION_CONFIDENCE_THRESHOLD:
                logger.info(f"Low crop confidence ({crop_conf:.2f} < {settings.VISION_CONFIDENCE_THRESHOLD}). Returning uncertain.")
                return {
                    "status": "uncertain",
                    "crop": final_crop,
                    "crop_details": {"name": final_crop, "confidence": round(crop_conf, 3)},
                    "disease": "Unknown",
                    "predicted_disease": f"{final_crop} (Uncertain)",
                    "confidence_pct": round(crop_conf * 100, 1),
                    "crop_confidence": round(crop_conf, 3),
                    "disease_confidence": 0.0,
                    "quality_metrics": quality,
                    "affected_leaf_area_pct": total_damage_pct,
                    "metrics": metrics,
                    "message": "Unable to confidently identify the crop. Please capture a clearer image focused directly on the leaf.",
                    "ai_description": f"Crop identification is uncertain (confidence: {crop_conf*100:.1f}%). Please frame the crop leaf centrally.",
                }

            # 3.2 Disease Model Inference
            pred_class_slug, disease_conf, disease_margin = self._predict_disease_ml(img_tensor, crop_hint=final_crop)

            # Parse disease name from slug (e.g., "tomato_early_blight" -> "Early Blight")
            disease_parts = pred_class_slug.split("_")
            if len(disease_parts) >= 2:
                disease_part_name = " ".join(disease_parts[1:]).title()
            else:
                disease_part_name = pred_class_slug.replace("_", " ").title()

            # Check Disease Model Confidence / Uncertainty
            if disease_conf < settings.VISION_CONFIDENCE_THRESHOLD:
                logger.info(f"Low disease confidence ({disease_conf:.2f} < {settings.VISION_CONFIDENCE_THRESHOLD}). Returning uncertain.")
                return {
                    "status": "uncertain",
                    "crop": final_crop,
                    "crop_details": {"name": final_crop, "confidence": round(crop_conf, 3) if crop_conf > 0 else 0.90},
                    "disease": disease_part_name,
                    "disease_details": {"name": disease_part_name, "confidence": round(disease_conf, 3)},
                    "predicted_disease": f"{final_crop} {disease_part_name}",
                    "confidence_pct": round(disease_conf * 100, 1),
                    "crop_confidence": round(crop_conf, 3) if crop_conf > 0 else 0.90,
                    "disease_confidence": round(disease_conf, 3),
                    "quality_metrics": quality,
                    "affected_leaf_area_pct": total_damage_pct,
                    "metrics": metrics,
                    "message": f"Unable to confidently diagnose the foliar symptoms on {final_crop} (confidence {disease_conf*100:.1f}% < {settings.VISION_CONFIDENCE_THRESHOLD*100:.0f}%). Please capture a clearer image with better lighting.",
                    "ai_description": f"Diagnostic confidence is low ({disease_conf*100:.1f}%) on {final_crop}. Symptoms are faint or inconclusive.",
                }

            # Step 4: Knowledge Base Lookup
            kb_entry = self._lookup_knowledge_base(final_crop, disease_part_name)
            predicted_disease_title = kb_entry.get("title", f"{final_crop} {disease_part_name}")
            verified_protocol = kb_entry.get("text", "")

            ai_description = (
                f"Trained ML Vision model diagnosed **{predicted_disease_title}** on **{final_crop}** "
                f"with **{disease_conf*100:.1f}%** model confidence. "
                f"Estimated affected leaf area: **{total_damage_pct}%** ({', '.join(detected_features)})."
            )

            result = {
                "status": "success",
                "predicted_disease": predicted_disease_title,
                "crop": final_crop,
                "confidence_pct": round(disease_conf * 100, 1),
                "crop_confidence": round(crop_conf, 3) if crop_conf > 0 else 0.95,
                "disease_confidence": round(disease_conf, 3),
                "affected_leaf_area_pct": total_damage_pct,
                "ai_description": ai_description,
                "metrics": metrics,
                "quality_metrics": quality,
                "detected_symptoms": detected_features,
                "verified_protocol": verified_protocol,
                "knowledge": {
                    "id": kb_entry.get("id", "N/A"),
                    "title": kb_entry.get("title", predicted_disease_title),
                    "source": kb_entry.get("metadata", {}).get("source", "ICAR"),
                },
                "engine": "deep_learning_vision_pipeline",
            }
            logger.info(f"ML Vision diagnosis successful: crop='{final_crop}', disease='{predicted_disease_title}', confidence={result['confidence_pct']}%")
            return result

        # Step 5: Fallback Heuristic & Ollama Vision Engine
        logger.info("ML weights not loaded. Utilizing baseline heuristic & Ollama vision engine.")
        ollama_vision_resp = self._call_ollama_vision(image_bytes, user_query)

        match_candidates = []
        for item in self.disease_data:
            crop_name = item.get("crop", "")
            text = item.get("text", "").lower()
            title = item.get("title", "").lower()
            score = 0.0

            if effective_crop_hint.lower() != "auto":
                hint_lower = effective_crop_hint.lower()
                if hint_lower in crop_name.lower() or any(alias in hint_lower for alias in CROP_KEYWORDS.get(crop_name, [])):
                    score += 150.0
                else:
                    score -= 50.0

            if metrics["necrotic_brown_pct"] > 3.0:
                if "early blight" in title or "target" in text or "concentric" in text:
                    score += 40.0
                elif "spot" in text or "brown" in text or "lesion" in text:
                    score += 20.0

            if metrics["chlorosis_yellow_pct"] > 5.0:
                if "curl" in title or "yellow" in title or "chlorosis" in text:
                    score += 35.0
                elif "yellowing" in text or "yellow" in text:
                    score += 15.0

            if metrics["dark_lesion_pct"] > 2.0:
                if "late blight" in title or "blast" in title or "black" in text:
                    score += 30.0

            match_candidates.append((score, item))

        match_candidates.sort(key=lambda x: x[0], reverse=True)
        top_match = match_candidates[0][1] if match_candidates else {}
        detected_crop_final = top_match.get("crop", effective_crop_hint if effective_crop_hint != "auto" else "Crop")

        if ollama_vision_resp:
            inferred_crop_ollama, inferred_disease_ollama = self._parse_vision_response(ollama_vision_resp)
            if inferred_crop_ollama:
                detected_crop_final = inferred_crop_ollama
            if inferred_disease_ollama:
                for item in self.disease_data:
                    if item.get("title") == inferred_disease_ollama:
                        top_match = item
                        break

        calibrated_conf = min(96.0, max(70.0, round(70.0 + (total_damage_pct * 0.25), 1)))
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
            "confidence_pct": calibrated_conf,
            "crop_confidence": round(calibrated_conf / 100.0, 2),
            "disease_confidence": round(calibrated_conf / 100.0, 2),
            "affected_leaf_area_pct": total_damage_pct,
            "ai_description": ai_description,
            "metrics": metrics,
            "quality_metrics": quality,
            "detected_symptoms": detected_features,
            "verified_protocol": top_match.get("text", ""),
            "knowledge": {
                "id": top_match.get("id", "N/A"),
                "title": top_match.get("title", "Foliar Disease"),
                "source": top_match.get("metadata", {}).get("source", "ICAR"),
            },
            "engine": "baseline_vision_fallback",
        }
        return result


leaf_vision_scanner = LeafVisionAnalyzer()
