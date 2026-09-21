# AgroNerve Live Camera Scanner & Extension Guide

This guide covers live camera operation in the Streamlit UI, REST API usage, and instructions for adding new crops and diseases.

---

## 1. Running the Live Camera Scanner

Launch the Streamlit web application:

```powershell
streamlit run ui/app.py
```

### In Tab 1 (Advisory Chat):
1. Expand **Attach / Capture Crop Leaf Photo for AI Diagnosis**.
2. Select **📸 Live Camera**.
3. Allow browser camera access and capture a live leaf photo.
4. The system will diagnose the plant pathology, display the real model confidence, and allow you to ask follow-up questions in natural language with full session memory.

### In Tab 2 (Visual Leaf Scanner):
1. Select **📸 Live Camera Capture**.
2. Capture a photo of the affected crop leaf.
3. The UI will instantly display:
   - **Diagnosis Title & Identified Crop**
   - **Real ML Model Confidence %**
   - **Foliar Damage Breakdown** (Chlorosis %, Necrosis %, Dark spots %)
   - **Image Quality Diagnostics** (Sharpness variance, luminance, foliage %)
   - **Verified ICAR Management Protocols**

---

## 2. Using the REST API

Start the FastAPI server:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoint: `POST /api/scan-leaf`

**Request:**
- `file`: Multipart image file (JPG, PNG, WebP)
- `crop_hint` (optional): Crop name filter (e.g. `Tomato`, `Paddy`, or `auto`)

**Sample JSON Response (Success):**
```json
{
  "status": "success",
  "crop": {
    "name": "Tomato",
    "confidence": 0.94
  },
  "disease": {
    "name": "Early Blight",
    "confidence": 0.91
  },
  "predicted_disease": "Early Blight (Alternaria solani)",
  "confidence_pct": 91.0,
  "severity": "unknown",
  "affected_leaf_area_pct": 14.5,
  "metrics": {
    "chlorosis_yellow_pct": 8.2,
    "necrotic_brown_pct": 6.3,
    "dark_lesion_pct": 0.0,
    "healthy_green_pct": 85.5
  },
  "quality_metrics": {
    "is_usable": true,
    "blur_score": 142.5,
    "luminance": 118.4,
    "foliage_ratio": 0.72
  },
  "knowledge": {
    "id": "DIS_006",
    "title": "Early Blight (Alternaria solani)",
    "source": "ICAR - Indian Institute of Horticultural Research (IIHR)"
  },
  "verified_protocol": "Spray Mancozeb 75% WP @ 2.5 g/L...",
  "engine": "deep_learning_vision_pipeline"
}
```

**Sample JSON Response (Uncertain / Low Confidence):**
```json
{
  "status": "uncertain",
  "crop": {
    "name": "Tomato",
    "confidence": 0.45
  },
  "disease": {},
  "confidence_pct": 45.0,
  "message": "Unable to confidently identify the crop/disease. Please capture a clearer image."
}
```

---

## 3. How to Add a New Crop

1. **Add Disease Entries to Knowledge Base**:
   Open `data/knowledge_base/disease.json` and append the new crop's disease definitions (e.g., Maize, Groundnut, Soybean):
   ```json
   {
     "id": "DIS_013",
     "crop": "Maize",
     "disease_name": "Common Rust (Puccinia sorghi)",
     "symptoms": "Small powdery cinnamon-brown pustules...",
     "favorable_conditions": "Moderate temperatures (16-25°C)...",
     "management": "Spray Azoxystrobin @ 1 ml/L...",
     "source": "ICAR - Indian Institute of Maize Research"
   }
   ```

2. **Add Alias Keywords**:
   In `core/vision_analyzer.py`, add regional aliases to `CROP_KEYWORDS`:
   ```python
   "Maize": ["maize", "corn", "makka", "cholam"]
   ```

3. **Ingest Training Images**:
   Place training images in `data/raw_datasets/maize_common_rust/` and `data/raw_datasets/maize_healthy/`, then run:
   ```powershell
   python training/download_dataset.py
   python training/prepare_dataset.py
   ```

4. **Retrain Models**:
   ```powershell
   python training/train_crop_model.py
   python training/train_disease_model.py
   ```

---

## 4. How the ML Model Connects to `disease.json`

1. The ML model outputs a machine-readable class slug: `tomato_early_blight`.
2. The pipeline decomposes this into `crop = "Tomato"` and `disease = "Early Blight"`.
3. `LeafVisionAnalyzer._lookup_knowledge_base()` searches `data/knowledge_base/disease.json` for matching records.
4. The exact symptoms, favorable weather conditions, ICAR management protocols, and source citations are retrieved and supplied to both the UI and conversational RAG engine.
