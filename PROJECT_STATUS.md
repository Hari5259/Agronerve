# AgroNerve — Project Status & Milestones 🌾

**Status:** Partially Committed Project (Foundational Offline Architecture Complete)  
**Version:** `v0.5.0-partial`  
**Target Repository:** [Hari5259/Agronerve](https://github.com/Hari5259/Agronerve)  

---

## 🚀 Completed Milestones (Phase 1: Foundational Advisory Architecture)

- [x] **Curated Multi-Domain Agricultural Datasets**
  - Disease symptoms, favorable conditions, and management protocols (ICAR / TNAU / IIHR standards).
  - Pesticide chemical formulations, active ingredients, exact dilution rates (ml/L or g/L), Pre-Harvest Intervals (PHI), and mandatory PPE rules (CIBRC compliance).
  - Weather forecast operations, operational spray/harvest windows, and pathogen risk triggers.
  - Irrigation crop-growth stage schedules and water requirement models (FAO-56 guidelines).
- [x] **Domain-Scoped RAG Pipeline**
  - ChromaDB persistent vector store partitioned into 4 domain collections (`disease_kb`, `pesticide_kb`, `weather_kb`, `irrigation_kb`).
  - Offline BM25/TF-IDF and semantic similarity ranking fallback.
- [x] **Two-Stage Intent Router**
  - Stage 1: Weighted discriminative keyword pattern matching with 1.5x margin verification.
  - Stage 2: Fast LLM fallback classification for ambiguous queries.
  - Achieves **95.0% accuracy** on the 40-query multi-domain evaluation benchmark.
- [x] **Dynamic Agent Orchestration Layer**
  - Automatic assembly of specialist agents (`Crop Disease Specialist`, `Pesticide & Dosage Specialist`, `Weather Advisory Specialist`, `Irrigation Planning Specialist`).
  - Domain-specific post-processing (safety warnings, KVK consultation notes, forecast recency disclaimers, water conservation tips).
- [x] **Interactive User Interface (Streamlit)**
  - Dynamic agent status badges with latency and engine indicators.
  - Grounding context inspector accordion showing raw ICAR/FAO citation chunks.
  - One-click sample query triggers.
  - Offline Agronomic Utility Calculators (Spray tank chemical dilution & crop water volume estimator).
  - Built-in Benchmark evaluation suite with confusion matrix viewer.
- [x] **Production REST API (FastAPI)**
  - Endpoints: `GET /`, `POST /api/query`, `POST /api/route`, `GET /api/domains`, `GET /api/knowledge`, `GET /api/benchmark`.
  - Swagger UI documentation at `/docs` and ReDoc at `/redoc`.
- [x] **Automated Test Suite**
  - Unit and integration tests covering router, RAG pipeline, chunking, and FastAPI endpoints (**13/13 passing**).

---

## 🔮 Next Roadmap Milestones (Phase 2 & Phase 3)

- [ ] **Multi-Domain Query Handling** (Joint cross-domain routing for queries spanning disease + weather + irrigation).
- [ ] **Regional Language Localization** (Hindi, Tamil, Telugu, Kannada support with multilingual embeddings).
- [ ] **Offline Voice Interface** (Whisper-based local Speech-to-Text and lightweight Text-to-Speech for non-literate farmers).
- [ ] **IoT & On-Device Soil Sensor Integration** (Bluetooth / USB soil moisture probe telemetry).
- [ ] **Opportunistic Background Synchronization** (Silent update synchronization when temporary connectivity is detected).
