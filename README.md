# AgroNerve 🌾

**An Intelligent Offline Agricultural Advisory System Using LLM-Driven Dynamic Agent Orchestration, Computer Vision, IoT Telemetry, and Retrieval-Augmented Generation (RAG)**

---

## 📌 Overview

**AgroNerve** is an edge-capable agricultural advisory platform designed to deliver expert crop guidance to farmers in low-connectivity and rural environments. By leveraging local LLMs served through **Ollama** and a structured vector knowledge base via **ChromaDB**, AgroNerve bypasses the need for active internet connectivity while delivering accurate, domain-grounded advice.

### 🌟 Key Features
- **100% Offline Capability:** Runs locally on consumer hardware with zero external API dependencies post-installation.
- **Dynamic Agent Orchestration:** Automatically identifies intent (via a two-stage hybrid classifier) and dynamically assembles domain specialists on the fly.
- **Multi-Domain Composite Reasoning:** Simultaneously queries multiple knowledge partitions when farmer questions span diseases, weather, and irrigation.
- **📸 Visual Leaf Disease Scanner:** Upload or capture leaf photos to quantify chlorosis, necrotic lesion %, and receive verified ICAR treatment protocols.
- **📡 IoT Soil & Weather Telemetry:** Live Volumetric Water Content (VWC %) soil moisture tracking with automated drought alerts and fungal risk warnings.
- **🔊 Voice Speech Synthesis (TTS):** Spoken audio readout for non-literate farmers.
- **🌐 Regional Language Support:** Multilingual UI and prompts for English, Tamil (தமிழ்), Hindi (हिन्दी), Telugu (తెలుగు), and Kannada (ಕನ್ನಡ).
- **Retrieval-Augmented Generation (RAG):** Contextually grounds model responses in curated agricultural databases (ICAR guidelines, FAO standards, CIBRC lists).

---

## 🏛️ System Architecture

```
                                  [ Farmer Query / Leaf Photo / IoT Telemetry ]
                                                        │
                                                        ▼
                                           ┌─────────────────────────┐
                                           │ Multi-Domain Classifier │
                                           │ (Keywords + Fast LLM)   │
                                           └────────────┬────────────┘
                                                        │
            ┌───────────────────────────┬───────────────┴───────────────┬───────────────────────────┐
            ▼                           ▼                               ▼                           ▼
  [ Disease Diagnosis ]       [ Pesticide & Dosage ]       [ Weather Advisory ]       [ Irrigation Planning ]
            │                           │                               │                           │
            └───────────────────────────┼───────────────────────────────┴───────────────────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │ Domain Partition Scope  │
                           │  (ChromaDB Vector Store)│
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │ Prompt Assembly Layer   │
                           │ (System + Context + Q)  │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           ┌─────────────────────────┐
                           │ Local LLM (Ollama)      │
                           └────────────┬────────────┘
                                        │
                                        ▼
                           [ Grounded Advisory Resp + Audio TTS ]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed locally (optional, built-in offline engine included)
- Pull recommended local models:
  ```bash
  ollama pull mistral
  ollama pull nomic-embed-text
  ```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hari5259/Agronerve.git
   cd Agronerve
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the UI (Streamlit Dashboard):**
   ```bash
   streamlit run ui/app.py
   # Accessible at http://localhost:8501
   ```

4. **Run the REST API (FastAPI):**
   ```bash
   uvicorn api.main:app --reload --port 8000
   # Swagger Docs at http://127.0.0.1:8000/docs
   ```

5. **Run the Evaluation Benchmark:**
   ```bash
   python evaluation/benchmark.py
   ```

6. **Run Unit & Integration Tests:**
   ```bash
   pytest -v
   ```

---

## 📁 Repository Structure

```
Agronerve/
├── .gitignore
├── LICENSE                     # MIT License
├── PROJECT_STATUS.md           # Completed milestones & capability tracking
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── config.py                   # App configuration & environment settings
├── api/
│   └── main.py                 # FastAPI backend server with Swagger docs
├── ui/
│   └── app.py                  # Streamlit 6-tab advisory dashboard
├── core/
│   ├── intent_router.py        # Two-stage intent classification engine
│   ├── multi_domain_router.py  # Compound cross-domain intent analyzer
│   ├── knowledge_seeder.py     # Agricultural dataset chunker & preprocessor
│   ├── orchestrator.py         # Dynamic agent orchestration engine
│   ├── rag_pipeline.py         # ChromaDB retrieval and fallback ranker
│   ├── vision_analyzer.py      # Leaf visual lesion & chlorosis analyzer
│   ├── sensor_telemetry.py     # IoT soil moisture & weather telemetry engine
│   ├── translator.py           # Multilingual localization adapter
│   └── voice_engine.py         # Offline speech synthesis and text cleaner
├── domains/
│   ├── disease.py              # Crop disease diagnosis profile & prompt
│   ├── pesticide.py            # Pesticide dosage calculation & safety rules
│   ├── weather.py              # Weather forecast reasoning & operation windows
│   └── irrigation.py           # FAO-56 crop water requirement models
├── data/
│   ├── knowledge_base/         # ICAR, FAO, and CIBRC domain JSON datasets
│   └── translations/           # English, Tamil, Hindi, Telugu, Kannada dictionaries
├── evaluation/
│   ├── benchmark.py            # Automated accuracy & confusion matrix runner
│   └── test_queries.json       # 40-query multi-domain evaluation benchmark
└── tests/
    ├── test_api.py             # FastAPI endpoint integration tests
    ├── test_multi_domain.py    # Compound query router tests
    ├── test_rag.py             # Knowledge retrieval & chunking unit tests
    ├── test_router.py          # Intent classification & threshold tests
    ├── test_sensor.py          # IoT telemetry & threshold alert tests
    ├── test_vision.py          # Leaf visual diagnostic scanner tests
    └── test_voice_i18n.py      # Voice text cleaning & localization tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
