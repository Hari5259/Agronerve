# AgroNerve 🌾

**An Intelligent Offline Agricultural Advisory System Using LLM-Driven Dynamic Agent Orchestration and Retrieval-Augmented Generation (RAG)**

---

## 📌 Overview

**AgroNerve** is a completely offline, edge-capable agricultural advisory platform designed to deliver expert-level crop guidance to farmers in low-connectivity and rural environments. By leveraging local LLMs served through **Ollama** and a structured vector knowledge base via **ChromaDB**, AgroNerve bypasses the need for active internet connectivity while delivering accurate, domain-grounded advice.

### 🌟 Key Highlights
- **100% Offline Capability:** Runs locally with zero external API dependencies post-installation.
- **Dynamic Agent Orchestration:** Automatically identifies user intent (via a two-stage hybrid classifier) and dynamically assembles domain-specific agents on the fly.
- **Retrieval-Augmented Generation (RAG):** Contextually grounds model responses in curated agricultural databases (ICAR guidelines, FAO standards, extension bulletins).
- **Lightweight Resource Footprint:** Optimized for consumer-grade hardware and local edge devices.

---

## 🏛️ System Architecture

```
                                  [ Farmer Query ]
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │ 2-Stage Intent Router   │
                            │ (Keywords + Fast LLM)   │
                            └────────────┬────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
  [ Disease Diagnosis ]       [ Pesticide & Dosage ]       [ Weather / Irrigation ]
            │                            │                            │
            └────────────────────────────┼────────────────────────────┘
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
                            [ Grounded Advisory Resp ]
```

---

## 🧩 Advisory Domains

1. **🌿 Crop Disease Identification:**
   - Symptom-to-disease mapping, diagnostic questioning, and pathogen identification.
2. **🧪 Pesticide & Pest Control:**
   - Safe dosage calculations, chemical vs. organic control options, safety protocols, and pre-harvest intervals.
3. **🌦️ Weather-Aware Advisory:**
   - Works with cached forecast data and local sensor readings with recency transparency.
4. **💧 Irrigation Scheduling:**
   - Soil-crop-water requirement models (FAO guidelines) for daily watering planning.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
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

3. **Run the UI (Streamlit):**
   ```bash
   streamlit run ui/app.py
   ```

4. **Run the API (FastAPI):**
   ```bash
   uvicorn api.main:app --reload --port 8000
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
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── config.py                   # App configuration & environment settings
├── api/
│   └── main.py                 # FastAPI backend server with Swagger docs
├── ui/
│   └── app.py                  # Streamlit multi-tab advisory dashboard
├── core/
│   ├── intent_router.py        # Two-stage intent classification engine
│   ├── knowledge_seeder.py     # Agricultural dataset chunker & preprocessor
│   ├── orchestrator.py         # Dynamic agent orchestration engine
│   └── rag_pipeline.py         # ChromaDB retrieval and fallback ranker
├── domains/
│   ├── disease.py              # Crop disease diagnosis profile & prompt
│   ├── pesticide.py            # Pesticide dosage calculation & safety rules
│   ├── weather.py              # Weather forecast reasoning & operation windows
│   └── irrigation.py           # FAO-56 crop water requirement models
├── data/
│   └── knowledge_base/         # ICAR, FAO, and CIBRC domain JSON datasets
├── evaluation/
│   ├── benchmark.py            # Automated accuracy & confusion matrix runner
│   └── test_queries.json       # 40-query multi-domain evaluation benchmark
└── tests/
    ├── test_api.py             # FastAPI endpoint integration tests
    ├── test_rag.py             # Knowledge retrieval & chunking unit tests
    └── test_router.py          # Intent classification & threshold tests
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
