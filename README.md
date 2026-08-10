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

---

## 📁 Repository Structure

```
Agronerve/
├── .gitignore
├── README.md
├── requirements.txt
├── api/
│   └── main.py                 # FastAPI backend server
├── ui/
│   └── app.py                  # Streamlit chat interface
├── core/
│   ├── intent_router.py        # Two-stage intent classification
│   ├── orchestrator.py         # Dynamic agent orchestration engine
│   └── rag_pipeline.py         # ChromaDB retrieval and context injection
├── domains/
│   ├── disease.py              # Disease diagnosis configuration & prompts
│   ├── pesticide.py            # Pesticide dosage calculation & rules
│   ├── weather.py              # Weather forecast reasoning
│   └── irrigation.py           # Soil-water requirement calculations
└── data/
    └── knowledge_base/         # Curated agricultural domain datasets
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
