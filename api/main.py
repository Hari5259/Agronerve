from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.orchestrator import AgentOrchestrator, DOMAIN_CONFIGS
from core.knowledge_seeder import KnowledgeChunker
from core.intent_router import IntentRouter
from core.multi_domain_router import MultiDomainRouter
from core.session_manager import session_manager
from core.vision_analyzer import leaf_vision_scanner
from core.sensor_telemetry import sensor_manager
from core.translator import language_manager, SUPPORTED_LANGUAGES
from core.voice_engine import voice_engine
from core.agri_calculator import agri_calculator
from core.safety_guardrails import safety_guardrails
from evaluation.benchmark import AgroNerveBenchmark
from config import settings

app = FastAPI(
    title="AgroNerve Multimodal Agricultural Advisory API",
    description="Offline Agricultural Advisory System API with Multimodal Vision, Dynamic Agent Orchestration & Session Continuity",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AgentOrchestrator()
router = IntentRouter()
multi_router = MultiDomainRouter()
benchmark_runner = AgroNerveBenchmark()


# Schemas
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        json_schema_extra={
            "example": "What is the recommended dosage of Chlorantraniliprole 18.5 SC for stem borer in paddy?"
        },
    )
    session_id: Optional[str] = Field(
        "default", json_schema_extra={"example": "session_123"}
    )
    language: Optional[str] = Field("en", json_schema_extra={"example": "en"})


class QueryResponse(BaseModel):
    session_id: Optional[str] = "default"
    query: str
    domain: str
    active_domains: Optional[List[str]] = None
    is_multi_domain: Optional[bool] = False
    agent_name: str
    response: str
    chunks_retrieved: int
    engine: str
    latency_seconds: float
    route_meta: Dict[str, Any]
    context_preview: str


class RouteRequest(BaseModel):
    query: str


class SpeechCleanRequest(BaseModel):
    text: str
    language: Optional[str] = "en"


class SprayDosageRequest(BaseModel):
    area_acres: float = Field(..., gt=0.0, json_schema_extra={"example": 2.5})
    dose_per_liter: float = Field(..., gt=0.0, json_schema_extra={"example": 0.4})
    unit: Optional[str] = Field("ml", json_schema_extra={"example": "ml"})
    tank_capacity_liters: Optional[float] = Field(16.0, json_schema_extra={"example": 16.0})
    water_volume_per_acre: Optional[float] = Field(200.0, json_schema_extra={"example": 200.0})


class WaterRequirementRequest(BaseModel):
    crop: str = Field(..., json_schema_extra={"example": "tomato"})
    stage: Optional[str] = Field("mid", json_schema_extra={"example": "mid"})
    area_acres: Optional[float] = Field(1.0, gt=0.0, json_schema_extra={"example": 1.0})
    eto_mm_day: Optional[float] = Field(4.5, gt=0.0, json_schema_extra={"example": 4.5})
    irrigation_efficiency: Optional[float] = Field(0.85, gt=0.0, le=1.0, json_schema_extra={"example": 0.85})


class FertilizerNPKRequest(BaseModel):
    n_kg: float = Field(..., ge=0.0, json_schema_extra={"example": 50.0})
    p_kg: float = Field(..., ge=0.0, json_schema_extra={"example": 25.0})
    k_kg: float = Field(..., ge=0.0, json_schema_extra={"example": 25.0})


class SafetyAuditRequest(BaseModel):
    advisory_text: str = Field(..., json_schema_extra={"example": "Spray Chlorantraniliprole 18.5 SC at 0.4 ml/L for stem borer."})


class SensorTelemetryUpdate(BaseModel):
    soil_moisture: float = Field(
        ..., ge=0.0, le=100.0, json_schema_extra={"example": 35.5}
    )
    ambient_temp: float = Field(..., json_schema_extra={"example": 28.5})
    humidity: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 65.0})
    rain: bool = Field(..., json_schema_extra={"example": False})


@app.get("/", tags=["Health"])
def health_check():
    """Returns AgroNerve service health and offline readiness."""
    total_chunks = len(KnowledgeChunker.get_all_chunks())
    return {
        "name": settings.APP_NAME,
        "status": "healthy",
        "mode": "offline-edge-ready",
        "version": "1.1.0",
        "multimodal_vision_enabled": True,
        "session_memory_enabled": True,
        "knowledge_base_chunks": total_chunks,
        "supported_domains": list(DOMAIN_CONFIGS.keys()),
        "supported_languages": SUPPORTED_LANGUAGES,
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Advisory"])
def process_agricultural_query(req: QueryRequest):
    """Processes a natural-language farmer query through dynamic agent assembly with session memory."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = orchestrator.process_query(
        req.query, session_id=req.session_id or "default", language=req.language or "en"
    )
    return result


@app.post("/api/chat/multimodal", tags=["Multimodal Chat"])
async def process_multimodal_chat(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    session_id: Optional[str] = Form("default"),
    language: Optional[str] = Form("en"),
):
    """Accepts an uploaded leaf photograph along with an optional user query, performs AI diagnosis, and initiates/continues chat context."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    result = orchestrator.process_multimodal_turn(
        image_bytes=image_bytes,
        user_text=query,
        session_id=session_id or "default",
        language=language or "en",
    )
    return result


@app.get("/api/chat/history/{session_id}", tags=["Multimodal Chat"])
def get_chat_history(session_id: str):
    """Retrieves conversation history and diagnosed plant context for an active session."""
    session = session_manager.get_or_create_session(session_id)
    return {
        "session_id": session_id,
        "current_crop": session.current_crop,
        "current_diagnosed_disease": session.current_diagnosed_disease,
        "messages": session.messages,
    }


@app.delete("/api/chat/session/{session_id}", tags=["Multimodal Chat"])
def reset_chat_session(session_id: str):
    """Resets memory for a specific chat session."""
    session_manager.clear_session(session_id)
    return {"status": "success", "message": f"Session '{session_id}' cleared."}


@app.post("/api/route", tags=["Routing"])
def classify_intent_only(req: RouteRequest):
    """Performs multi-domain intent analysis."""
    multi_result = multi_router.analyze_multi_domain(req.query)
    return multi_result


@app.post("/api/scan-leaf", tags=["Computer Vision"])
async def scan_leaf_image(file: UploadFile = File(...), crop_hint: str = "auto"):
    """Accepts an uploaded leaf photograph or live camera capture and performs ML visual crop disease diagnosis with uncertainty rejection."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    raw_result = leaf_vision_scanner.analyze_image_bytes(image_bytes, crop_hint=crop_hint)

    status = raw_result.get("status", "error")
    if status == "success":
        crop_info = (
            raw_result.get("crop")
            if isinstance(raw_result.get("crop"), dict)
            else {
                "name": raw_result.get("crop", "Unknown"),
                "confidence": raw_result.get("crop_confidence", 0.95),
            }
        )
        disease_info = {
            "name": raw_result.get("predicted_disease", "Unknown"),
            "confidence": raw_result.get("disease_confidence", 0.90),
        }
        return {
            "status": "success",
            "crop": crop_info,
            "disease": disease_info,
            "predicted_disease": raw_result.get("predicted_disease", "Crop Disease"),
            "confidence_pct": raw_result.get("confidence_pct", 90.0),
            "severity": "unknown",
            "affected_leaf_area_pct": raw_result.get("affected_leaf_area_pct", 0.0),
            "metrics": raw_result.get("metrics", {}),
            "quality_metrics": raw_result.get("quality_metrics", {}),
            "detected_symptoms": raw_result.get("detected_symptoms", []),
            "verified_protocol": raw_result.get("verified_protocol", ""),
            "knowledge": raw_result.get("knowledge", {}),
            "ai_description": raw_result.get("ai_description", ""),
            "engine": raw_result.get("engine", "vision_pipeline"),
        }
    elif status == "uncertain":
        return {
            "status": "uncertain",
            "crop": raw_result.get("crop", {}),
            "disease": raw_result.get("disease", {}),
            "confidence_pct": raw_result.get("confidence_pct", 0.0),
            "severity": "unknown",
            "quality_metrics": raw_result.get("quality_metrics", {}),
            "message": raw_result.get(
                "message",
                "Unable to confidently identify the crop/disease. Please capture a clearer image.",
            ),
            "ai_description": raw_result.get("ai_description", ""),
        }
    else:
        return {
            "status": status,
            "crop": {},
            "disease": {},
            "confidence_pct": 0.0,
            "severity": "unknown",
            "quality_metrics": raw_result.get("quality_metrics", {}),
            "message": raw_result.get(
                "message", "Please capture a clearer image with better lighting."
            ),
            "ai_description": raw_result.get("ai_description", ""),
        }


@app.get("/api/sensor/telemetry", tags=["IoT Sensors"])
def get_sensor_telemetry():
    """Returns real-time soil moisture and environmental weather sensor metrics."""
    return sensor_manager.get_telemetry()


@app.post("/api/sensor/telemetry", tags=["IoT Sensors"])
def update_sensor_telemetry(data: SensorTelemetryUpdate):
    """Allows remote IoT telemetry updates."""
    sensor_manager.set_manual_telemetry(
        soil_moisture=data.soil_moisture,
        ambient_temp=data.ambient_temp,
        humidity=data.humidity,
        rain=data.rain,
    )
    return {
        "status": "success",
        "message": "Telemetry state updated successfully.",
        "telemetry": sensor_manager.get_telemetry(),
    }


@app.post("/api/voice/clean", tags=["Voice Engine"])
def clean_text_for_speech(req: SpeechCleanRequest):
    """Prepares advisory text for offline Text-to-Speech synthesis."""
    clean = voice_engine.clean_text_for_speech(req.text)
    return {"cleaned_speech_text": clean, "language": req.language}


@app.get("/api/languages", tags=["Localization"])
def get_languages():
    """Returns available Indian regional languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/api/domains", tags=["Domain Configuration"])
def get_domain_modules():
    """Returns active domain modules, system prompts, and chunk counts."""
    chunks = KnowledgeChunker.get_all_chunks()
    counts = {}
    for d in ["disease", "pesticide", "weather", "irrigation"]:
        counts[d] = len([c for c in chunks if c.get("domain") == d])

    return {
        "domains": [
            {
                "id": k,
                "name": v["name"],
                "knowledge_chunks": counts.get(k, 0),
                "system_prompt": v["system_prompt"],
            }
            for k, v in DOMAIN_CONFIGS.items()
            if k != "general"
        ]
    }


@app.get("/api/knowledge", tags=["Knowledge Base"])
def browse_knowledge_base(
    domain: Optional[str] = Query(None, description="Filter by domain")
):
    """Browse stored agricultural knowledge base chunks."""
    all_chunks = KnowledgeChunker.get_all_chunks()
    if domain and domain != "all":
        filtered = [c for c in all_chunks if c.get("domain") == domain.lower()]
        return {"total": len(filtered), "chunks": filtered}
    return {"total": len(all_chunks), "chunks": all_chunks}


@app.get("/api/benchmark", tags=["Evaluation"])
def run_evaluation_benchmark():
    """Executes the test query benchmark and returns accuracy and confusion matrix."""
    return benchmark_runner.run_benchmark()


@app.post("/api/calculate/spray-dosage", tags=["Agro-Calculators"])
def calculate_spray_dosage(req: SprayDosageRequest):
    """Calculates chemical dilution, required knapsack refills, and total spray volume."""
    return agri_calculator.calculate_spray_dosage(
        area_acres=req.area_acres,
        dose_per_liter=req.dose_per_liter,
        unit=req.unit or "ml",
        tank_capacity_liters=req.tank_capacity_liters or 16.0,
        water_volume_per_acre=req.water_volume_per_acre or 200.0,
    )


@app.post("/api/calculate/water-requirement", tags=["Agro-Calculators"])
def calculate_water_requirement(req: WaterRequirementRequest):
    """Calculates crop evapotranspiration (ETc) and daily/weekly water requirements under FAO-56."""
    return agri_calculator.calculate_water_requirement(
        crop=req.crop,
        stage=req.stage or "mid",
        area_acres=req.area_acres or 1.0,
        eto_mm_day=req.eto_mm_day or 4.5,
        irrigation_efficiency=req.irrigation_efficiency or 0.85,
    )


@app.post("/api/calculate/fertilizer-npk", tags=["Agro-Calculators"])
def calculate_fertilizer_npk(req: FertilizerNPKRequest):
    """Calculates required commercial fertilizer products (Urea, DAP, MOP) for targeted NPK."""
    return agri_calculator.calculate_fertilizer_npk_sources(
        n_kg=req.n_kg,
        p_kg=req.p_kg,
        k_kg=req.k_kg,
    )


@app.post("/api/safety/validate-advisory", tags=["Safety Guardrails"])
def validate_advisory_safety(req: SafetyAuditRequest):
    """Performs safety audit against banned agrochemicals, PPE mandates, and toxic concentrations."""
    return safety_guardrails.audit_advisory(req.advisory_text)


@app.get("/api/sensor/advisory-alerts", tags=["IoT Sensors"])
def get_sensor_advisory_alerts():
    """Returns prioritized real-time agronomic alerts triggered by live sensor readings."""
    return {
        "alerts": sensor_manager.get_proactive_field_alerts(),
        "recommended_irrigation_duration_min": sensor_manager.calculate_irrigation_duration_minutes(),
    }


@app.get("/api/chat/summary/{session_id}", tags=["Multimodal Chat"])
def get_consultation_summary(session_id: str):
    """Generates an executive agronomic summary of an active farmer consultation session."""
    return session_manager.get_session_summary(session_id)


