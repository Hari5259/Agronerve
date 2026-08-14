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
    """Accepts an uploaded leaf photograph and performs offline visual disease diagnosis."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    result = leaf_vision_scanner.analyze_image_bytes(image_bytes, crop_hint=crop_hint)
    return result


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
