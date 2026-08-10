from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.orchestrator import AgentOrchestrator, DOMAIN_CONFIGS
from core.knowledge_seeder import KnowledgeChunker
from core.intent_router import IntentRouter
from evaluation.benchmark import AgroNerveBenchmark
from config import settings

app = FastAPI(
    title="AgroNerve API",
    description="Intelligent Offline Agricultural Advisory System API using Dynamic Agent Orchestration & RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = AgentOrchestrator()
router = IntentRouter()
benchmark_runner = AgroNerveBenchmark()

# Pydantic Request & Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the recommended dosage of Chlorantraniliprole 18.5 SC for stem borer in paddy?")

class QueryResponse(BaseModel):
    query: str
    domain: str
    agent_name: str
    response: str
    chunks_retrieved: int
    engine: str
    latency_seconds: float
    route_meta: Dict[str, Any]
    context_preview: str

class RouteRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    domain: str
    score: float
    is_confident: bool
    stage_used: int

@app.get("/", tags=["Health"])
def health_check():
    """Returns AgroNerve service health and offline readiness."""
    total_chunks = len(KnowledgeChunker.get_all_chunks())
    return {
        "name": settings.APP_NAME,
        "status": "healthy",
        "mode": "offline-edge-ready",
        "version": "1.0.0",
        "knowledge_base_chunks": total_chunks,
        "supported_domains": list(DOMAIN_CONFIGS.keys())
    }

@app.post("/api/query", response_model=QueryResponse, tags=["Advisory"])
def process_agricultural_query(req: QueryRequest):
    """Processes a natural-language farmer query through dynamic agent assembly and RAG grounding."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = orchestrator.process_query(req.query)
    return result

@app.post("/api/route", response_model=RouteResponse, tags=["Routing"])
def classify_intent_only(req: RouteRequest):
    """Performs 2-stage intent classification without running full generation."""
    route_result = router.route(req.query)
    return route_result

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
                "system_prompt": v["system_prompt"]
            }
            for k, v in DOMAIN_CONFIGS.items() if k != "general"
        ]
    }

@app.get("/api/knowledge", tags=["Knowledge Base"])
def browse_knowledge_base(domain: Optional[str] = Query(None, description="Filter by domain")):
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
