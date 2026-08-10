from fastapi import FastAPI
from pydantic import BaseModel
from core.orchestrator import AgentOrchestrator

app = FastAPI(
    title="AgroNerve API",
    description="Offline Agricultural Advisory System API",
    version="1.0.0"
)

orchestrator = AgentOrchestrator()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    domain: str
    query: str
    context: str
    status: str

@app.get("/")
def read_root():
    return {"name": "AgroNerve API", "status": "online", "mode": "offline-ready"}

@app.post("/api/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    result = orchestrator.process_query(req.query)
    return result
