from typing import Dict, Any
from core.intent_router import IntentRouter
from core.rag_pipeline import RAGPipeline

class AgentOrchestrator:
    """Dynamically assembles specialized advisory agents on the fly."""

    def __init__(self):
        self.router = IntentRouter()
        self.rag = RAGPipeline()

    def process_query(self, query: str) -> Dict[str, Any]:
        domain = self.router.route(query)
        context_chunks = self.rag.retrieve(query, domain)
        context_str = self.rag.format_context(context_chunks)

        return {
            "domain": domain,
            "query": query,
            "context": context_str,
            "status": "ready"
        }
