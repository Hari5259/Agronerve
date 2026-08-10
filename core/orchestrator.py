import time
import requests
from typing import Dict, Any, List
from config import settings
from core.intent_router import IntentRouter
from core.rag_pipeline import RAGPipeline

from domains.disease import SYSTEM_PROMPT as DISEASE_PROMPT, post_process_disease_response
from domains.pesticide import SYSTEM_PROMPT as PESTICIDE_PROMPT, post_process_pesticide_response
from domains.weather import SYSTEM_PROMPT as WEATHER_PROMPT, post_process_weather_response
from domains.irrigation import SYSTEM_PROMPT as IRRIGATION_PROMPT, post_process_irrigation_response

DOMAIN_CONFIGS = {
    "disease": {
        "name": "Crop Disease Specialist",
        "system_prompt": DISEASE_PROMPT,
        "post_processor": post_process_disease_response
    },
    "pesticide": {
        "name": "Pesticide & Dosage Specialist",
        "system_prompt": PESTICIDE_PROMPT,
        "post_processor": post_process_pesticide_response
    },
    "weather": {
        "name": "Weather & Climate Specialist",
        "system_prompt": WEATHER_PROMPT,
        "post_processor": post_process_weather_response
    },
    "irrigation": {
        "name": "Irrigation Planning Specialist",
        "system_prompt": IRRIGATION_PROMPT,
        "post_processor": post_process_irrigation_response
    },
    "general": {
        "name": "Agricultural Generalist",
        "system_prompt": "You are AgroNerve, an expert offline agricultural assistant. Provide clear, accurate agronomic guidance.",
        "post_processor": lambda text: text
    }
}

class AgentOrchestrator:
    """Dynamic Agent Orchestration engine that assembles specialist agents per query."""

    def __init__(self):
        self.router = IntentRouter()
        self.rag = RAGPipeline()

    def _call_ollama(self, system_prompt: str, context: str, query: str) -> str:
        """Invokes local Ollama inference."""
        full_prompt = (
            f"### SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"### VERIFIED GROUNDING CONTEXT:\n{context}\n\n"
            f"### FARMER QUERY:\n{query}\n\n"
            "### ADVISORY RESPONSE:\n"
        )
        payload = {
            "model": settings.DEFAULT_LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 512
            }
        }
        res = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=30.0
        )
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        raise RuntimeError(f"Ollama returned status code {res.status_code}")

    def _fallback_offline_synthesizer(self, domain: str, chunks: List[Dict[str, Any]], query: str) -> str:
        """Grounded offline knowledge synthesizer when Ollama LLM server is offline."""
        if not chunks:
            return (
                f"I have analyzed your query regarding **{domain}**, but did not find matching offline knowledge entries. "
                "Please verify the crop name or describe additional symptoms for a more precise diagnosis."
            )

        top_chunk = chunks[0]
        text = top_chunk.get("text", "")

        lines = [f"Based on AgroNerve's verified agricultural database for **{domain.capitalize()} Advisory**:\n"]
        lines.append(text)

        if len(chunks) > 1:
            lines.append("\n**Additional Relevant Knowledge:**")
            for chunk in chunks[1:]:
                title = chunk.get("title") or chunk.get("crop") or "Reference"
                lines.append(f"- **{title}**: {chunk.get('text', '').splitlines()[0]}")

        return "\n".join(lines)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main execution turn: Intent Classification -> Retrieval -> Dynamic Assembly -> Post-Processing."""
        start_time = time.time()

        # 1. Two-stage Intent Classification
        route_meta = self.router.route(query)
        domain = route_meta["domain"]

        # 2. Domain Scoped Knowledge Retrieval
        retrieved_chunks = self.rag.retrieve(query, domain, top_k=settings.TOP_K_RETRIEVAL)
        context_str = self.rag.format_context(retrieved_chunks)

        # 3. Dynamic Agent Configuration Selection
        domain_cfg = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["general"])
        system_prompt = domain_cfg["system_prompt"]
        post_processor = domain_cfg["post_processor"]

        # 4. Response Generation (Ollama LLM or Grounded Synthesizer)
        llm_engine_used = "ollama"
        try:
            raw_response = self._call_ollama(system_prompt, context_str, query)
        except Exception:
            llm_engine_used = "offline_knowledge_engine"
            raw_response = self._fallback_offline_synthesizer(domain, retrieved_chunks, query)

        # 5. Domain-Specific Post-Processing
        final_response = post_processor(raw_response)
        elapsed_seconds = round(time.time() - start_time, 2)

        return {
            "query": query,
            "domain": domain,
            "agent_name": domain_cfg["name"],
            "response": final_response,
            "chunks_retrieved": len(retrieved_chunks),
            "route_meta": route_meta,
            "engine": llm_engine_used,
            "latency_seconds": elapsed_seconds,
            "context_preview": context_str[:300] + "..." if len(context_str) > 300 else context_str
        }
