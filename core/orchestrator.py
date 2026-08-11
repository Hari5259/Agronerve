import time
import requests
from typing import Dict, Any, List
from config import settings
from core.intent_router import IntentRouter
from core.multi_domain_router import MultiDomainRouter
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
    """Dynamic Agent Orchestration engine supporting single-domain and composite multi-domain agents."""

    def __init__(self):
        self.router = IntentRouter()
        self.multi_router = MultiDomainRouter()
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

    def _fallback_offline_synthesizer(self, active_domains: List[str], chunks: List[Dict[str, Any]], query: str) -> str:
        """Grounded offline knowledge synthesizer for single and compound multi-domain queries."""
        if not chunks:
            domain_label = " + ".join(d.capitalize() for d in active_domains)
            return (
                f"I have analyzed your query across **{domain_label}**, but did not find matching offline knowledge entries. "
                "Please specify your crop name and symptoms or field conditions."
            )

        lines = []
        if len(active_domains) > 1:
            domains_str = " & ".join([d.capitalize() for d in active_domains])
            lines.append(f"**⚡ AgroNerve Composite Multi-Agent Advisory ({domains_str}):**\n")
        else:
            lines.append(f"Based on AgroNerve's verified agricultural database for **{active_domains[0].capitalize()} Advisory**:\n")

        for i, chunk in enumerate(chunks[:3]):
            title = chunk.get("title") or chunk.get("crop") or f"Document #{i+1}"
            lines.append(f"### 📍 {title} [{chunk.get('domain', '').capitalize()}]")
            lines.append(chunk.get("text", "").strip())
            lines.append("")

        return "\n".join(lines)

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main execution turn: Intent Analysis -> Multi-Partition Retrieval -> Dynamic Assembly -> Multi-Post-Processing."""
        start_time = time.time()

        # 1. Multi-domain Intent Analysis
        multi_meta = self.multi_router.analyze_multi_domain(query)
        active_domains = multi_meta["active_domains"]
        is_multi_domain = multi_meta["is_multi_domain"]
        primary_domain = multi_meta["primary_domain"]

        # 2. Multi-Partition Scoped Knowledge Retrieval
        all_chunks = []
        seen_ids = set()
        for dom in active_domains:
            chunks = self.rag.retrieve(query, dom, top_k=3 if is_multi_domain else settings.TOP_K_RETRIEVAL)
            for c in chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)

        context_str = self.rag.format_context(all_chunks)

        # 3. Dynamic Agent Assembly (Single or Composite)
        if is_multi_domain:
            agent_name = "Composite Specialist (" + " + ".join([DOMAIN_CONFIGS.get(d, {}).get("name", d) for d in active_domains]) + ")"
            system_prompt = (
                "You are the AgroNerve Composite Agricultural Specialist Agent.\n"
                "The farmer's query requires joint reasoning across: " + ", ".join(active_domains) + ".\n"
                "Synthesize a unified, step-by-step advisory addressing each domain's safety guidelines and protocols."
            )
        else:
            domain_cfg = DOMAIN_CONFIGS.get(primary_domain, DOMAIN_CONFIGS["general"])
            agent_name = domain_cfg["name"]
            system_prompt = domain_cfg["system_prompt"]

        # 4. Response Generation
        llm_engine_used = "ollama"
        try:
            raw_response = self._call_ollama(system_prompt, context_str, query)
        except Exception:
            llm_engine_used = "offline_knowledge_engine"
            raw_response = self._fallback_offline_synthesizer(active_domains, all_chunks, query)

        # 5. Apply Post-Processors for all active domains
        final_response = raw_response
        for dom in active_domains:
            post_fn = DOMAIN_CONFIGS.get(dom, {}).get("post_processor")
            if post_fn:
                final_response = post_fn(final_response)

        elapsed_seconds = round(time.time() - start_time, 2)

        return {
            "query": query,
            "domain": primary_domain,
            "active_domains": active_domains,
            "is_multi_domain": is_multi_domain,
            "agent_name": agent_name,
            "response": final_response,
            "chunks_retrieved": len(all_chunks),
            "route_meta": multi_meta,
            "engine": llm_engine_used,
            "latency_seconds": elapsed_seconds,
            "context_preview": context_str[:400] + "..." if len(context_str) > 400 else context_str
        }
