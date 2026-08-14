import time
import logging
import requests
from typing import Dict, Any, List, Optional
from config import settings
from core.intent_router import IntentRouter
from core.multi_domain_router import MultiDomainRouter
from core.rag_pipeline import RAGPipeline
from core.session_manager import session_manager
from core.vision_analyzer import leaf_vision_scanner
from core.translator import language_manager

from domains.disease import (
    SYSTEM_PROMPT as DISEASE_PROMPT,
    post_process_disease_response,
)
from domains.pesticide import (
    SYSTEM_PROMPT as PESTICIDE_PROMPT,
    post_process_pesticide_response,
)
from domains.weather import (
    SYSTEM_PROMPT as WEATHER_PROMPT,
    post_process_weather_response,
)
from domains.irrigation import (
    SYSTEM_PROMPT as IRRIGATION_PROMPT,
    post_process_irrigation_response,
)

logger = logging.getLogger(__name__)

DOMAIN_CONFIGS = {
    "disease": {
        "name": "Crop Disease Specialist",
        "system_prompt": DISEASE_PROMPT,
        "post_processor": post_process_disease_response,
    },
    "pesticide": {
        "name": "Pesticide & Dosage Specialist",
        "system_prompt": PESTICIDE_PROMPT,
        "post_processor": post_process_pesticide_response,
    },
    "weather": {
        "name": "Weather & Climate Specialist",
        "system_prompt": WEATHER_PROMPT,
        "post_processor": post_process_weather_response,
    },
    "irrigation": {
        "name": "Irrigation Planning Specialist",
        "system_prompt": IRRIGATION_PROMPT,
        "post_processor": post_process_irrigation_response,
    },
    "general": {
        "name": "Agricultural Generalist",
        "system_prompt": "You are AgroNerve, an expert offline agricultural assistant. Provide clear, accurate agronomic guidance.",
        "post_processor": lambda text: text,
    },
}


class AgentOrchestrator:
    """Dynamic Agent Orchestration engine supporting multimodal image inputs, session memory, and multi-domain reasoning."""

    def __init__(self):
        self.router = IntentRouter()
        self.multi_router = MultiDomainRouter()
        self.rag = RAGPipeline()
        self.vision = leaf_vision_scanner

    def _call_ollama(
        self, system_prompt: str, context: str, history: str, query: str
    ) -> str:
        """Invokes local Ollama inference with conversation history."""
        prompt_parts = [f"### SYSTEM INSTRUCTIONS:\n{system_prompt}"]
        if history:
            prompt_parts.append(f"{history}")
        if context:
            prompt_parts.append(f"### VERIFIED GROUNDING CONTEXT:\n{context}")
        prompt_parts.append(f"### FARMER QUERY:\n{query}\n\n### ADVISORY RESPONSE:\n")

        full_prompt = "\n\n".join(prompt_parts)
        payload = {
            "model": settings.DEFAULT_LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 512},
        }
        res = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30.0
        )
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        raise RuntimeError(f"Ollama returned status code {res.status_code}")

    def _fallback_offline_synthesizer(
        self,
        active_domains: List[str],
        chunks: List[Dict[str, Any]],
        query: str,
        visual_context: Optional[Dict[str, Any]] = None,
        language: str = "en",
    ) -> str:
        """Grounded offline knowledge synthesizer for text and visual inquiries with i18n support."""
        logger.info(
            f"Using offline knowledge synthesis for query: '{query}' (language: {language})"
        )
        lines = []

        if visual_context:
            visual_label = language_manager.get_text("visual_scan_result", language)
            confidence_label = language_manager.get_text(
                "diagnostic_confidence", language
            )
            damage_label = language_manager.get_text(
                "estimated_foliar_damage", language
            )
            lines.append(
                f"📸 **{visual_label}:** {visual_context.get('ai_description', '')}"
            )
            lines.append(
                f"**{confidence_label}:** {visual_context.get('confidence_pct', 85)}% | **{damage_label}:** {visual_context.get('affected_leaf_area_pct', 20)}%\n"
            )

        if len(active_domains) > 1:
            composite_label = language_manager.get_text("composite_advisory", language)
            domains_str = " & ".join([d.capitalize() for d in active_domains])
            lines.append(f"**⚡ {composite_label} ({domains_str}):**\n")
        elif not visual_context:
            domain_advisory_base = language_manager.get_text(
                "domain_advisory_base", language
            )
            formatted_base = domain_advisory_base.format(
                domain=active_domains[0].capitalize()
            )
            lines.append(f"**{formatted_base}:**\n")

        if chunks:
            for i, chunk in enumerate(chunks[:2]):
                title = chunk.get("title") or chunk.get("crop") or f"Document #{i+1}"
                lines.append(f"### 📍 {title}")
                lines.append(chunk.get("text", "").strip())
                lines.append("")
        elif not visual_context:
            lines.append(language_manager.get_text("no_knowledge_found", language))

        return "\n".join(lines)

    def process_multimodal_turn(
        self,
        image_bytes: bytes,
        user_text: Optional[str] = None,
        session_id: str = "default",
        crop_hint: str = "auto",
        language: str = "en",
    ) -> Dict[str, Any]:
        """Handles an uploaded leaf image, updates session context, and generates conversational AI diagnosis."""
        logger.info(
            f"Processing multimodal turn for session '{session_id}' in language '{language}'"
        )
        start_time = time.time()
        session = session_manager.get_or_create_session(session_id)

        # 1. Run visual diagnostic scan with crop prioritization
        effective_crop_hint = (
            crop_hint if crop_hint != "auto" else (session.current_crop or "auto")
        )
        vision_result = self.vision.analyze_image_bytes(
            image_bytes, crop_hint=effective_crop_hint, user_query=user_text
        )

        # 2. Update persistent session context
        session.update_visual_context(vision_result)
        diagnosed_disease = vision_result.get("predicted_disease", "Crop Disease")
        detected_crop = vision_result.get("crop", "Crop")

        # 3. Retrieve grounding treatment knowledge
        search_query = f"{detected_crop} {diagnosed_disease} symptoms management"
        retrieved_chunks = self.rag.retrieve(search_query, "disease", top_k=3)
        context_str = self.rag.format_context(retrieved_chunks)

        # 4. Generate AI response
        effective_query = (
            user_text
            or f"Please analyze this leaf photograph of my {detected_crop} crop and suggest treatment."
        )
        history_str = session.get_conversation_history_prompt()

        lang_instruction = language_manager.get_language_prompt_instruction(language)
        if retrieved_chunks:
            effective_system_prompt = DISEASE_PROMPT + lang_instruction
        else:
            # Fallback when database is empty / not working - tell Ollama to fetch correct info from its training
            fallback_prompt = (
                "You are the AgroNerve Crop Disease Specialist Agent, acting as an experienced plant pathologist and agronomist.\n"
                "Your goal is to accurately diagnose crop diseases, identify causal organisms, and provide verified management protocols.\n"
                "Since no local database chunks are available, please use your own internal agricultural knowledge to provide the correct "
                "diagnosis and treatment recommendations for this crop."
            )
            effective_system_prompt = fallback_prompt + lang_instruction

        llm_engine_used = "ollama"
        try:
            logger.info("Sending multimodal context query to local LLM.")
            raw_response = self._call_ollama(
                system_prompt=effective_system_prompt,
                context=context_str,
                history=history_str,
                query=f"[Farmer sent leaf image]: {effective_query}\nVisual findings: {vision_result.get('ai_description')}",
            )
        except Exception as e:
            logger.warning(
                f"Ollama generation failed: {str(e)}. Falling back to offline vision synthesis."
            )
            llm_engine_used = "offline_vision_engine"
            raw_response = self._fallback_offline_synthesizer(
                active_domains=["disease"],
                chunks=retrieved_chunks,
                query=effective_query,
                visual_context=vision_result,
                language=language,
            )

        final_response = post_process_disease_response(raw_response)

        # 5. Record turn in session memory
        session.add_message(
            "user",
            effective_query,
            {"has_image": True, "vision_metrics": vision_result.get("metrics")},
        )
        session.add_message(
            "assistant",
            final_response,
            {"domain": "disease", "vision_result": vision_result},
        )

        elapsed_seconds = round(time.time() - start_time, 2)
        logger.info(
            f"Multimodal turn completed in {elapsed_seconds}s using {llm_engine_used}."
        )

        return {
            "session_id": session_id,
            "query": effective_query,
            "domain": "disease",
            "active_domains": ["disease"],
            "is_multi_domain": False,
            "agent_name": "Crop Disease Specialist (Visual AI)",
            "response": final_response,
            "vision_result": vision_result,
            "chunks_retrieved": len(retrieved_chunks),
            "engine": llm_engine_used,
            "latency_seconds": elapsed_seconds,
            "context_preview": (
                context_str[:400] + "..." if len(context_str) > 400 else context_str
            ),
        }

    def process_query(
        self, query: str, session_id: str = "default", language: str = "en"
    ) -> Dict[str, Any]:
        """Main execution turn: Contextual Multi-turn Query Processing with Session Memory."""
        logger.info(
            f"Processing query '{query}' for session '{session_id}' in language '{language}'"
        )
        start_time = time.time()
        session = session_manager.get_or_create_session(session_id)

        # 1. Multi-domain Intent Analysis (analyzed on raw incoming query)
        multi_meta = self.multi_router.analyze_multi_domain(query)
        active_domains = multi_meta["active_domains"]
        is_multi_domain = multi_meta["is_multi_domain"]
        primary_domain = multi_meta["primary_domain"]

        # 2. Context-augmented retrieval search query (combines raw query with active crop context)
        retrieval_query = query
        if session.current_crop and session.current_crop.lower() not in query.lower():
            retrieval_query = f"{session.current_crop} {query}"

        # Multi-Partition Scoped Knowledge Retrieval
        all_chunks = []
        seen_ids = set()
        for dom in active_domains:
            chunks = self.rag.retrieve(
                retrieval_query,
                dom,
                top_k=3 if is_multi_domain else settings.TOP_K_RETRIEVAL,
            )
            for c in chunks:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_chunks.append(c)

        context_str = self.rag.format_context(all_chunks)
        history_str = session.get_conversation_history_prompt()

        # 3. Dynamic Agent Assembly
        if is_multi_domain:
            agent_name = (
                "Composite Specialist ("
                + " + ".join(
                    [DOMAIN_CONFIGS.get(d, {}).get("name", d) for d in active_domains]
                )
                + ")"
            )
            system_prompt = (
                "You are the AgroNerve Composite Agricultural Specialist Agent.\n"
                "The farmer's query requires joint reasoning across: "
                + ", ".join(active_domains)
                + ".\n"
                "Synthesize a unified, step-by-step advisory addressing each domain's safety guidelines and protocols."
            )
        else:
            domain_cfg = DOMAIN_CONFIGS.get(primary_domain, DOMAIN_CONFIGS["general"])
            agent_name = domain_cfg["name"]
            system_prompt = domain_cfg["system_prompt"]

        lang_instruction = language_manager.get_language_prompt_instruction(language)
        if all_chunks:
            effective_system_prompt = system_prompt + lang_instruction
        else:
            # Fallback when database is empty / not working
            fallback_prompt = (
                f"You are the AgroNerve {agent_name}.\n"
                f"Your goal is to assist the farmer with their query regarding {primary_domain}.\n"
                "Since the local database is currently unavailable, use your own internal agricultural knowledge to provide the most "
                "accurate and helpful recommendations."
            )
            effective_system_prompt = fallback_prompt + lang_instruction

        # 4. Response Generation
        llm_engine_used = "ollama"
        try:
            logger.info("Sending query to local LLM.")
            raw_response = self._call_ollama(
                effective_system_prompt, context_str, history_str, query
            )
        except Exception as e:
            logger.warning(
                f"Ollama generation failed: {str(e)}. Falling back to offline synthesis."
            )
            llm_engine_used = "offline_knowledge_engine"
            raw_response = self._fallback_offline_synthesizer(
                active_domains, all_chunks, query, language=language
            )

        # 5. Apply Post-Processors
        final_response = raw_response
        for dom in active_domains:
            post_fn = DOMAIN_CONFIGS.get(dom, {}).get("post_processor")
            if post_fn:
                final_response = post_fn(final_response)

        # 6. Record turn in session memory
        session.add_message("user", query)
        session.add_message(
            "assistant",
            final_response,
            {"domain": primary_domain, "active_domains": active_domains},
        )

        elapsed_seconds = round(time.time() - start_time, 2)
        logger.info(
            f"Query turn completed in {elapsed_seconds}s using {llm_engine_used}."
        )

        return {
            "session_id": session_id,
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
            "context_preview": (
                context_str[:400] + "..." if len(context_str) > 400 else context_str
            ),
        }
