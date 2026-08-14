import math
import re
import logging
from typing import List, Dict, Any, Optional
from config import settings
from core.knowledge_seeder import KnowledgeChunker

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using ChromaDB domain partitions with local fallback."""

    def __init__(self):
        self.persist_dir = settings.CHROMA_PERSIST_DIR
        self.top_k = settings.TOP_K_RETRIEVAL
        self.chunks = KnowledgeChunker.get_all_chunks()
        self._init_chroma()

    def _init_chroma(self):
        """Attempts to initialize local ChromaDB collections with domain partitions."""
        self.chroma_client = None
        self.collections: Dict[str, Any] = {}
        try:
            import chromadb

            logger.info(
                f"Initializing ChromaDB client at persist_dir: '{self.persist_dir}'"
            )
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            for domain in ["disease", "pesticide", "weather", "irrigation"]:
                col = self.chroma_client.get_or_create_collection(
                    name=f"agronerve_{domain}_kb"
                )
                self.collections[domain] = col

                # Check if collection is empty, then seed
                if col.count() == 0:
                    domain_chunks = [c for c in self.chunks if c["domain"] == domain]
                    if domain_chunks:
                        logger.info(
                            f"Seeding {len(domain_chunks)} chunks for domain: '{domain}'"
                        )
                        col.add(
                            ids=[c["id"] for c in domain_chunks],
                            documents=[c["text"] for c in domain_chunks],
                            metadatas=[c.get("metadata", {}) for c in domain_chunks],
                        )
            logger.info("ChromaDB partitions initialized and seeded successfully.")
        except Exception as e:
            # Fallback gracefully to offline in-memory partitioned retrieval
            logger.warning(
                f"ChromaDB initialization failed: {str(e)}. Falling back to in-memory matching."
            )
            self.chroma_client = None

    def retrieve(
        self, query: str, domain: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant domain knowledge chunks for a query."""
        k = top_k or self.top_k
        logger.info(
            f"RAG retrieval query: '{query}' for domain: '{domain}' (top_k={k})"
        )

        # 1. Try ChromaDB retrieval if available
        if self.chroma_client and domain in self.collections:
            try:
                results = self.collections[domain].query(
                    query_texts=[query], n_results=k
                )
                if (
                    results
                    and "documents" in results
                    and results["documents"]
                    and results["documents"][0]
                ):
                    docs = results["documents"][0]
                    ids = (
                        results["ids"][0]
                        if "ids" in results
                        else [f"doc_{i}" for i in range(len(docs))]
                    )
                    logger.info(
                        f"ChromaDB retrieval successful. Found {len(docs)} chunks."
                    )
                    return [
                        {"id": id_, "text": doc, "domain": domain}
                        for id_, doc in zip(ids, docs)
                    ]
            except Exception as e:
                logger.warning(
                    f"ChromaDB retrieval failed for query '{query}': {str(e)}. Retrying offline fallback."
                )
                pass

        # 2. Resilient Offline Scoped Keyword & Semantic Similarity Fallback
        return self._offline_retrieve(query, domain, k)

    def _offline_retrieve(
        self, query: str, domain: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """BM25/TF-IDF inspired term matching across domain-partitioned chunks."""
        target_chunks = [
            c for c in self.chunks if domain == "general" or c["domain"] == domain
        ]
        if not target_chunks:
            target_chunks = self.chunks

        query_tokens = set(re.findall(r"\w+", query.lower()))
        if not query_tokens:
            return target_chunks[:top_k]

        scored_chunks = []
        for chunk in target_chunks:
            chunk_text_lower = chunk["text"].lower()
            chunk_tokens = re.findall(r"\w+", chunk_text_lower)
            doc_len = len(chunk_tokens) or 1

            score = 0.0
            # Term frequency & title matches
            for token in query_tokens:
                if len(token) < 2:
                    continue
                tf = chunk_text_lower.count(token)
                if tf > 0:
                    score += (tf / doc_len) * (
                        2.5 if token in chunk.get("title", "").lower() else 1.0
                    )
                    score += 1.0  # Base occurrence boost

            # Specific crop match boost
            if "crop" in chunk and chunk["crop"]:
                crop_lower = chunk["crop"].lower()
                for q_tok in query_tokens:
                    if q_tok in crop_lower:
                        score += 5.0

            scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored_chunks[:top_k]]
        logger.info(f"Offline retrieval completed. Found {len(results)} chunks.")
        return results

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a prompt-ready context string."""
        if not chunks:
            return "No specific offline knowledge chunks retrieved."
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(
                f"[Grounding Knowledge Chunk #{i+1}]\n{chunk.get('text', '').strip()}"
            )
        return "\n\n".join(formatted)
