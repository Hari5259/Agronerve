import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from core.knowledge_seeder import KnowledgeChunker
from core.rag_pipeline import RAGPipeline

def test_knowledge_chunker_loading():
    chunks = KnowledgeChunker.get_all_chunks()
    assert len(chunks) > 0
    domains = {c["domain"] for c in chunks}
    assert {"disease", "pesticide", "weather", "irrigation"}.issubset(domains)

def test_rag_partition_retrieval():
    rag = RAGPipeline()
    disease_chunks = rag.retrieve("paddy blast symptoms", "disease", top_k=3)
    assert len(disease_chunks) > 0
    assert any("blast" in c["text"].lower() for c in disease_chunks)

def test_rag_context_formatting():
    rag = RAGPipeline()
    sample_chunks = [{"text": "Chunk 1 sample text"}, {"text": "Chunk 2 sample text"}]
    formatted = rag.format_context(sample_chunks)
    assert "Grounding Knowledge Chunk #1" in formatted
    assert "Chunk 2 sample text" in formatted
