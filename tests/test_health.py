from dotenv import load_dotenv

load_dotenv()

from app.rag_engine import rag_engine

import pytest
import os

def test_health_check():
    """Test health check returns valid status."""
    result = rag_engine.health_check()
    assert result["status"] in ["healthy", "degraded", "unhealthy"]

def test_load_documents():
    """Test document loading."""
    pdf_path = "./data/indian-penal-code.pdf"
    if os.path.exists(pdf_path):
        chunks = rag_engine.load_documents(pdf_path, "test")
        assert chunks > 0
        assert rag_engine.retriever is not None

def test_query_requires_load():
    """Test query fails without loaded documents."""
    rag_engine.retriever = None  # Reset
    with pytest.raises(ValueError):
        rag_engine.query("test question")

def test_query_returns_tuple():
    """Test query returns correct tuple structure."""
    pdf_path = "./data/indian-penal-code.pdf"
    if os.path.exists(pdf_path):
        rag_engine.load_documents(pdf_path, "test")
        result = rag_engine.query("section 420")
        assert isinstance(result, tuple)
        assert len(result) == 6
        answer, sources, confidence, ret_time, gen_time, cache_hit = result
        assert isinstance(answer, str)
        assert isinstance(sources, list)
        assert isinstance(cache_hit, bool)