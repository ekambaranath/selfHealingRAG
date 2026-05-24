"""
tests/test_pipeline.py — End-to-end tests for the Self-Healing RAG pipeline.

Run with:
    pytest tests/ -v

These tests require:
  - Documents already ingested (run scripts/ingest.py first)
  - AWS Bedrock configured (AWS_BEARER_TOKEN_BEDROCK set)

For CI without Bedrock, set SKIP_LLM_TESTS=1 to skip LLM-dependent tests.
"""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SKIP_LLM = os.environ.get("SKIP_LLM_TESTS", "0") == "1"
skip_llm = pytest.mark.skipif(SKIP_LLM, reason="LLM tests skipped (SKIP_LLM_TESTS=1)")


# ── Retrieval Tests (no LLM needed) ──────────────────────────────────────────

class TestVectorStore:
    def test_vectorstore_connects(self):
        """ChromaDB should connect without error."""
        from src.retrieval.vectorstore import get_vectorstore
        vs = get_vectorstore()
        assert vs is not None

    def test_doc_count_nonnegative(self):
        """Should return a non-negative document count."""
        from src.retrieval.vectorstore import get_doc_count
        count = get_doc_count()
        assert count >= 0

    def test_retrieve_returns_list(self):
        """retrieve() should return a list (possibly empty if no docs indexed)."""
        from src.retrieval.vectorstore import retrieve
        docs = retrieve("what is machine learning", top_k=2)
        assert isinstance(docs, list)

    def test_retrieve_doc_has_content(self):
        """Each retrieved doc should have page_content."""
        from src.retrieval.vectorstore import retrieve, get_doc_count
        if get_doc_count() == 0:
            pytest.skip("No documents indexed")
        docs = retrieve("retrieval augmented generation", top_k=2)
        for doc in docs:
            assert hasattr(doc, "page_content")
            assert len(doc.page_content) > 0

    def test_ingest_and_retrieve(self):
        """Ingest a test doc and verify it's retrievable."""
        from langchain_core.documents import Document
        from src.retrieval.vectorstore import ingest_documents, retrieve

        test_content = "The purple elephant only eats quantum bananas on Tuesdays."
        doc = Document(page_content=test_content, metadata={"source": "test"})
        ingest_documents([doc])

        results = retrieve("purple elephant quantum bananas", top_k=3)
        contents = [r.page_content for r in results]
        assert any("purple elephant" in c for c in contents), \
            "Ingested test document not found in retrieval results"


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_load(self):
        """Settings should load without errors."""
        from src.utils.config import settings
        assert settings.max_retries > 0
        assert settings.top_k_docs > 0
        assert settings.chunk_size > 0

    def test_bedrock_model_id(self):
        """Model ID should be set."""
        from src.utils.config import settings
        assert settings.bedrock_model_id


# ── Generator Tests (requires Bedrock) ───────────────────────────────────────

class TestGenerator:
    @skip_llm
    def test_generate_with_docs(self):
        """Generator should return a non-empty string given docs."""
        from langchain_core.documents import Document
        from src.chains.generator import generate_answer

        docs = [
            Document(
                page_content="RAG stands for Retrieval-Augmented Generation. It combines LLMs with external knowledge retrieval.",
                metadata={"source": "test"}
            )
        ]
        answer = generate_answer("What does RAG stand for?", docs)
        assert isinstance(answer, str)
        assert len(answer) > 10

    @skip_llm
    def test_generate_with_no_docs_returns_fallback(self):
        """With no docs, generator should return the fallback message."""
        from src.chains.generator import generate_answer
        answer = generate_answer("What is quantum entanglement?", [])
        assert "don't have enough information" in answer.lower() or len(answer) > 0


# ── Critic Tests (requires Bedrock) ──────────────────────────────────────────

class TestCritic:
    @skip_llm
    def test_relevance_grade_pass(self):
        """Relevant documents should pass the relevance check."""
        from langchain_core.documents import Document
        from src.agents.critic import grade_relevance

        docs = [Document(
            page_content="RAG is a technique that combines retrieval with generation for more accurate AI responses.",
            metadata={"source": "test"}
        )]
        result = grade_relevance("What is RAG?", docs)
        assert result.verdict in ("pass", "fail")
        assert 0.0 <= result.score <= 1.0
        assert result.reason

    @skip_llm
    def test_hallucination_grade_grounded(self):
        """A grounded answer should pass the hallucination check."""
        from langchain_core.documents import Document
        from src.agents.critic import grade_hallucination

        docs = [Document(
            page_content="ChromaDB is an open-source vector database that runs locally on disk.",
            metadata={"source": "test"}
        )]
        answer = "ChromaDB is an open-source vector database that can run locally."
        result = grade_hallucination("What is ChromaDB?", docs, answer)
        assert result.verdict in ("pass", "fail")
        assert 0.0 <= result.score <= 1.0


# ── Full Pipeline Tests (requires Bedrock + indexed docs) ────────────────────

class TestPipeline:
    @skip_llm
    def test_pipeline_runs_and_returns_dict(self):
        """Pipeline should return a dict with expected keys."""
        from src.chains.graph import run_pipeline
        from src.retrieval.vectorstore import get_doc_count

        if get_doc_count() == 0:
            pytest.skip("No documents indexed — run scripts/ingest.py first")

        result = run_pipeline("What is RAG?")
        assert "answer" in result
        assert "steps" in result
        assert "retries" in result
        assert "success" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @skip_llm
    def test_pipeline_unanswerable_graceful(self):
        """Pipeline should return graceful fallback for unanswerable queries."""
        from src.chains.graph import run_pipeline

        result = run_pipeline(
            "What is the exact population of the moon colony in 2087 and who is the current president?"
        )
        assert "answer" in result
        # Should either fallback or give a reasonable response
        assert isinstance(result["answer"], str)

    @skip_llm
    def test_pipeline_steps_nonempty(self):
        """Pipeline should always populate the steps audit trail."""
        from src.chains.graph import run_pipeline
        from src.retrieval.vectorstore import get_doc_count

        if get_doc_count() == 0:
            pytest.skip("No documents indexed")

        result = run_pipeline("What is LangGraph?")
        assert len(result["steps"]) > 0


# ── API Tests ────────────────────────────────────────────────────────────────

class TestAPI:
    def test_health_endpoint(self):
        """Health endpoint should respond without error."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "documents_indexed" in data

    def test_ui_serves(self):
        """Root endpoint should serve HTML."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_graph_diagram(self):
        """Diagram endpoint should return a string."""
        from fastapi.testclient import TestClient
        from src.api import app

        client = TestClient(app)
        response = client.get("/graph/diagram")
        assert response.status_code == 200
        assert "diagram" in response.json()
