"""
Central configuration — fully local, no API keys needed.
All tuneable knobs for the Self-Healing RAG pipeline live here.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv(override=False)


class Settings(BaseSettings):
    # ── Ollama (local LLM) ───────────────────────────────────────────────────
    # Default model: phi3:mini — recommended for 16GB RAM machines
    # Alternatives you can set via env:
    #   mistral:7b       — faster, slightly less accurate
    #   llama3.1:8b      — better quality, needs 32GB RAM
    #   llama3.1:70b     — best quality, needs 64GB RAM
    #   gemma2:9b        — Google's model, great instruction following
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="tinyllama", alias="OLLAMA_MODEL")

    # ── RAG Tuning ───────────────────────────────────────────────────────────
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    top_k_docs: int = Field(default=4, alias="TOP_K_DOCS")
    max_tokens: int = Field(default=1024, alias="MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="TEMPERATURE")

    # ── Document Chunking ────────────────────────────────────────────────────
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = "self_healing_rag"

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def data_dir(self) -> Path:
        return Path(__file__).parent.parent.parent / "data" / "sample_docs"

    @property
    def ollama_configured(self) -> bool:
        """Check if Ollama is reachable."""
        import httpx
        try:
            r = httpx.get(f"{self.ollama_base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


settings = Settings()
