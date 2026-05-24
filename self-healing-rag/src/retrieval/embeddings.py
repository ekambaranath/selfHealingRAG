"""
Local HuggingFace embeddings — completely FREE, no API key needed.
Model: all-MiniLM-L6-v2 (~90MB, downloaded once, then cached).
"""

from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("embeddings")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached HuggingFace embeddings instance."""
    log.info("loading_embeddings_model", model=settings.embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("embeddings_ready", model=settings.embedding_model)
    return embeddings
