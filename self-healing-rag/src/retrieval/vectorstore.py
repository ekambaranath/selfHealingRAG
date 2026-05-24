"""
ChromaDB vector store — local, persistent, no API key needed.
Handles ingestion of documents and semantic search retrieval.
"""

from typing import List
from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.retrieval.embeddings import get_embeddings
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("vectorstore")


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Returns a persistent ChromaDB-backed vector store."""
    log.info("connecting_vectorstore", path=settings.chroma_persist_dir)
    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )
    count = vectorstore._collection.count()
    log.info("vectorstore_ready", documents_indexed=count)
    return vectorstore


def ingest_documents(docs: List[Document]) -> int:
    """Chunk and index documents into ChromaDB. Returns chunk count."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info("chunked_documents", total_docs=len(docs), total_chunks=len(chunks))

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    log.info("ingestion_complete", chunks_added=len(chunks))
    return len(chunks)


def retrieve(query: str, top_k: int | None = None) -> List[Document]:
    """Semantic similarity search — returns top-k most relevant chunks."""
    k = top_k or settings.top_k_docs
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    log.info("retrieved_docs", query=query[:60], count=len(docs))
    return docs


def get_doc_count() -> int:
    """Returns total number of indexed chunks."""
    return get_vectorstore()._collection.count()
