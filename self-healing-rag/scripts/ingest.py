"""
scripts/ingest.py — Run this once to populate the vector store.

Usage:
    python scripts/ingest.py

It will:
1. Load all .txt and .pdf files from data/sample_docs/
2. Chunk and embed them
3. Persist to ChromaDB

You can also drop your own .txt files into data/sample_docs/ before running.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from src.retrieval.vectorstore import ingest_documents, get_doc_count
from src.utils.logging import get_logger

log = get_logger("ingest")

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"


def load_text_file(path: Path) -> list[Document]:
    try:
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()
    except Exception as e:
        log.error("failed_to_load", path=str(path), error=str(e))
        return []


def load_pdf_file(path: Path) -> list[Document]:
    try:
        loader = PyPDFLoader(str(path))
        return loader.load()
    except Exception as e:
        log.error("failed_to_load_pdf", path=str(path), error=str(e))
        return []


def main():
    log.info("ingestion_start", docs_dir=str(SAMPLE_DOCS_DIR))

    all_docs: list[Document] = []

    # Load .txt files
    txt_files = list(SAMPLE_DOCS_DIR.glob("*.txt"))
    for f in txt_files:
        docs = load_text_file(f)
        for doc in docs:
            doc.metadata["source"] = f.name
        all_docs.extend(docs)
        log.info("loaded_txt", file=f.name, chunks=len(docs))

    # Load .pdf files
    pdf_files = list(SAMPLE_DOCS_DIR.glob("*.pdf"))
    for f in pdf_files:
        docs = load_pdf_file(f)
        for doc in docs:
            doc.metadata["source"] = f.name
        all_docs.extend(docs)
        log.info("loaded_pdf", file=f.name, pages=len(docs))

    if not all_docs:
        log.warning("no_documents_found", path=str(SAMPLE_DOCS_DIR))
        print("\n⚠️  No documents found in data/sample_docs/")
        print("   Sample docs were pre-created — re-run setup.sh or check the directory.")
        return

    log.info("ingesting_all", total_documents=len(all_docs))
    total_chunks = ingest_documents(all_docs)

    final_count = get_doc_count()
    print(f"\n✅ Ingestion complete!")
    print(f"   Files processed : {len(txt_files + pdf_files)}")
    print(f"   Chunks added    : {total_chunks}")
    print(f"   Total in DB     : {final_count}")


if __name__ == "__main__":
    main()
