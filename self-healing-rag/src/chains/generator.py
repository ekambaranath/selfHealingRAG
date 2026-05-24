"""
Answer generator — calls local Llama 3.1 via Ollama.
Zero cost, zero API key, fully private.
"""

from typing import List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("generator")

SYSTEM_PROMPT = """You are a precise, factual assistant.
Answer the user's question ONLY using the provided context documents.
If the context does not contain enough information to answer, say exactly:
"I don't have enough information in the provided documents to answer this question."

Rules:
- Ground every claim in the provided context
- Do NOT invent facts, statistics, or details not in the context
- Be concise and direct
- Cite which document the information comes from when relevant
"""

ANSWER_TEMPLATE = """Context Documents:
{context}

---
Question: {question}

Answer (based strictly on the context above):"""


def build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_predict=settings.max_tokens,
        temperature=settings.temperature,
        timeout=120,  # 2 min — prevents silent hang on slow/OOM machines
    )


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"Document {i}")
        parts.append(f"[Doc {i} — {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_answer(query: str, docs: List[Document]) -> str:
    """Generate a grounded answer from retrieved docs using local Ollama."""
    if not docs:
        return "I don't have enough information in the provided documents to answer this question."

    context = format_context(docs)
    prompt = ANSWER_TEMPLATE.format(context=context, question=query)

    log.info("generating_answer", query=query[:60], num_docs=len(docs), model=settings.ollama_model)

    llm = build_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    answer = response.content
    log.info("answer_generated", words=len(answer.split()))
    return answer
