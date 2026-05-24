"""
Query Rewriter Agent — activated when the critic rejects an answer.
Fully local via Ollama. Three escalating reformulation strategies.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("query_rewriter")

REWRITE_SYSTEM = """You are a search query optimization expert.
The initial search query failed to find relevant documents.
Rewrite the query to find better results.
Respond with ONLY the rewritten query — no explanation, no quotes, no punctuation prefix."""

REWRITE_TEMPLATES = {
    1: """Search query failed. Rewrite using synonyms or broader phrasing.

Original query: {query}
Failure reason: {reason}

Rewritten query:""",

    2: """Query has failed twice. Simplify to core keywords or a simpler sub-question.

Original query: {query}
Failure reason: {reason}

Simplified query:""",

    3: """Final retry. Extract only the 2-3 most essential keywords.

Original query: {query}

Keywords:""",
}


def rewrite_query(query: str, retry_num: int, failure_reason: str = "") -> str:
    """Reformulate the query based on retry number (1, 2, or 3)."""
    template_key = min(retry_num, 3)
    prompt = REWRITE_TEMPLATES[template_key].format(
        query=query,
        reason=failure_reason or "Insufficient relevant information found",
    )

    log.info("rewriting_query", retry=retry_num, original=query[:60])

    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_predict=64,      # Query rewrites should be short
        temperature=0.4,     # Slight creativity to find different angles
    )

    response = llm.invoke([
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=prompt),
    ])

    rewritten = response.content.strip().strip('"').strip("'").split("\n")[0].strip()
    # Fallback if model returns empty
    if not rewritten:
        rewritten = " ".join(query.split()[:4])

    log.info("query_rewritten", original=query[:60], rewritten=rewritten[:60])
    return rewritten
