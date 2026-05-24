"""
Self-Healing RAG Pipeline — LangGraph Stateful Cyclical Workflow

Flow:
  rewrite_query → retrieve → grade_relevance
                                  │ PASS
                             generate_answer
                                  │
                        grade_hallucination
                                  │ PASS
                             finalize ──► END

On FAIL at either critic step:
  - Increment retry counter
  - If retries < MAX_RETRIES: rewrite_query (loop)
  - If retries >= MAX_RETRIES: fallback ──► END
"""

from __future__ import annotations

from typing import List, TypedDict, Annotated
import operator

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src.retrieval.vectorstore import retrieve
from src.chains.generator import generate_answer
from src.agents.critic import grade_relevance, grade_hallucination
from src.agents.query_rewriter import rewrite_query
from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("graph")

GRACEFUL_FALLBACK = (
    "I don't have enough information in the provided documents to answer this question accurately. "
    "Please try rephrasing your question or ensure the relevant documents have been indexed."
)


# ── State Schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    original_query: str
    current_query: str
    retrieved_docs: List[Document]
    generated_answer: str
    retry_count: int
    failure_reason: str
    steps: Annotated[List[str], operator.add]
    _relevance_verdict: str
    _hallucination_verdict: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_rewrite_query(state: RAGState) -> dict:
    retry = state["retry_count"]
    if retry == 0:
        log.info("graph_node", node="rewrite_query", action="first_pass")
        return {
            "current_query": state["original_query"],
            "steps": [f"🔍 Pass 1: Using original query: '{state['original_query']}'"],
        }
    else:
        log.info("graph_node", node="rewrite_query", action="retry", retry=retry)
        new_query = rewrite_query(
            query=state["original_query"],
            retry_num=retry,
            failure_reason=state.get("failure_reason", ""),
        )
        return {
            "current_query": new_query,
            "steps": [f"✏️ Retry {retry}: Rewriting query → '{new_query}'"],
        }


def node_retrieve(state: RAGState) -> dict:
    log.info("graph_node", node="retrieve", query=state["current_query"][:60])
    docs = retrieve(state["current_query"])
    return {
        "retrieved_docs": docs,
        "steps": [f"📚 Retrieved {len(docs)} document chunks"],
    }


def node_grade_relevance(state: RAGState) -> dict:
    log.info("graph_node", node="grade_relevance")
    docs = state["retrieved_docs"]

    if not docs:
        return {
            "failure_reason": "No documents retrieved",
            "_relevance_verdict": "fail",
            "steps": ["❌ Relevance: No documents retrieved"],
        }

    result = grade_relevance(state["current_query"], docs)
    emoji = "✅" if result.verdict == "pass" else "❌"
    step = f"{emoji} Relevance grade: {result.verdict.upper()} (score={result.score:.2f}) — {result.reason}"

    return {
        "failure_reason": result.reason if result.verdict == "fail" else "",
        "_relevance_verdict": result.verdict,
        "steps": [step],
    }


def node_generate(state: RAGState) -> dict:
    log.info("graph_node", node="generate")
    answer = generate_answer(state["current_query"], state["retrieved_docs"])
    return {
        "generated_answer": answer,
        "steps": ["🤖 Generated answer using Claude via Bedrock"],
    }


def node_grade_hallucination(state: RAGState) -> dict:
    log.info("graph_node", node="grade_hallucination")
    result = grade_hallucination(
        state["current_query"],
        state["retrieved_docs"],
        state["generated_answer"],
    )
    emoji = "✅" if result.verdict == "pass" else "❌"
    step = f"{emoji} Hallucination grade: {result.verdict.upper()} (score={result.score:.2f}) — {result.reason}"

    return {
        "failure_reason": result.reason if result.verdict == "fail" else "",
        "_hallucination_verdict": result.verdict,
        "steps": [step],
    }


def node_finalize(state: RAGState) -> dict:
    log.info("graph_node", node="finalize", success=True)
    return {"steps": ["✅ Pipeline complete — answer approved by all critics"]}


def node_graceful_fallback(state: RAGState) -> dict:
    log.info("graph_node", node="graceful_fallback", retries=state["retry_count"])
    return {
        "generated_answer": GRACEFUL_FALLBACK,
        "steps": [f"⚠️ Max retries ({settings.max_retries}) reached — returning graceful fallback"],
    }


# ── Edge Conditions ───────────────────────────────────────────────────────────

def route_after_relevance(state: RAGState) -> str:
    if state.get("_relevance_verdict") == "pass":
        return "generate"
    return "increment_retry"


def route_after_hallucination(state: RAGState) -> str:
    if state.get("_hallucination_verdict") == "pass":
        return "finalize"
    return "increment_retry" 

def node_increment_retry(state: RAGState) -> dict:
    new_count = state["retry_count"] + 1
    log.info("graph_node", node="increment_retry", retry=new_count)
    return {"retry_count": new_count}


def route_after_increment(state: RAGState) -> str:
    if state["retry_count"] >= settings.max_retries:
        return "fallback"
    return "rewrite_query"


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade_relevance", node_grade_relevance)
    graph.add_node("generate", node_generate)
    graph.add_node("grade_hallucination", node_grade_hallucination)
    graph.add_node("finalize", node_finalize)
    graph.add_node("fallback", node_graceful_fallback)
    graph.add_node("increment_retry", node_increment_retry)

    graph.set_entry_point("rewrite_query")

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_edge("generate", "grade_hallucination")

    graph.add_conditional_edges(
    "grade_relevance",
    route_after_relevance,
    {"generate": "generate", "increment_retry": "increment_retry"},
)

    graph.add_conditional_edges(
    "grade_hallucination",
    route_after_hallucination,
    {"finalize": "finalize", "increment_retry": "increment_retry"},
)

    graph.add_conditional_edges(
    "increment_retry",
    route_after_increment,
    {"rewrite_query": "rewrite_query", "fallback": "fallback"},
)

    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(query: str) -> dict:
    """
    Run the full Self-Healing RAG pipeline.

    Returns:
      answer      : str
      steps       : List[str]  — full audit trail
      retries     : int
      success     : bool
      query_used  : str        — final (possibly rewritten) query
    """
    log.info("pipeline_start", query=query[:80])

    initial_state: RAGState = {
        "original_query": query,
        "current_query": query,
        "retrieved_docs": [],
        "generated_answer": "",
        "retry_count": 0,
        "failure_reason": "",
        "steps": [],
        "_relevance_verdict": "",
        "_hallucination_verdict": "",
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state, config={"recursion_limit": 50})

    answer = final_state["generated_answer"]
    success = answer != GRACEFUL_FALLBACK

    log.info("pipeline_complete", success=success, retries=final_state["retry_count"])

    return {
        "answer": answer,
        "steps": final_state.get("steps", []),
        "retries": final_state["retry_count"],
        "success": success,
        "query_used": final_state["current_query"],
    }
