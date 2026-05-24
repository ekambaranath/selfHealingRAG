"""
Critic Agent — the heart of "self-healing".
Now fully local via Ollama — no API key, no cost, no data leaves your machine.

Two grading functions:
  grade_relevance:     Are retrieved docs relevant to the query?
  grade_hallucination: Is the answer grounded in the docs?
"""

import json
from typing import List, Literal
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("critic")


@dataclass
class GradeResult:
    verdict: Literal["pass", "fail"]
    reason: str
    score: float  # 0.0 to 1.0


# ── Prompts ───────────────────────────────────────────────────────────────────
# Note: local models need slightly more explicit JSON instructions than cloud models

RELEVANCE_SYSTEM = """You are a strict relevance evaluator for a RAG system.
Evaluate whether the retrieved documents are relevant to the user's question.

You MUST respond with ONLY a valid JSON object. No explanation. No markdown. No extra text.
Example output: {"verdict": "pass", "score": 0.85, "reason": "Documents directly address the question"}

verdict must be exactly "pass" or "fail".
score must be a float between 0.0 and 1.0.
Use "fail" if score is below 0.5."""

RELEVANCE_TEMPLATE = """User Question: {question}

Retrieved Documents:
{context}

Respond with JSON only. Are these documents relevant?"""


HALLUCINATION_SYSTEM = """You are a strict hallucination detector for a RAG system.
Check if the generated answer is fully grounded in the provided context documents.
Hallucination = any fact in the answer that is NOT found in the context.

You MUST respond with ONLY a valid JSON object. No explanation. No markdown. No extra text.
Example output: {"verdict": "pass", "score": 0.9, "reason": "All claims found in context"}

verdict must be exactly "pass" or "fail".
score must be a float between 0.0 and 1.0.
Use "fail" if score is below 0.6."""

HALLUCINATION_TEMPLATE = """Context Documents:
{context}

Generated Answer:
{answer}

Respond with JSON only. Is this answer grounded in the context?"""


# ── Core critic call ──────────────────────────────────────────────────────────

def _call_critic(system: str, user_prompt: str) -> GradeResult:
    """
    Calls local Ollama with a grading prompt.
    Uses temperature=0 for deterministic, consistent verdicts.
    Has a robust JSON parser that handles common local model quirks.
    """
    # Use a lower temperature and slightly smaller model for the critic
    # to keep grading fast and deterministic
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        num_predict=256,     # Critic only needs a short JSON response
        temperature=0.0,     # Fully deterministic grading
        format="json",       # Ollama's JSON mode — forces valid JSON output
        timeout=60,          # 1 min — critic responses are short, fail fast
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_prompt),
    ])
    raw = response.content.strip()

    # Strip accidental markdown fences (some models add them despite instructions)
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    try:
        data = json.loads(raw)
        verdict = data.get("verdict", "fail").lower()
        if verdict not in ("pass", "fail"):
            verdict = "fail"
        return GradeResult(
            verdict=verdict,
            score=float(data.get("score", 0.0)),
            reason=str(data.get("reason", "No reason provided")),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        log.warning("critic_parse_error", raw=raw[:200], error=str(e))
        # Conservative default: fail so pipeline retries
        return GradeResult(verdict="fail", score=0.0, reason=f"Parse error — will retry: {e}")


# ── Public grading functions ──────────────────────────────────────────────────

def grade_relevance(query: str, docs: List[Document]) -> GradeResult:
    """Grade whether retrieved documents are relevant to the query."""
    from src.chains.generator import format_context
    context = format_context(docs)
    prompt = RELEVANCE_TEMPLATE.format(question=query, context=context)

    log.info("grading_relevance", query=query[:60])
    result = _call_critic(RELEVANCE_SYSTEM, prompt)
    log.info("relevance_grade",
             verdict=result.verdict, score=result.score, reason=result.reason[:80])
    return result


def grade_hallucination(query: str, docs: List[Document], answer: str) -> GradeResult:
    """Grade whether the generated answer is grounded in the retrieved docs."""
    from src.chains.generator import format_context
    context = format_context(docs)
    prompt = HALLUCINATION_TEMPLATE.format(context=context, answer=answer)

    log.info("grading_hallucination", query=query[:60])
    result = _call_critic(HALLUCINATION_SYSTEM, prompt)
    log.info("hallucination_grade",
             verdict=result.verdict, score=result.score, reason=result.reason[:80])
    return result
