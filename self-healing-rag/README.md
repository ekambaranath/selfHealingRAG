# 🔄 Self-Healing RAG Pipeline

A production-grade Retrieval-Augmented Generation system that critiques its own output, detects hallucinations, and self-corrects using LangGraph's stateful cyclical workflows.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Rewriter │ ← reformulates on retry
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Retrieval│ ← ChromaDB + HuggingFace embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator  │ ← Claude via AWS Bedrock
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Critic Agent   │ ← Hallucination + Relevance grader
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
   PASS      FAIL (max 3 retries)
    │         │
    ▼         ▼
 Final     Re-retrieve with
 Answer    new query / graceful fallback
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph (stateful, cyclical) |
| LLM | Claude 3.5 Sonnet via AWS Bedrock |
| Embeddings | sentence-transformers (local, free) |
| Vector Store | ChromaDB (local, no API needed) |
| API Server | FastAPI |
| Frontend | Vanilla HTML/JS |

## Quick Start (GitHub Codespaces)

### 1. Set Environment Variables

```bash
export AWS_REGION="us-east-2"
export AWS_BEARER_TOKEN_BEDROCK="your-token-here"
export CLAUDE_CODE_USE_BEDROCK=1
```

### 2. Install & Run

```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
./scripts/run.sh
```

Open port **8000** when Codespaces prompts you.

## 💰 Cost Estimate

**Embeddings: FREE** (local HuggingFace model)
**ChromaDB: FREE** (local disk)
**Only AWS Bedrock calls cost money.**

Claude 3.5 Sonnet: $3/1M input tokens, $15/1M output tokens

| Activity | Cost |
|----------|------|
| Single query (1st-try pass) | ~$0.009 |
| Single query (1 retry) | ~$0.018 |
| 10 test queries | ~$0.09 |
| 50 queries mixed | ~$1.00 |
| Full day testing | ~$2–5 |
