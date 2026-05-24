# 🧠 Self-Healing RAG Pipeline

> **A production-grade RAG system that thinks, critiques itself, detects hallucinations, and self-corrects — entirely on your local machine. No API keys. No cloud costs. No data leaves your device.**

---

## 🔍 What Problem Does It Solve?

Standard RAG pipelines are **fragile and blind**:

- They retrieve documents and generate answers — but never check if the answer is actually grounded in what was retrieved
- If retrieval returns irrelevant chunks, the LLM hallucinates confidently
- There's no retry mechanism — one bad retrieval = one bad answer, permanently
- They require cloud LLMs (OpenAI, Bedrock) — expensive, privacy-invasive, and rate-limited

**Self-Healing RAG fixes all of this:**

| Problem | How It's Fixed |
|---|---|
| LLM hallucinates | Hallucination critic grades every answer before it reaches the user |
| Bad retrieval | Relevance critic rejects irrelevant docs and triggers query rewrite |
| No retry logic | LangGraph loops back up to 3 times with progressively better queries |
| Cloud dependency | Runs 100% locally via Ollama — zero cost, zero privacy risk |
| Silent failures | Graceful fallback message instead of broken/empty response |

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                    SELF-HEALING RAG PIPELINE                     ║
║                  100% Local · Zero Cost · Private                ║
╚══════════════════════════════════════════════════════════════════╝

         🧑 User Query
              │
              ▼
   ┌─────────────────────┐
   │   🔄 Query Rewriter  │  ←─────────────────────────────┐
   │  reformulates query  │                                 │
   │  on each retry pass  │                                 │
   └──────────┬──────────┘                                 │
              │                                             │
              ▼                                             │
   ┌─────────────────────┐                                 │
   │  🗄️  Vector Retrieval │                                 │
   │  ChromaDB + MiniLM   │                                 │
   │  Top-K doc chunks    │                                 │
   └──────────┬──────────┘                                 │
              │                                             │
              ▼                                             │
   ┌─────────────────────┐    ❌ FAIL                      │
   │  🔎 Relevance Critic │ ──────────────► ┌────────────┐ │
   │  Are docs relevant   │                 │ 🔁 Increment│ │
   │  to the query?       │                 │   Retry    │ │
   └──────────┬──────────┘                 └──────┬─────┘ │
              │ ✅ PASS                            │       │
              ▼                          retries < max?    │
   ┌─────────────────────┐                   YES  │  NO   │
   │  🤖 LLM Generator   │  ◄────────────────────┘   │   │
   │  Ollama · TinyLlama  │                            │   │
   │  tinyllama / phi3    │                        ┌───▼───┴──┐
   └──────────┬──────────┘                        │ 🛡️ Graceful│
              │                                   │  Fallback │
              ▼                                   └───────────┘
   ┌─────────────────────┐    ❌ FAIL
   │  🧪 Hallucination    │ ──────────────► ┌────────────┐
   │     Critic           │                 │ 🔁 Increment│
   │  Is answer grounded  │                 │   Retry    │
   │  in the docs?        │                 └──────┬─────┘
   └──────────┬──────────┘                        │
              │ ✅ PASS                     retries < max?
              ▼                                YES  │  NO
   ┌─────────────────────┐  ◄────────────────────┘   │
   │  ✅ Finalize Answer  │                        ┌───▼──────┐
   │  Approved by all     │                        │ 🛡️ Graceful│
   │  critics             │                        │  Fallback │
   └──────────┬──────────┘                        └───────────┘
              │
              ▼
         🧑 Final Answer
         (grounded · verified · hallucination-free)
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| 🔗 Orchestration | LangGraph | Stateful cyclical graph — enables retry loops |
| 🤖 LLM | Ollama · `tinyllama` | 100% local, free, no API key needed |
| 📐 Embeddings | `all-MiniLM-L6-v2` | Lightweight HuggingFace model, runs on CPU |
| 🗄️ Vector Store | ChromaDB | Local persistent vector DB, zero setup |
| ⚡ API Server | FastAPI | Async, fast, auto-documented |
| 🖥️ Frontend | Vanilla HTML/JS | Zero-dependency, loads instantly |

---

## ⚡ Quick Start (GitHub Codespaces)

### 1. Install & Run

```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
```

### 2. Pull the local LLM

```bash
ollama pull tinyllama
```

> 💡 **Model options by RAM:**
> | RAM | Recommended Model | Command |
> |---|---|---|
> | 8GB | tinyllama (1.1B) | `ollama pull tinyllama` |
> | 16GB | phi3 (3.8B) | `ollama pull phi3` |
> | 32GB+ | llama3.1:8b | `ollama pull llama3.1:8b` |

### 3. Start the server

```bash
./scripts/run.sh
```

Open **port 8000** when Codespaces prompts you.

---

## 💡 How Self-Healing Works

Every query goes through **3 LLM calls**:

```
Call 1 → Relevance Critic   (Are the retrieved docs relevant?)
Call 2 → Answer Generator   (Generate grounded answer)
Call 3 → Hallucination Critic (Is the answer factual?)
```

If either critic fails, a dedicated **`increment_retry` node** updates the retry counter (LangGraph state-safe), and the **Query Rewriter** reformulates the search before trying again — up to `MAX_RETRIES` times (default: 3).

---

## 💰 Cost

| Resource | Cost |
|---|---|
| Ollama LLM | **FREE** (runs locally) |
| ChromaDB | **FREE** (local disk) |
| HuggingFace Embeddings | **FREE** (runs on CPU) |
| **Total** | **$0.00** |

---

## 📁 Project Structure

```
self-healing-rag/
├── src/
│   ├── api.py                  # FastAPI server & endpoints
│   ├── agents/
│   │   ├── critic.py           # Relevance + hallucination graders
│   │   └── query_rewriter.py   # Reformulates failed queries
│   ├── chains/
│   │   ├── graph.py            # LangGraph pipeline & retry logic
│   │   └── generator.py        # Ollama answer generation
│   ├── retrieval/
│   │   ├── vectorstore.py      # ChromaDB operations
│   │   └── embeddings.py       # HuggingFace MiniLM embeddings
│   └── utils/
│       ├── config.py           # All settings (model, retries, etc.)
│       └── logging.py          # Structured logging
├── static/
│   └── index.html              # Chat UI
├── scripts/
│   ├── setup.sh                # One-time install
│   ├── run.sh                  # Start server
│   └── ingest.py               # Bulk document ingestion
├── data/sample_docs/           # Sample documents to index
└── tests/                      # Pipeline tests
```

---

## ⚙️ Configuration

Edit `src/utils/config.py` or set environment variables:

```bash
OLLAMA_MODEL=tinyllama        # LLM model to use
MAX_RETRIES=3                 # Max self-healing retry attempts
TOP_K_DOCS=4                  # Docs retrieved per query
MAX_TOKENS=1024               # Max tokens in generated answer
TEMPERATURE=0.1               # LLM temperature (lower = more factual)
CHUNK_SIZE=500                # Document chunk size
```

---

## 🔧 Known Fixes Applied

- ✅ **Recursion limit bug** — `retry_count` now incremented via dedicated `node_increment_retry` node (LangGraph edge functions cannot mutate state)
- ✅ **Frontend JSON crash** — safe `try/catch` around error response parsing
- ✅ **Request timeout** — `AbortController` prevents infinite UI hang
- ✅ **Ollama timeout** — generator (120s) and critic (60s) timeouts prevent silent hangs
- ✅ **Model default** — switched from `llama3.1:8b` (OOM on 16GB) to `tinyllama`
