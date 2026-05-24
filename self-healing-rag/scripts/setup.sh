#!/bin/bash
set -e

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Self-Healing RAG — Local Setup         ║"
echo "║   Stack: Ollama + ChromaDB + LangGraph   ║"
echo "╚══════════════════════════════════════════╝"
echo ""

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

# ── Step 1: Install Ollama ────────────────────────────────────────────────────
echo "── Step 1: Installing Ollama ──"
if command -v ollama &>/dev/null; then
  echo "✅ Ollama already installed: $(ollama --version)"
else
  echo "   Downloading Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  echo "✅ Ollama installed"
fi

# ── Step 2: Start Ollama server ───────────────────────────────────────────────
echo ""
echo "── Step 2: Starting Ollama server ──"
if pgrep -x "ollama" > /dev/null; then
  echo "✅ Ollama server already running"
else
  ollama serve &>/tmp/ollama.log &
  sleep 3
  echo "✅ Ollama server started (logs: /tmp/ollama.log)"
fi

# ── Step 3: Pull the LLM model ───────────────────────────────────────────────
echo ""
echo "── Step 3: Pulling model: $MODEL ──"
echo "   This downloads the model weights (~5GB for llama3.1:8b)"
echo "   This only happens once — cached after first download"
echo ""
ollama pull "$MODEL"
echo "✅ Model ready: $MODEL"

# ── Step 4: Install Python deps ───────────────────────────────────────────────
echo ""
echo "── Step 4: Installing Python dependencies ──"
pip install -q --upgrade pip
pip install -r requirements.txt
echo "✅ Python dependencies installed"

# ── Step 5: Pre-download embedding model ─────────────────────────────────────
echo ""
echo "── Step 5: Downloading embedding model (all-MiniLM-L6-v2, ~90MB) ──"
python3 -c "
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('all-MiniLM-L6-v2')
test = m.encode(['test'])
print(f'  Embedding model ready (dim: {len(test[0])})')
"
echo "✅ Embeddings ready"

# ── Step 6: Ingest sample documents ──────────────────────────────────────────
echo ""
echo "── Step 6: Indexing sample documents into ChromaDB ──"
python3 scripts/ingest.py
echo "✅ Documents indexed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅ Setup complete! Zero API keys used.  ║"
echo "║                                          ║"
echo "║  Start the server:                       ║"
echo "║    ./scripts/run.sh                      ║"
echo "║                                          ║"
echo "║  Open: http://localhost:8000             ║"
echo "╚══════════════════════════════════════════╝"
echo ""
