#!/bin/bash

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Self-Healing RAG — Starting Server     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
  echo "⚠️  Ollama not running — starting it..."
  ollama serve &>/tmp/ollama.log &
  sleep 3
fi

# Verify model is available
if ! ollama list | grep -q "$MODEL"; then
  echo "⚠️  Model $MODEL not found — pulling now..."
  ollama pull "$MODEL"
fi

echo "LLM      : $MODEL (local, via Ollama)"
echo "Embeddings: all-MiniLM-L6-v2 (local)"
echo "Vector DB : ChromaDB (local)"
echo "Cost      : \$0.00"
echo "Port      : 8000"
echo ""
echo "Press Ctrl+C to stop."
echo ""

uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
