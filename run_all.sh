#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MODEL="${MODEL:-gemma:2b}"
EPOCHS="${EPOCHS:-5}"
MAX_EXPAND="${MAX_EXPAND:-50000}"
SKIP_RAG="${SKIP_RAG:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python not found at: $PYTHON_BIN"
  echo "Create venv and install dependencies first."
  exit 1
fi

echo "[INFO] Using Python: $PYTHON_BIN"

echo "[STEP 1/4] Phase 1: Crawl + IE"
"$PYTHON_BIN" src/crawler.py

echo "[STEP 2/4] Phase 2: Build + Align + Expand"
"$PYTHON_BIN" src/kb_builder.py --max-expand-triples "$MAX_EXPAND"

echo "[STEP 3/4] Phase 3: Reasoning + KGE"
"$PYTHON_BIN" src/reasoning_kge.py --epochs "$EPOCHS"

if [[ "$SKIP_RAG" != "1" ]]; then
  echo "[STEP 4/4] Phase 4: RAG smoke test"
  if curl -sS http://localhost:11434/api/tags >/dev/null 2>&1; then
    printf "Who is Marie Curie?\nexit\n" | "$PYTHON_BIN" src/rag_chatbot.py --model "$MODEL" >/tmp/rag_smoke_output.txt || true
    echo "[INFO] RAG smoke command executed."
  else
    echo "[WARN] Ollama server not running at http://localhost:11434."
    echo "[WARN] Skipping RAG smoke test. Start Ollama and rerun for Phase 4 verification."
  fi
else
  echo "[INFO] SKIP_RAG=1, skipping Phase 4."
fi

echo "[CHECK] Validating required output files"
required_files=(
  "data/raw/crawler_output.jsonl"
  "data/processed/extracted_knowledge.csv"
  "data/kb/initial_kb.ttl"
  "data/kb/alignment.ttl"
  "data/kb/ontology.owl"
  "data/kb/expanded_kb.nt"
  "data/kge/train.txt"
  "data/kge/valid.txt"
  "data/kge/test.txt"
  "data/kge/metrics.csv"
  "data/kge/nearest_neighbors.csv"
)

all_ok=1
for f in "${required_files[@]}"; do
  if [[ -s "$f" ]]; then
    echo "[PASS] $f"
  else
    echo "[FAIL] Missing or empty: $f"
    all_ok=0
  fi
done

if [[ $all_ok -eq 1 ]]; then
  echo "[DONE] Pipeline completed and required outputs are present."
else
  echo "[DONE] Pipeline ran but some outputs are missing. Review logs above."
  exit 2
fi
