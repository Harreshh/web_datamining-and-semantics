# Knowledge Graph End-to-End Project

This repository implements a full 4-phase Knowledge Graph (KG) pipeline:

1. Data acquisition + information extraction (crawler + NER + relation extraction)
2. KB construction + alignment + expansion (RDFLib + Wikidata)
3. Reasoning + Knowledge Graph Embedding (OWLReady2 + PyKEEN)
4. SPARQL-RAG chatbot with local LLM (Ollama)

## Project Structure

```text
project-root/
├── data/
│   ├── raw/                # crawler_output.jsonl
│   ├── processed/          # extracted_knowledge.csv
│   ├── kb/                 # initial_kb.ttl, expanded_kb.nt, ontology.owl, alignment.ttl
│   └── kge/                # train.txt, valid.txt, test.txt, metrics.csv
├── src/
│   ├── crawler.py
│   ├── kb_builder.py
│   ├── reasoning_kge.py
│   └── rag_chatbot.py
├── notebooks/
├── family.owl
├── README.md
└── requirements.txt
```

## Environment Setup

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Install spaCy model

```bash
python -m spacy download en_core_web_trf
```

If `en_core_web_trf` is not available on your machine, code falls back to `en_core_web_sm`.

### 3) Install and start Ollama (Phase 4)

- Install Ollama from https://ollama.com/download
- Pull a small model:

```bash
ollama pull gemma:2b
```

Keep Ollama running locally (`http://localhost:11434`).

## Execution Guide

### One-command run (fastest)

```bash
chmod +x run_all.sh
source .venv/bin/activate
EPOCHS=2 MAX_EXPAND=50000 SKIP_RAG=0 ./run_all.sh
```

What this does:
- Runs all 4 phases in order.
- Verifies required output files and prints `PASS` / `FAIL`.
- Skips only the live RAG smoke test if Ollama is not running locally.

Useful flags:
- `EPOCHS=2` for faster KGE smoke run (increase for better metrics)
- `MAX_EXPAND=50000` to control KB expansion size
- `SKIP_RAG=1` to skip Phase 4 during testing

### Phase 1: Crawl + NER + relation extraction

```bash
python src/crawler.py
```

Outputs:
- `data/raw/crawler_output.jsonl`
- `data/processed/extracted_knowledge.csv`

Optional custom run:

```bash
python src/crawler.py --urls https://en.wikipedia.org/wiki/Marie_Curie https://en.wikipedia.org/wiki/CERN --min-words 500
```

### Phase 2: Build, align, and expand KB

```bash
python src/kb_builder.py --max-expand-triples 50000
```

Outputs:
- `data/kb/initial_kb.ttl`
- `data/kb/alignment.ttl`
- `data/kb/ontology.owl`
- `data/kb/expanded_kb.nt`

If network/API limits block linking:

```bash
python src/kb_builder.py --disable-linking
```

### Phase 3: Reasoning + KGE

```bash
python src/reasoning_kge.py --epochs 20
```

Outputs:
- `data/kge/train.txt`
- `data/kge/valid.txt`
- `data/kge/test.txt`
- `data/kge/metrics.csv`
- `data/kge/tsne.png`
- `data/kge/nearest_neighbors.csv`

Fast split-only mode (skip heavy training):

```bash
python src/reasoning_kge.py --skip-training
```

### Phase 4: RAG chatbot (NL question -> SPARQL -> KG answer)

```bash
python src/rag_chatbot.py --model gemma:2b
```

Interactive demo:
- Ask natural language questions.
- Script generates SPARQL.
- If query fails, it tries automatic self-repair.

## Recommended Hardware

- CPU: 4+ cores
- RAM: 16 GB preferred for `en_core_web_trf` + PyKEEN
- Disk: 5+ GB free (models + outputs)
- Optional GPU for faster KGE training

## Report Checklist (6-10 pages)

1. IE section:
- Domain description
- 3 examples of entity ambiguity

2. KB section:
- Final counts: entities/triples/relations
- Alignment examples

3. KGE section:
- TransE vs ComplEx evaluation (MRR, Hits@10)
- t-SNE embedding visualization

4. RAG section:
- At least 5 question evaluations
- Baseline (no RAG) vs SPARQL-RAG comparison

5. Reflection:
- KB noise analysis
- Rule-based vs embedding-based reasoning comparison

## Troubleshooting

- If spaCy model missing: install with `python -m spacy download en_core_web_trf`.
- If Wikidata queries fail: retry later or run `--disable-linking`.
- If PyKEEN training is too slow: reduce `--epochs` or run `--skip-training`.
- If Ollama connection fails: ensure local server is running and model exists.
