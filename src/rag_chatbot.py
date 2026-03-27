#!/usr/bin/env python3
"""Phase 4: KG-aware SPARQL chatbot using Ollama + self-repair loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph
from rdflib.namespace import RDF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG chatbot over a local RDF knowledge graph")
    parser.add_argument("--kb", type=Path, default=Path("data/kb/expanded_kb.nt"))
    parser.add_argument("--model", default="gemma:2b", help="Ollama model name")
    parser.add_argument("--max-retries", type=int, default=2, help="SPARQL self-repair retries")
    return parser.parse_args()


def build_schema_summary(graph: Graph, max_items: int = 40) -> str:
    predicates = sorted({str(p) for _, p, _ in graph})[:max_items]
    classes = sorted({str(o) for _, _, o in graph.triples((None, RDF.type, None))})

    prefixes = []
    for prefix, ns in graph.namespace_manager.namespaces():
        prefixes.append(f"PREFIX {prefix}: <{ns}>")

    return "\n".join(
        [
            "Schema Summary",
            "--------------",
            "Prefixes:",
            *prefixes[:20],
            "",
            "Predicates:",
            *predicates[:max_items],
            "",
            "Classes:",
            *classes[:max_items],
        ]
    )


def ollama_generate(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def extract_sparql(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for p in parts:
            if "SELECT" in p.upper() or "ASK" in p.upper() or "CONSTRUCT" in p.upper():
                return p.replace("sparql", "").strip()
    return cleaned


def generate_query(model: str, schema: str, question: str) -> str:
    prompt = f"""
You are a SPARQL generator for an RDF graph.
Use only predicates/classes from this schema summary:
{schema}

Task: Convert the user question to one valid SPARQL query.
Rules:
1) Return only the SPARQL query.
2) Use LIMIT 20 unless user asks for more.
3) Prefer SELECT queries.

Question: {question}
"""
    return extract_sparql(ollama_generate(model, prompt))


def repair_query(model: str, schema: str, old_query: str, error_msg: str) -> str:
    prompt = f"""
The SPARQL query below failed. Repair it.
Schema:
{schema}

Broken query:
{old_query}

Error:
{error_msg}

Return only a corrected SPARQL query.
"""
    return extract_sparql(ollama_generate(model, prompt))


def execute_query(graph: Graph, query: str) -> Tuple[bool, str, List[dict]]:
    try:
        results = graph.query(query)
    except Exception as exc:
        return False, str(exc), []

    rows = []
    for row in results:
        rows.append({str(var): str(row[var]) for var in results.vars})
    return True, "", rows


def chat_loop(graph: Graph, model: str, schema: str, max_retries: int) -> None:
    print("\nSPARQL-RAG Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not question:
            continue

        try:
            query = generate_query(model, schema, question)
        except requests.RequestException as exc:
            print("Bot: Cannot reach Ollama at http://localhost:11434.")
            print("Bot: Start Ollama and pull the model, e.g. `ollama pull gemma:2b`.")
            print(f"Bot: Technical error: {exc}\n")
            continue
        print(f"\nGenerated SPARQL:\n{query}\n")

        ok, err, rows = execute_query(graph, query)
        retries = 0
        while not ok and retries < max_retries:
            retries += 1
            print(f"[WARN] Query failed. Attempting repair #{retries}.")
            query = repair_query(model, schema, query, err)
            ok, err, rows = execute_query(graph, query)

        if not ok:
            print(f"Bot: Could not execute query after {max_retries} retries.\nError: {err}\n")
            continue

        if not rows:
            print("Bot: No results found.\n")
            continue

        print("Bot results:")
        for idx, row in enumerate(rows[:20], start=1):
            print(f"{idx}. {json.dumps(row, ensure_ascii=False)}")
        print()


def main() -> None:
    args = parse_args()

    g = Graph()
    fmt = "nt" if args.kb.suffix == ".nt" else "turtle"
    g.parse(args.kb, format=fmt)

    schema = build_schema_summary(g)
    print("[INFO] Loaded graph triples:", len(g))

    chat_loop(graph=g, model=args.model, schema=schema, max_retries=args.max_retries)


if __name__ == "__main__":
    main()
