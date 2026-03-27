#!/usr/bin/env python3
"""Generate KG-backed answers for the report's 5 RAG evaluation questions."""

from __future__ import annotations

import json
from pathlib import Path

from rdflib import Graph


def as_list(results):
    rows = []
    for row in results:
        rows.append([str(v) for v in row])
    return rows


def main() -> None:
    kb_path = Path("data/kb/expanded_kb.nt")
    g = Graph()
    g.parse(kb_path, format="nt")

    queries = {
        "Q1_Who_is_Marie_Curie": """
            SELECT ?p ?o
            WHERE { <http://example.org/kg/Marie_Curie> ?p ?o }
            LIMIT 20
        """,
        "Q2_Org_linked_to_Einstein": """
            SELECT DISTINCT ?o
            WHERE {
                <http://example.org/kg/Albert_Einstein> ?p ?o .
                FILTER(CONTAINS(LCASE(STR(?o)), "org") || CONTAINS(LCASE(STR(?o)), "university") || CONTAINS(LCASE(STR(?o)), "institute") || CONTAINS(LCASE(STR(?o)), "association"))
            }
            LIMIT 20
        """,
        "Q3_Entities_connected_to_CERN": """
            SELECT DISTINCT ?p ?o
            WHERE { <http://example.org/kg/CERN> ?p ?o }
            LIMIT 20
        """,
        "Q4_Is_NYC_place": """
            ASK {
                <http://example.org/kg/New_York_City> a ?t .
                FILTER(
                    CONTAINS(LCASE(STR(?t)), "place") ||
                    CONTAINS(LCASE(STR(?t)), "city") ||
                    CONTAINS(LCASE(STR(?t)), "location") ||
                    CONTAINS(LCASE(STR(?t)), "settlement")
                )
            }
        """,
        "Q5_Top_related_to_UN": """
            SELECT ?neighbor (COUNT(?p) AS ?support)
            WHERE { <http://example.org/kg/United_Nations> ?p ?neighbor }
            GROUP BY ?neighbor
            ORDER BY DESC(?support)
            LIMIT 20
        """,
    }

    output = {}
    for key, query in queries.items():
        result = g.query(query)
        result_type = str(getattr(result, "type", "SELECT")).upper()
        if result_type == "ASK":
            output[key] = {"type": "ASK", "answer": bool(result.askAnswer)}
        else:
            rows = as_list(result)
            output[key] = {
                "type": "SELECT",
                "row_count": len(rows),
                "rows": rows[:10],
            }

    out_path = Path("data/processed/rag_eval_answers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
