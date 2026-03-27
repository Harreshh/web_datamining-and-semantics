#!/usr/bin/env python3
"""Phase 1: Crawl web pages and extract entities + lightweight relations.

Outputs:
- data/raw/crawler_output.jsonl
- data/processed/extracted_knowledge.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import httpx
import trafilatura

try:
    import spacy
except ImportError as exc:  # pragma: no cover
    raise SystemExit("spaCy is required. Install dependencies from requirements.txt") from exc

DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/University_of_Oxford",
    "https://en.wikipedia.org/wiki/CERN",
    "https://en.wikipedia.org/wiki/United_Nations",
    "https://en.wikipedia.org/wiki/New_York_City",
]

TARGET_ENTITY_LABELS = {"PERSON", "ORG", "GPE"}


@dataclass
class RelationTriple:
    subject: str
    predicate: str
    object: str
    sentence: str
    url: str
    subject_label: str
    object_label: str


def slugify(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_]", "", value)
    return value[:120] or "Unknown"


def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        fallback = "en_core_web_sm"
        try:
            print(f"[WARN] Model {model_name} not found. Falling back to {fallback}.")
            return spacy.load(fallback)
        except OSError as exc:
            raise SystemExit(
                "No spaCy model found. Install one with: python -m spacy download en_core_web_trf"
            ) from exc


def fetch_clean_text(url: str, timeout_seconds: int = 20) -> str:
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if text:
        return text

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return trafilatura.extract(response.text) or ""


def extract_relations(doc, url: str) -> List[RelationTriple]:
    triples: List[RelationTriple] = []
    seen = set()

    for sent in doc.sents:
        ents = [e for e in sent.ents if e.label_ in TARGET_ENTITY_LABELS]
        if len(ents) < 2:
            continue

        predicate = None
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ in {"ROOT", "nsubj", "dobj", "pobj", "attr"}:
                predicate = token.lemma_.lower()
                break
        if not predicate:
            predicate = "related_to"

        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                s = ents[i]
                o = ents[j]
                if s.text.strip().lower() == o.text.strip().lower():
                    continue
                key = (s.text.strip(), predicate, o.text.strip(), sent.text.strip())
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    RelationTriple(
                        subject=s.text.strip(),
                        predicate=predicate,
                        object=o.text.strip(),
                        sentence=sent.text.strip(),
                        url=url,
                        subject_label=s.label_,
                        object_label=o.label_,
                    )
                )
    return triples


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_knowledge_csv(path: Path, triples: List[RelationTriple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "predicate",
                "object",
                "sentence",
                "url",
                "subject_label",
                "object_label",
                "subject_id",
                "object_id",
            ],
        )
        writer.writeheader()
        for t in triples:
            writer.writerow(
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "sentence": t.sentence,
                    "url": t.url,
                    "subject_label": t.subject_label,
                    "object_label": t.object_label,
                    "subject_id": slugify(t.subject),
                    "object_id": slugify(t.object),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl URLs and extract relation triples")
    parser.add_argument("--urls", nargs="*", help="Optional list of URLs")
    parser.add_argument("--min-words", type=int, default=500, help="Minimum words to keep a document")
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=Path("data/raw/crawler_output.jsonl"),
        help="Output JSONL for cleaned pages",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/processed/extracted_knowledge.csv"),
        help="Output CSV with extracted triples",
    )
    parser.add_argument("--spacy-model", default="en_core_web_trf", help="spaCy model name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = args.urls if args.urls else DEFAULT_URLS

    nlp = load_spacy_model(args.spacy_model)

    raw_records: List[dict] = []
    all_triples: List[RelationTriple] = []

    for url in urls:
        print(f"[INFO] Fetching: {url}")
        try:
            text = fetch_clean_text(url)
        except Exception as exc:
            print(f"[WARN] Failed {url}: {exc}")
            continue

        word_count = len(text.split())
        if word_count < args.min_words:
            print(f"[INFO] Skipping {url} (<{args.min_words} words)")
            continue

        doc = nlp(text)
        triples = extract_relations(doc, url)
        entities = [
            {"text": ent.text, "label": ent.label_, "id": slugify(ent.text)}
            for ent in doc.ents
            if ent.label_ in TARGET_ENTITY_LABELS
        ]

        raw_records.append(
            {
                "url": url,
                "word_count": word_count,
                "text": text,
                "entities": entities,
                "relations": [t.__dict__ for t in triples],
            }
        )
        all_triples.extend(triples)

        print(f"[INFO] Extracted entities={len(entities)} relations={len(triples)}")

    write_jsonl(args.jsonl_out, raw_records)
    write_knowledge_csv(args.csv_out, all_triples)

    unique_entities = {slugify(t.subject) for t in all_triples} | {slugify(t.object) for t in all_triples}
    print("\n[SUMMARY]")
    print(f"Documents kept: {len(raw_records)}")
    print(f"Triples extracted: {len(all_triples)}")
    print(f"Unique entities: {len(unique_entities)}")
    print(f"JSONL: {args.jsonl_out}")
    print(f"CSV:   {args.csv_out}")


if __name__ == "__main__":
    main()
