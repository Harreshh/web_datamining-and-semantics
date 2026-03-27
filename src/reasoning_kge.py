#!/usr/bin/env python3
"""Phase 3: SWRL reasoning + KGE training/evaluation."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD

try:
    from owlready2 import DataProperty, Thing, get_ontology, infer_property_values, sync_reasoner_pellet
except Exception:  # pragma: no cover
    get_ontology = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reasoning and train KGE models")
    parser.add_argument("--kb", type=Path, default=Path("data/kb/expanded_kb.nt"))
    parser.add_argument("--kge-dir", type=Path, default=Path("data/kge"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--metrics-out", type=Path, default=Path("data/kge/metrics.csv"))
    parser.add_argument("--neighbors-out", type=Path, default=Path("data/kge/nearest_neighbors.csv"))
    parser.add_argument("--tsne-out", type=Path, default=Path("data/kge/tsne.png"))
    return parser.parse_args()


def run_swrl_reasoning_demo() -> None:
    if get_ontology is None:
        print("[WARN] owlready2 not available; skipping SWRL reasoning demo")
        return

    onto = get_ontology("http://example.org/reasoning.owl")

    with onto:
        class Person(Thing):
            pass

        class OldPerson(Person):
            pass

        class age(DataProperty):
            domain = [Person]
            range = [int]

    alice = onto.Person("Alice")
    alice.age = [72]
    bob = onto.Person("Bob")
    bob.age = [35]

    # SWRL style rule emulation: if age >= 60, classify as OldPerson.
    for person in onto.Person.instances():
        if person.age and int(person.age[0]) >= 60:
            person.is_a.append(onto.OldPerson)

    try:
        infer_property_values()
        sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    except Exception:
        # Pellet may not be installed; manual assignment above still demonstrates rule behavior.
        pass

    olds = [p.name for p in onto.OldPerson.instances()]
    print(f"[INFO] SWRL demo inferred OldPerson instances: {olds}")


def load_kg_triples(kb_path: Path) -> List[Tuple[str, str, str]]:
    g = Graph()
    fmt = "nt" if kb_path.suffix == ".nt" else "turtle"
    g.parse(kb_path, format=fmt)

    triples: List[Tuple[str, str, str]] = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))
    return triples


def split_triples(
    triples: List[Tuple[str, str, str]],
    seed: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    random.Random(seed).shuffle(triples)
    n = len(triples)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    return triples[:train_end], triples[train_end:valid_end], triples[valid_end:]


def write_txt_triples(path: Path, triples: Iterable[Tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{s}\t{p}\t{o}\n")


def run_pykeen(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    model_name: str,
    epochs: int,
) -> Dict[str, float]:
    from pykeen.pipeline import pipeline

    result = pipeline(
        model=model_name,
        training=str(train_path),
        validation=str(valid_path),
        testing=str(test_path),
        training_kwargs={"num_epochs": epochs},
        random_seed=42,
    )

    metrics = {
        "model": model_name,
        "mrr": float(result.metric_results.get_metric("inverse_harmonic_mean_rank")),
        "hits_at_10": float(result.metric_results.get_metric("hits_at_10")),
    }
    return metrics


def generate_tsne_and_neighbors(
    train_path: Path,
    model_name: str,
    epochs: int,
    tsne_out: Path,
    neighbors_out: Path,
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import torch
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    tf = TriplesFactory.from_path(str(train_path), delimiter="\t", create_inverse_triples=False)

    result = pipeline(
        model=model_name,
        training=tf,
        testing=tf,
        training_kwargs={"num_epochs": max(5, min(epochs, 15))},
        random_seed=42,
    )

    model = result.model
    tf = result.training

    entity_labels = list(tf.entity_to_id.keys())
    entity_ids = torch.arange(len(entity_labels))

    with torch.no_grad():
        emb = model.entity_representations[0](indices=entity_ids).cpu().numpy()

    emb_2d = TSNE(n_components=2, random_state=42, init="random", perplexity=min(30, max(5, len(entity_labels) // 5))).fit_transform(emb)

    tsne_out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10, alpha=0.8)
    for idx in range(min(20, len(entity_labels))):
        plt.text(emb_2d[idx, 0], emb_2d[idx, 1], entity_labels[idx].rsplit("/", 1)[-1], fontsize=7)
    plt.title(f"t-SNE of {model_name} entity embeddings")
    plt.tight_layout()
    plt.savefig(tsne_out, dpi=160)

    # Nearest neighbors for a handful of entities.
    neighbors_out.parent.mkdir(parents=True, exist_ok=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    normalized = emb / np.clip(norms, 1e-9, None)
    sim = normalized @ normalized.T

    with neighbors_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "neighbor", "similarity"])
        writer.writeheader()
        for i in range(min(10, len(entity_labels))):
            top = np.argsort(-sim[i])[1:6]
            for j in top:
                writer.writerow(
                    {
                        "entity": entity_labels[i],
                        "neighbor": entity_labels[j],
                        "similarity": f"{sim[i, j]:.4f}",
                    }
                )


def main() -> None:
    args = parse_args()

    run_swrl_reasoning_demo()

    triples = load_kg_triples(args.kb)
    if len(triples) < 30:
        raise SystemExit(f"Not enough URI triples ({len(triples)}). Build/expand KB first.")

    train, valid, test = split_triples(triples, seed=args.seed)

    train_path = args.kge_dir / "train.txt"
    valid_path = args.kge_dir / "valid.txt"
    test_path = args.kge_dir / "test.txt"

    write_txt_triples(train_path, train)
    write_txt_triples(valid_path, valid)
    write_txt_triples(test_path, test)

    print(f"[INFO] Wrote splits: train={len(train)}, valid={len(valid)}, test={len(test)}")

    if args.skip_training:
        print("[INFO] --skip-training set; skipping PyKEEN models")
        return

    metrics: List[Dict[str, float]] = []
    for model_name in ["TransE", "ComplEx"]:
        try:
            metric = run_pykeen(train_path, valid_path, test_path, model_name=model_name, epochs=args.epochs)
            metrics.append(metric)
            print(f"[INFO] {model_name} MRR={metric['mrr']:.4f} Hits@10={metric['hits_at_10']:.4f}")
        except Exception as exc:
            print(f"[WARN] Failed training {model_name}: {exc}")

    if metrics:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "mrr", "hits_at_10"])
            writer.writeheader()
            writer.writerows(metrics)
        print(f"[INFO] Saved metrics: {args.metrics_out}")

        try:
            generate_tsne_and_neighbors(
                train_path=train_path,
                model_name="TransE",
                epochs=args.epochs,
                tsne_out=args.tsne_out,
                neighbors_out=args.neighbors_out,
            )
            print(f"[INFO] Saved t-SNE plot: {args.tsne_out}")
            print(f"[INFO] Saved nearest neighbors: {args.neighbors_out}")
        except Exception as exc:
            print(f"[WARN] Could not generate t-SNE/neighbors: {exc}")


if __name__ == "__main__":
    main()
