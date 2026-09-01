#!/usr/bin/env python3
"""Minimal WANDS benchmark for dense vectors vs ColBERTv2 late interaction in Qdrant."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
FILES = {
    "product": "product.csv",
    "query": "query.csv",
    "label": "label.csv",
}
LABEL_GAIN = {"Exact": 3, "Partial": 1, "Irrelevant": 0}


def stable_token_seed(token: str) -> int:
    return int.from_bytes(hashlib.blake2s(token.encode(), digest_size=4).digest(), "little")


class HashTextEmbedding:
    """Tiny deterministic encoder for offline smoke tests; not a semantic model."""

    def __init__(self, size: int = 64):
        self.size = size

    def _token_vec(self, token: str) -> np.ndarray:
        seed = stable_token_seed(token)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.size).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def _dense(self, text: str) -> list[float]:
        tokens = text.lower().split()[:64]
        if not tokens:
            return [0.0] * self.size
        vector = np.mean([self._token_vec(token) for token in tokens], axis=0)
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        return vector.astype(float).tolist()

    def embed(self, texts: list[str]):
        for text in texts:
            yield self._dense(text)

    def query_embed(self, text: str):
        yield self._dense(text)


class HashLateInteractionEmbedding:
    """Token matrix encoder for verifying Qdrant multivectors without HF downloads."""

    def __init__(self, size: int = 64):
        self.size = size

    def _token_vec(self, token: str) -> np.ndarray:
        seed = stable_token_seed(token)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.size).astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)

    def _matrix(self, text: str) -> list[list[float]]:
        tokens = text.lower().split()[:32] or [""]
        return [self._token_vec(token).astype(float).tolist() for token in tokens]

    def embed(self, texts: list[str]):
        for text in texts:
            yield self._matrix(text)

    def query_embed(self, text: str):
        yield self._matrix(text)


@dataclass(frozen=True)
class Product:
    product_id: str
    text: str


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def download_if_missing(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        target = data_dir / filename
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/{filename}"
        partial = target.with_name(f"{target.name}.tmp")
        partial.unlink(missing_ok=True)
        print(f"Downloading {url} -> {target}")
        try:
            urllib.request.urlretrieve(url, partial)
            if not partial.exists() or partial.stat().st_size == 0:
                raise OSError("downloaded file is empty")
            partial.replace(target)
        except Exception as exc:  # noqa: BLE001 - CLI should show a direct actionable error
            partial.unlink(missing_ok=True)
            raise SystemExit(
                f"Failed to download {url}: {exc}\n"
                "Retry with network access or place WANDS product.csv/query.csv/label.csv in --data-dir."
            ) from exc


def read_queries(path: Path, limit: int) -> list[Query]:
    rows: list[Query] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append(Query(query_id=row["query_id"], text=row["query"]))
            if len(rows) >= limit:
                break
    return rows


def read_labels(path: Path, query_ids: set[str]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {qid: {} for qid in query_ids}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            qid = row["query_id"]
            if qid in query_ids:
                qrels[qid][row["product_id"]] = LABEL_GAIN.get(row["label"], 0)
    return qrels


def product_text(row: dict[str, str]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("product_class", ""),
        row.get("category hierarchy", ""),
        row.get("product_description", ""),
        row.get("product_features", ""),
    ]
    return " | ".join(part.strip().replace("\n", " ") for part in parts if part and part.strip())


def read_products(path: Path, needed_ids: set[str], doc_limit: int, seed: int) -> list[Product]:
    rng = random.Random(seed)
    selected: dict[str, Product] = {}
    reservoir: list[Product] = []
    seen_distractors = 0

    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            product_id = row["product_id"]
            product = Product(product_id=product_id, text=product_text(row))
            if product_id in needed_ids:
                selected[product_id] = product
                continue
            if len(reservoir) < doc_limit:
                reservoir.append(product)
            else:
                j = rng.randint(0, seen_distractors)
                if j < doc_limit:
                    reservoir[j] = product
            seen_distractors += 1

    for product in reservoir:
        if len(selected) >= doc_limit:
            break
        selected.setdefault(product.product_id, product)
    return list(selected.values())[:doc_limit]


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def as_list(vector) -> list[float] | list[list[float]]:
    arr = np.asarray(vector)
    return arr.astype(float).tolist()


def encode_dense(model: TextEmbedding, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="Dense embeddings"):
        vectors.extend(as_list(v) for v in model.embed(batch))
    return vectors


def encode_colbert(model: LateInteractionTextEmbedding, texts: list[str], batch_size: int) -> list[list[list[float]]]:
    vectors: list[list[list[float]]] = []
    for batch in tqdm(list(batched(texts, batch_size)), desc="ColBERT token embeddings"):
        vectors.extend(as_list(v) for v in model.embed(batch))
    return vectors


def recreate_collections(client: QdrantClient, dense_size: int, colbert_size: int) -> None:
    for collection in ["wands_dense", "wands_colbert"]:
        try:
            client.delete_collection(collection)
        except (UnexpectedResponse, ValueError):
            pass

    client.create_collection(
        "wands_dense",
        vectors_config={"dense": VectorParams(size=dense_size, distance=Distance.COSINE)},
    )
    client.create_collection(
        "wands_colbert",
        vectors_config={
            "colbert": VectorParams(
                size=colbert_size,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
            )
        },
    )


def upsert(
    client: QdrantClient,
    products: list[Product],
    dense_vectors,
    colbert_vectors,
    batch_size: int,
    index_signature: str | None = None,
) -> None:
    for start in tqdm(range(0, len(products), batch_size), desc="Upsert Qdrant"):
        chunk = products[start : start + batch_size]
        dense_points = []
        colbert_points = []
        for offset, product in enumerate(chunk):
            idx = start + offset
            payload = {"product_id": product.product_id, "text": product.text[:2048]}
            if index_signature is not None:
                payload["index_signature"] = index_signature
            dense_points.append(PointStruct(id=idx, vector={"dense": dense_vectors[idx]}, payload=payload))
            colbert_points.append(PointStruct(id=idx, vector={"colbert": colbert_vectors[idx]}, payload=payload))
        client.upsert("wands_dense", points=dense_points, wait=True)
        client.upsert("wands_colbert", points=colbert_points, wait=True)


def dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    gains = [qrels.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return 0.0 if ideal_dcg == 0 else dcg(gains) / ideal_dcg


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, gain in qrels.items() if gain > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def query_collection(client: QdrantClient, collection: str, using: str, vector, limit: int) -> tuple[list[str], float]:
    started = time.perf_counter()
    result = client.query_points(collection_name=collection, query=vector, using=using, limit=limit, with_payload=True)
    elapsed_ms = (time.perf_counter() - started) * 1000
    ids = [point.payload["product_id"] for point in result.points]
    return ids, elapsed_ms


def make_qdrant_client(qdrant_url: str) -> QdrantClient:
    if qdrant_url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=qdrant_url, timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/wands"))
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL, or :memory: for embedded local smoke tests")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--docs", type=int, default=700)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--encoder",
        choices=("fastembed", "hash"),
        default="fastembed",
        help="fastembed uses BGE + real ColBERTv2; hash is dependency/network-free smoke mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_if_missing(args.data_dir)
    queries = read_queries(args.data_dir / "query.csv", args.queries)
    qrels = read_labels(args.data_dir / "label.csv", {query.query_id for query in queries})
    needed_ids = {doc_id for labels in qrels.values() for doc_id, gain in labels.items() if gain > 0}
    products = read_products(args.data_dir / "product.csv", needed_ids, args.docs, args.seed)
    product_ids = {product.product_id for product in products}
    qrels = {qid: {doc_id: gain for doc_id, gain in labels.items() if doc_id in product_ids} for qid, labels in qrels.items()}
    queries = [query for query in queries if any(gain > 0 for gain in qrels[query.query_id].values())]

    if args.encoder == "fastembed":
        dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
        dense_name = "BAAI/bge-small-en-v1.5"
        colbert_name = "colbert-ir/colbertv2.0"
    else:
        dense_model = HashTextEmbedding()
        colbert_model = HashLateInteractionEmbedding()
        dense_name = "hash dense smoke encoder"
        colbert_name = "hash token-matrix smoke encoder"

    print(f"Loaded {len(products)} products and {len(queries)} queries from WANDS")
    print(f"Dense baseline: {dense_name}, one vector per product")
    print(f"Late interaction: {colbert_name}, one vector per token + Qdrant MAX_SIM")

    texts = [product.text for product in products]
    dense_vectors = encode_dense(dense_model, texts, args.batch_size)
    colbert_vectors = encode_colbert(colbert_model, texts, args.batch_size)

    dense_size = len(dense_vectors[0])
    colbert_size = len(colbert_vectors[0][0])
    client = make_qdrant_client(args.qdrant_url)
    recreate_collections(client, dense_size, colbert_size)
    upsert(client, products, dense_vectors, colbert_vectors, args.batch_size)

    rows = []
    for query in tqdm(queries, desc="Evaluate queries"):
        dense_q = next(dense_model.query_embed(query.text))
        colbert_q = next(colbert_model.query_embed(query.text))
        dense_ids, dense_ms = query_collection(client, "wands_dense", "dense", as_list(dense_q), args.top_k)
        colbert_ids, colbert_ms = query_collection(client, "wands_colbert", "colbert", as_list(colbert_q), args.top_k)
        labels = qrels[query.query_id]
        rows.append(
            {
                "query": query.text,
                "dense_ndcg": ndcg_at_k(dense_ids, labels, args.top_k),
                "colbert_ndcg": ndcg_at_k(colbert_ids, labels, args.top_k),
                "dense_recall": recall_at_k(dense_ids, labels, args.top_k),
                "colbert_recall": recall_at_k(colbert_ids, labels, args.top_k),
                "dense_ms": dense_ms,
                "colbert_ms": colbert_ms,
                "dense_top": dense_ids[:3],
                "colbert_top": colbert_ids[:3],
            }
        )

    def avg(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    print("\n=== WANDS sample results ===")
    print(f"Queries: {len(rows)} | Products indexed: {len(products)} | topK: {args.top_k}")
    print(f"Dense     nDCG@{args.top_k}: {avg('dense_ndcg'):.3f} | Recall@{args.top_k}: {avg('dense_recall'):.3f} | mean search: {avg('dense_ms'):.1f} ms")
    print(f"ColBERTv2 nDCG@{args.top_k}: {avg('colbert_ndcg'):.3f} | Recall@{args.top_k}: {avg('colbert_recall'):.3f} | mean search: {avg('colbert_ms'):.1f} ms")
    print("\nPer-query examples:")
    for row in rows[:5]:
        delta = row["colbert_ndcg"] - row["dense_ndcg"]
        print(f"- {row['query']!r}: dense nDCG={row['dense_ndcg']:.3f}, ColBERT nDCG={row['colbert_ndcg']:.3f}, delta={delta:+.3f}")
        print(f"  dense top3={row['dense_top']}")
        print(f"  colbert top3={row['colbert_top']}")

    print("\nQdrant collections created: wands_dense, wands_colbert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
