#!/usr/bin/env python3
"""Web UI for comparing WANDS dense search vs ColBERTv2 late interaction in Qdrant."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query as ApiQuery
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from demo import (
    HashLateInteractionEmbedding,
    HashTextEmbedding,
    LateInteractionTextEmbedding,
    TextEmbedding,
    as_list,
    download_if_missing,
    encode_colbert,
    encode_dense,
    make_qdrant_client,
    ndcg_at_k,
    read_labels,
    read_products,
    read_queries,
    recall_at_k,
    recreate_collections,
    upsert,
)

DEFAULT_DATA_DIR = Path("data/wands")
INDEX_SCHEMA_VERSION = 2


HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Qdrant WANDS: ColBERT vs Dense</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#121a2f; --muted:#94a3b8; --text:#e5e7eb; --line:#243149; --accent:#8b5cf6; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; --dense:#38bdf8; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: radial-gradient(circle at top left, #1e1b4b 0, var(--bg) 38rem); color: var(--text); }
    header { padding: 28px 32px 18px; border-bottom: 1px solid rgba(255,255,255,0.08); }
    h1 { margin:0 0 8px; font-size: clamp(28px, 4vw, 46px); letter-spacing:-0.04em; }
    .sub { color: var(--muted); max-width: 980px; line-height: 1.45; }
    main { padding: 24px 32px 40px; max-width: 1600px; margin: 0 auto; }
    form { display:flex; gap:12px; align-items:center; margin-bottom: 16px; }
    input[type="search"] { flex: 1; min-width: 0; border: 1px solid var(--line); background: rgba(15,23,42,0.92); color: var(--text); border-radius: 14px; padding: 16px 18px; font-size: 18px; outline: none; }
    input[type="search"]:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(139,92,246,0.25); }
    button { border:0; border-radius: 14px; padding: 16px 22px; background: var(--accent); color:white; font-weight: 700; font-size: 16px; cursor:pointer; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .suggestions { display:flex; flex-wrap: wrap; gap: 8px; margin: 0 0 20px; }
    .suggestions button { background: rgba(255,255,255,0.08); color: var(--text); border: 1px solid var(--line); padding: 9px 12px; font-size: 13px; font-weight: 600; }
    .query-nav { display:flex; align-items:center; gap:10px; margin:0 0 16px; }
    .query-nav button { background:rgba(255,255,255,0.08); color:var(--text); border:1px solid var(--line); padding:9px 14px; font-size:13px; }
    .query-position { min-width:82px; text-align:center; color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }
    .status { min-height: 24px; color: var(--muted); margin-bottom: 16px; }
    .legend { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:-4px 0 18px; color:var(--muted); font-size:12px; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items:start; }
    .column { background: rgba(18,26,47,0.9); border: 1px solid rgba(255,255,255,0.08); border-radius: 22px; overflow:hidden; box-shadow: 0 18px 60px rgba(0,0,0,0.28); }
    .column header { padding: 18px 20px; border-bottom:1px solid var(--line); background: rgba(255,255,255,0.03); display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
    .column h2 { margin:0; font-size: 23px; letter-spacing:-0.03em; }
    .colbert h2 { color:#c4b5fd; }
    .dense h2 { color:#7dd3fc; }
    .metric { color: var(--muted); font-size: 13px; white-space:nowrap; }
    .results { display:flex; flex-direction:column; }
    .card { padding: 16px 18px; border-bottom:1px solid rgba(255,255,255,0.07); }
    .card:last-child { border-bottom:0; }
    .rank { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
    .badge { width:30px; height:30px; border-radius: 999px; display:grid; place-items:center; font-weight:800; background: rgba(255,255,255,0.08); color: white; }
    .title { font-weight:750; line-height:1.25; }
    .meta { display:flex; flex-wrap:wrap; gap:8px; margin: 10px 0; color: var(--muted); font-size: 12px; }
    .pill { border: 1px solid var(--line); background:rgba(255,255,255,0.04); border-radius:999px; padding:4px 8px; }
    .relevance { font-weight:800; letter-spacing:.01em; }
    .relevance-exact { border-color:rgba(34,197,94,.55); background:rgba(34,197,94,.14); color:#86efac; }
    .relevance-partial { border-color:rgba(245,158,11,.55); background:rgba(245,158,11,.14); color:#fcd34d; }
    .relevance-irrelevant { border-color:rgba(239,68,68,.55); background:rgba(239,68,68,.14); color:#fca5a5; }
    .relevance-unjudged { border-color:rgba(148,163,184,.35); background:rgba(148,163,184,.08); color:#cbd5e1; }
    .text { color:#cbd5e1; font-size: 14px; line-height:1.42; display:-webkit-box; -webkit-line-clamp:4; -webkit-box-orient: vertical; overflow:hidden; }
    .empty { color: var(--muted); padding: 20px; }
    @media (max-width: 900px) { main, header { padding-left: 16px; padding-right:16px; } form { flex-direction:column; align-items:stretch; } .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>WANDS search: ColBERT vs Dense</h1>
    <div class="sub">Type any product query. The left side uses ColBERTv2 late interaction over Qdrant multivectors; the right side uses ordinary one-vector dense search over the same WANDS products.</div>
  </header>
  <main>
    <form id="searchForm">
      <input id="queryInput" type="search" value="chair and a half recliner" placeholder="Try: chair and a half recliner, outdoor wall lights, round coffee table" autocomplete="off" />
      <button id="searchButton" type="submit">Search</button>
    </form>
    <div class="suggestions" id="suggestions"></div>
    <div class="query-nav" aria-label="Browse labelled WANDS queries">
      <button id="prevQuery" type="button">← Prev</button>
      <span class="query-position" id="queryPosition">— / —</span>
      <button id="nextQuery" type="button">Next →</button>
    </div>
    <div class="status" id="status">Loading index status…</div>
    <div class="legend" aria-label="WANDS relevance legend">
      <span>WANDS relevance:</span>
      <span class="pill relevance relevance-exact">Exact</span>
      <span class="pill relevance relevance-partial">Partial</span>
      <span class="pill relevance relevance-irrelevant">Irrelevant</span>
      <span class="pill relevance relevance-unjudged">Not judged</span>
    </div>
    <section class="grid">
      <article class="column colbert">
        <header><h2>ColBERT / late interaction</h2><div class="metric" id="colbertMetric">—</div></header>
        <div class="results" id="colbertResults"><div class="empty">Submit a query to see ColBERT results.</div></div>
      </article>
      <article class="column dense">
        <header><h2>Dense / single vector</h2><div class="metric" id="denseMetric">—</div></header>
        <div class="results" id="denseResults"><div class="empty">Submit a query to see dense results.</div></div>
      </article>
    </section>
  </main>
  <script>
    const form = document.getElementById('searchForm');
    const input = document.getElementById('queryInput');
    const button = document.getElementById('searchButton');
    const prevQuery = document.getElementById('prevQuery');
    const nextQuery = document.getElementById('nextQuery');
    const queryPosition = document.getElementById('queryPosition');
    const statusEl = document.getElementById('status');
    const suggestionsEl = document.getElementById('suggestions');
    const colbertResults = document.getElementById('colbertResults');
    const denseResults = document.getElementById('denseResults');
    const colbertMetric = document.getElementById('colbertMetric');
    const denseMetric = document.getElementById('denseMetric');
    let labelledQueries = [];
    let queryIndex = -1;

    function syncQueryPosition(query) {
      const match = labelledQueries.findIndex(item => item.toLowerCase() === query.toLowerCase());
      if (match >= 0) queryIndex = match;
      queryPosition.textContent = match >= 0 ? `${match + 1} / ${labelledQueries.length}` : `— / ${labelledQueries.length}`;
    }

    function browseQuery(direction) {
      if (!labelledQueries.length || button.disabled) return;
      const current = labelledQueries.findIndex(item => item.toLowerCase() === input.value.trim().toLowerCase());
      const start = current >= 0 ? current : queryIndex;
      queryIndex = (start + direction + labelledQueries.length) % labelledQueries.length;
      input.value = labelledQueries[queryIndex];
      syncQueryPosition(input.value);
      runSearch();
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
    }

    function relevanceBadge(relevance) {
      const levels = {
        3: { label: 'Exact', className: 'relevance-exact' },
        1: { label: 'Partial', className: 'relevance-partial' },
        0: { label: 'Irrelevant', className: 'relevance-irrelevant' },
      };
      const level = relevance === null || relevance === undefined
        ? { label: 'Not judged', className: 'relevance-unjudged' }
        : levels[relevance] || { label: `Gain ${relevance}`, className: 'relevance-unjudged' };
      return `<span class="pill relevance ${level.className}">${level.label}</span>`;
    }

    function renderResults(target, items) {
      if (!items.length) {
        target.innerHTML = '<div class="empty">No results.</div>';
        return;
      }
      target.innerHTML = items.map(item => `
        <div class="card">
          <div class="rank"><span class="badge">${item.rank}</span><div class="title">${escapeHtml(item.title)}</div></div>
          <div class="meta">
            <span class="pill">product ${escapeHtml(item.product_id)}</span>
            <span class="pill">score ${Number(item.score).toFixed(3)}</span>
            ${relevanceBadge(item.relevance)}
          </div>
          <div class="text">${escapeHtml(item.text)}</div>
        </div>`).join('');
    }

    async function refreshStatus() {
      const res = await fetch('/api/status');
      const data = await res.json();
      labelledQueries = data.queries;
      syncQueryPosition(input.value.trim());
      statusEl.textContent = `Indexed ${data.products} WANDS products with ${data.encoder}; Qdrant: ${data.qdrant_url}; topK: ${data.top_k}`;
      suggestionsEl.innerHTML = data.sample_queries.map(q => `<button type="button" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join('');
      suggestionsEl.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => { input.value = btn.dataset.query; runSearch(); }));
    }

    async function runSearch() {
      const q = input.value.trim();
      if (!q) return;
      button.disabled = true;
      prevQuery.disabled = true;
      nextQuery.disabled = true;
      statusEl.textContent = `Searching for “${q}”…`;
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        syncQueryPosition(data.query);
        renderResults(colbertResults, data.colbert.results);
        renderResults(denseResults, data.dense.results);
        colbertMetric.textContent = `${data.colbert.elapsed_ms.toFixed(1)} ms` + (data.colbert.ndcg_at_k !== null ? ` · nDCG ${data.colbert.ndcg_at_k.toFixed(3)}` : '');
        denseMetric.textContent = `${data.dense.elapsed_ms.toFixed(1)} ms` + (data.dense.ndcg_at_k !== null ? ` · nDCG ${data.dense.ndcg_at_k.toFixed(3)}` : '');
        statusEl.textContent = `Query: “${data.query}”` + (data.known_query_id ? ` · WANDS query_id ${data.known_query_id}` : ' · free-form query');
      } catch (err) {
        statusEl.textContent = `Search failed: ${err.message || err}`;
      } finally {
        button.disabled = false;
        prevQuery.disabled = false;
        nextQuery.disabled = false;
      }
    }

    form.addEventListener('submit', event => { event.preventDefault(); runSearch(); });
    prevQuery.addEventListener('click', () => browseQuery(-1));
    nextQuery.addEventListener('click', () => browseQuery(1));
    refreshStatus().then(runSearch).catch(err => { statusEl.textContent = `Startup failed: ${err.message || err}`; });
  </script>
</body>
</html>
"""


@dataclass
class SearchHit:
    rank: int
    product_id: str
    score: float
    title: str
    text: str
    relevance: int | None


@dataclass
class DemoState:
    client: QdrantClient
    dense_model: Any
    colbert_model: Any
    qrels_by_text: dict[str, tuple[str, dict[str, int]]]
    sample_queries: list[str]
    query_texts: list[str]
    encoder_name: str
    qdrant_url: str
    product_count: int
    top_k: int


_STATE: DemoState | None = None
_STATE_LOCK = threading.Lock()


def parse_title(text: str) -> str:
    return text.split(" | ", 1)[0].strip() or text[:80]


def index_signature(products: list[Any], encoder: str, query_count: int, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema": INDEX_SCHEMA_VERSION,
                "encoder": encoder,
                "query_count": query_count,
                "seed": seed,
                "dense_model": "BAAI/bge-small-en-v1.5",
                "colbert_model": "colbert-ir/colbertv2.0",
            },
            sort_keys=True,
        ).encode()
    )
    for product in products:
        digest.update(b"\0")
        digest.update(product.product_id.encode())
        digest.update(b"\0")
        digest.update(product.text.encode())
    return digest.hexdigest()


def has_reusable_index(client: QdrantClient, expected_count: int, signature: str) -> bool:
    try:
        for collection in ("wands_dense", "wands_colbert"):
            if not client.collection_exists(collection):
                return False
            if client.count(collection, exact=True).count != expected_count:
                return False
            points = client.retrieve(collection, ids=[0], with_payload=True)
            if not points or (points[0].payload or {}).get("index_signature") != signature:
                return False
        return True
    except Exception:
        return False


def wait_for_qdrant(client: QdrantClient, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            client.get_collections()
            return
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Qdrant did not become ready within {timeout_seconds}s") from exc
            time.sleep(1)


@contextlib.contextmanager
def index_initialization_lock(data_dir: Path):
    """Serialize dataset/index initialization across Compose app processes."""
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / ".index-initialization.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def query_hits(client: QdrantClient, collection: str, using: str, vector: Any, labels: dict[str, int] | None, limit: int) -> tuple[list[SearchHit], list[str], float]:
    started = time.perf_counter()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        using=using,
        limit=limit,
        with_payload=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    hits: list[SearchHit] = []
    ranked_ids: list[str] = []
    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}
        product_id = str(payload.get("product_id", point.id))
        text = str(payload.get("text", ""))
        ranked_ids.append(product_id)
        hits.append(
            SearchHit(
                rank=rank,
                product_id=product_id,
                score=float(point.score),
                title=parse_title(text),
                text=text,
                relevance=labels.get(product_id, 0) if labels is not None else None,
            )
        )
    return hits, ranked_ids, elapsed_ms


def build_state(args: argparse.Namespace) -> DemoState:
    with index_initialization_lock(args.data_dir):
        download_if_missing(args.data_dir)
        queries = read_queries(args.data_dir / "query.csv", args.queries)
        qrels = read_labels(args.data_dir / "label.csv", {query.query_id for query in queries})
        needed_ids = {doc_id for labels in qrels.values() for doc_id, gain in labels.items() if gain > 0}
        products = read_products(args.data_dir / "product.csv", needed_ids, args.docs, args.seed)
        product_ids = {product.product_id for product in products}
        qrels = {qid: {doc_id: gain for doc_id, gain in labels.items() if doc_id in product_ids} for qid, labels in qrels.items()}
        queries = [query for query in queries if any(gain > 0 for gain in qrels[query.query_id].values())]
        if not queries:
            raise RuntimeError("No labelled WANDS queries remain after sampling products; increase --queries or --docs")

        dense_model: Any
        colbert_model: Any
        if args.encoder == "fastembed":
            dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            colbert_model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
            encoder_name = "BGE dense + ColBERTv2"
        else:
            dense_model = HashTextEmbedding()
            colbert_model = HashLateInteractionEmbedding()
            encoder_name = "hash smoke encoder"

        client = make_qdrant_client(args.qdrant_url)
        wait_for_qdrant(client)
        signature = index_signature(products, args.encoder, len(queries), args.seed)
        if has_reusable_index(client, len(products), signature):
            print(f"Reusing existing Qdrant index with {len(products)} WANDS products", flush=True)
        else:
            texts = [product.text for product in products]
            print(f"Initializing Qdrant index with {len(products)} WANDS products using {encoder_name}", flush=True)
            dense_vectors = encode_dense(dense_model, texts, args.batch_size)
            colbert_vectors = encode_colbert(colbert_model, texts, args.batch_size)
            recreate_collections(client, len(dense_vectors[0]), len(colbert_vectors[0][0]))
            upsert(client, products, dense_vectors, colbert_vectors, args.batch_size, index_signature=signature)

    qrels_by_text = {query.text.lower(): (query.query_id, qrels[query.query_id]) for query in queries}
    sample_queries = [query.text for query in queries[:8]]
    return DemoState(
        client=client,
        dense_model=dense_model,
        colbert_model=colbert_model,
        qrels_by_text=qrels_by_text,
        sample_queries=sample_queries,
        query_texts=[query.text for query in queries],
        encoder_name=encoder_name,
        qdrant_url=args.qdrant_url,
        product_count=len(products),
        top_k=args.top_k,
    )


def get_state() -> DemoState:
    with _STATE_LOCK:
        if _STATE is None:
            raise HTTPException(status_code=503, detail="Demo is still indexing")
        return _STATE


def create_app(state: DemoState | None = None) -> FastAPI:
    global _STATE
    if state is not None:
        _STATE = state

    app = FastAPI(title="Qdrant WANDS ColBERT vs Dense Demo")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        state = get_state()
        return {
            "encoder": state.encoder_name,
            "products": state.product_count,
            "qdrant_url": state.qdrant_url,
            "sample_queries": state.sample_queries,
            "queries": state.query_texts,
            "top_k": state.top_k,
        }

    @app.get("/api/search")
    def search(q: str = ApiQuery(..., min_length=1)) -> dict[str, Any]:
        state = get_state()
        query_text = q.strip()
        if not query_text:
            raise HTTPException(status_code=400, detail="Query must not be empty")
        query_id, labels = state.qrels_by_text.get(query_text.lower(), (None, None))
        dense_q = as_list(next(state.dense_model.query_embed(query_text)))
        colbert_q = as_list(next(state.colbert_model.query_embed(query_text)))
        dense_hits, dense_ids, dense_ms = query_hits(state.client, "wands_dense", "dense", dense_q, labels, state.top_k)
        colbert_hits, colbert_ids, colbert_ms = query_hits(state.client, "wands_colbert", "colbert", colbert_q, labels, state.top_k)

        def metrics(ids: list[str], elapsed_ms: float, hits: list[SearchHit]) -> dict[str, Any]:
            has_labels = labels is not None
            return {
                "elapsed_ms": elapsed_ms,
                "ndcg_at_k": ndcg_at_k(ids, labels, state.top_k) if labels is not None else None,
                "recall_at_k": recall_at_k(ids, labels, state.top_k) if labels is not None else None,
                "results": [hit.__dict__ for hit in hits],
            }

        return {
            "query": query_text,
            "known_query_id": query_id,
            "dense": metrics(dense_ids, dense_ms, dense_hits),
            "colbert": metrics(colbert_ids, colbert_ms, colbert_hits),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--qdrant-url", default=":memory:", help="Qdrant URL, or :memory: for embedded local Qdrant")
    parser.add_argument("--queries", type=int, default=12, help="Labelled WANDS queries to load for examples/metrics")
    parser.add_argument("--docs", type=int, default=700, help="WANDS products to index")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder", choices=("fastembed", "hash"), default="fastembed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def ensure_port_available(host: str, port: int) -> None:
    if port == 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise SystemExit(
                f"Port {host}:{port} is already in use. Pick another port, e.g. --port 50413. ({exc})"
            ) from exc


app = create_app()


def main() -> int:
    global _STATE
    args = parse_args()
    ensure_port_available(args.host, args.port)
    if args.encoder == "fastembed":
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    _STATE = build_state(args)
    print(f"Open http://{args.host}:{args.port} and search side-by-side", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        raise SystemExit(main())
