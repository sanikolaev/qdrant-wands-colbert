# Qdrant + WANDS + ColBERTv2 demo

Minimal local demo comparing:

1. **ordinary dense vector search**: one BGE vector per product;
2. **late-interaction ColBERTv2 search in Qdrant**: one vector per token, scored with Qdrant multivectors and `MAX_SIM`.

The dataset is [Wayfair WANDS](https://github.com/wayfair/WANDS): product search queries, products, and relevance labels.

## Why this is useful

Dense search compresses a whole product title/description into one vector. ColBERTv2 keeps a matrix of token vectors and lets query tokens match different product tokens independently. This usually helps multi-constraint product queries such as `chair and a half recliner`, where every term matters.

On a local sample run in this repository:

```text
Queries: 10 | Products indexed: 500 | topK: 10
Dense     nDCG@10: 0.664 | Recall@10: 0.313 | mean search: 0.8 ms
ColBERTv2 nDCG@10: 0.883 | Recall@10: 0.422 | mean search: 41.4 ms
```

Example query:

```text
'chair and a half recliner': dense nDCG=0.249, ColBERT nDCG=0.927, delta=+0.678
```

The trade-off is visible too: ColBERT is more accurate on this sample, but slower and stores many vectors per item.

## Ten queries where ColBERT wins clearly

A larger real-model run evaluated 80 randomly sampled WANDS queries against the same 6,000-product candidate pool. The pool contained all 5,065 products judged relevant to those queries plus 935 distractors. These are the ten largest positive nDCG@10 deltas:

| Query | Dense nDCG@10 | ColBERT nDCG@10 | Delta |
|---|---:|---:|---:|
| `farmhouse bread box` | 0.342 | 0.799 | **+0.457** |
| `kitchen wooden stand` | 0.056 | 0.450 | **+0.393** |
| `outdoor waterproof chest` | 0.212 | 0.539 | **+0.327** |
| `edge chair mat` | 0.606 | 0.933 | **+0.327** |
| `wisdom stone river 3-3/4` | 0.250 | 0.548 | **+0.297** |
| `big basket for dirty cloths` | 0.462 | 0.745 | **+0.284** |
| `desk for kids tjat ate 10 year old` | 0.717 | 1.000 | **+0.283** |
| `living room ideas` | 0.471 | 0.672 | **+0.201** |
| `chinese flower stand` | 0.404 | 0.595 | **+0.192** |
| `wainscoting ideas` | 0.395 | 0.586 | **+0.192** |

This is deliberately a list of the strongest ColBERT examples, not an aggregate claim. Across all 80 sampled queries, ColBERT won 34, dense won 20, and 26 tied. Mean nDCG@10 was 0.752 for ColBERT versus 0.729 for dense. Mean Qdrant search latency was 482.9 ms versus 4.3 ms respectively on this local embedded run. The generated detailed report, including ranked products and WANDS relevance gains, is written to the ignored runtime file `data/colbert_vs_dense_80q.json`.

## Run the web demo

This is the interactive browser demo: type your own query and compare results side-by-side. The recommended launch starts both Qdrant and the application:

```bash
git clone https://github.com/sanikolaev/qdrant-wands-colbert.git
cd qdrant-wands-colbert
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8000
```

The left side shows **ColBERTv2 / late interaction** over Qdrant multivectors. The right side shows **dense / single-vector** search over the same products. The default Compose configuration indexes 6,000 products and exposes all 80 labelled queries used by the comparison.

### Persistent initialization

The first `docker compose up` automatically downloads WANDS, downloads the models, and initializes both Qdrant collections. Three named Docker volumes persist this state:

- `qdrant_storage` — dense and ColBERT collections;
- `wands_data` — downloaded WANDS CSV files;
- `model_cache` — downloaded embedding models (`FASTEMBED_CACHE_PATH=/models/fastembed`).

On later starts, the application validates the stored index signature and point counts. If they match the configured encoder/query/document/seed inputs, it logs `Reusing existing Qdrant index ...` and skips embedding and upsert. Changing those inputs rebuilds the collections. Normal `docker compose down` keeps all data; the operator removes it explicitly with:

Dataset download and index creation are serialized by a lock file in the shared `wands_data` volume, so concurrent Compose app processes cannot delete or mix each other's collections.

```bash
docker compose down -v
```

Ports and corpus size can be overridden without editing the file, for example:

```bash
DEMO_PORT=8080 DEMO_DOCS=500 DEMO_QUERIES=10 docker compose up --build
```

For development without Docker, `web_demo.py` still defaults to Qdrant's embedded local `:memory:` mode:

```bash
uv sync
HF_HUB_DISABLE_XET=1 uv run python web_demo.py --docs 500 --queries 10 --top-k 10
```

## Run the CLI benchmark

```bash
uv sync
docker compose up -d qdrant
HF_HUB_DISABLE_XET=1 uv run python demo.py --queries 10 --docs 500 --top-k 10
```

The web and CLI scripts download three WANDS CSV files into `./data/wands/`, download the embedding models via FastEmbed, create two Qdrant collections, and index the same products in both modes. The CLI additionally prints nDCG/recall/latency over WANDS-labelled queries.

Collections created:

- `wands_dense` — named dense vector `dense`, cosine distance.
- `wands_colbert` — named multivector `colbert`, cosine distance, `MAX_SIM` comparator.

## Fully local smoke mode

If Docker Hub or Hugging Face is temporarily unavailable, this verifies the same Qdrant collection/query mechanics with deterministic hash embeddings:

```bash
uv sync
uv run python demo.py --qdrant-url :memory: --encoder hash --queries 2 --docs 80 --top-k 5
```

This mode is only a smoke test. It is not ColBERTv2 and should not be used for quality claims.

## Useful knobs

```bash
uv run python demo.py \
  --qdrant-url http://localhost:6333 \
  --queries 25 \
  --docs 1000 \
  --top-k 10 \
  --batch-size 8
```

- `--queries`: number of WANDS queries with relevance labels.
- `--docs`: number of products to index. The loader includes relevant products for selected queries first, then adds distractors.
- `--encoder fastembed`: real `BAAI/bge-small-en-v1.5` + `colbert-ir/colbertv2.0` path.
- `--encoder hash`: offline smoke path.
- `--qdrant-url :memory:`: embedded local Qdrant client for smoke/CI-style runs.

## Notes

- The first real run can take a few minutes because ColBERTv2 embeddings are CPU-heavy and model files need to be downloaded.
- Docker Compose startup, first-run initialization, persistent model/data/index volumes, and restart reuse have been verified end-to-end with both hash smoke embeddings and the real BGE + ColBERTv2 models.
- `HF_HUB_DISABLE_XET=1` avoids optional Hugging Face Xet download behavior and made unauthenticated downloads reliable in this environment.
