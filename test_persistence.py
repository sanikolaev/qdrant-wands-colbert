import os
import fcntl
import subprocess
import sys
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import web_demo
import pytest

from demo import Product, download_if_missing


class RecordingEmbedding:
    def query_embed(self, query_text):
        assert query_text == "timed query"
        yield [1.0, 2.0]


def test_query_vectorization_timing_materializes_embedding() -> None:
    vector, elapsed_ms = web_demo.encode_query_timed(RecordingEmbedding(), "timed query")

    assert vector == [1.0, 2.0]
    assert elapsed_ms >= 0


def test_hash_encoder_is_stable_across_python_processes() -> None:
    code = "from demo import HashTextEmbedding; print(next(HashTextEmbedding(4).query_embed('same token')))"
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))

    assert outputs[0] == outputs[1]


def test_index_initialization_lock_is_exclusive(tmp_path) -> None:
    with web_demo.index_initialization_lock(tmp_path):
        with (tmp_path / ".index-initialization.lock").open("a+") as competing_file:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_failed_download_never_leaves_a_reusable_partial_file(tmp_path, monkeypatch) -> None:
    def fail_after_partial_write(_url: str, destination) -> None:
        destination.write_bytes(b"partial")
        raise OSError("connection lost")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_after_partial_write)

    with pytest.raises(SystemExit, match="Failed to download"):
        download_if_missing(tmp_path)

    assert not (tmp_path / "product.csv").exists()
    assert not (tmp_path / "product.csv.tmp").exists()


def test_index_signature_is_deterministic_and_configuration_sensitive() -> None:
    products = [Product(product_id="p1", text="alpha"), Product(product_id="p2", text="beta")]

    signature = web_demo.index_signature(products, encoder="hash", query_count=3, seed=42)

    assert signature == web_demo.index_signature(products, encoder="hash", query_count=3, seed=42)
    assert signature != web_demo.index_signature(list(reversed(products)), encoder="hash", query_count=3, seed=42)
    assert signature != web_demo.index_signature(products, encoder="fastembed", query_count=3, seed=42)


def test_reusable_index_requires_both_complete_collections_and_matching_signature() -> None:
    client = QdrantClient(":memory:")
    web_demo.recreate_collections(client, dense_size=4, colbert_size=4)
    signature = "expected-signature"
    payload = {"product_id": "p1", "text": "alpha", "index_signature": signature}
    client.upsert(
        "wands_dense",
        points=[PointStruct(id=0, vector={"dense": [1.0, 0.0, 0.0, 0.0]}, payload=payload)],
        wait=True,
    )
    client.upsert(
        "wands_colbert",
        points=[PointStruct(id=0, vector={"colbert": [[1.0, 0.0, 0.0, 0.0]]}, payload=payload)],
        wait=True,
    )

    assert web_demo.has_reusable_index(client, expected_count=1, signature=signature)
    assert not web_demo.has_reusable_index(client, expected_count=2, signature=signature)
    assert not web_demo.has_reusable_index(client, expected_count=1, signature="other")


def test_upsert_records_signature_in_persistent_payload() -> None:
    client = QdrantClient(":memory:")
    web_demo.recreate_collections(client, dense_size=4, colbert_size=4)
    products = [Product(product_id="p1", text="alpha")]

    web_demo.upsert(
        client,
        products,
        dense_vectors=[[1.0, 0.0, 0.0, 0.0]],
        colbert_vectors=[[[1.0, 0.0, 0.0, 0.0]]],
        batch_size=1,
        index_signature="signature-v1",
    )

    point = client.retrieve("wands_dense", ids=[0], with_payload=True)[0]
    assert point.payload["index_signature"] == "signature-v1"
