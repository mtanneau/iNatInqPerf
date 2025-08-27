# adaptors/milvus_backend.py
from typing import Dict, Any, Tuple, Sequence
import os
import numpy as np

from .base import VectorBackend

# Milvus / PyMilvus 2.x
try:
    from pymilvus import (
        connections,
        FieldSchema, CollectionSchema, DataType,
        Collection, utility,
    )
except Exception as e:
    raise RuntimeError(
        "pymilvus is required for the Milvus backend. "
        "Add 'pymilvus>=2.4' to pyproject and run `uv sync`."
    ) from e


def _metric_for(metric: str) -> str:
    """Map our metric names to Milvus metric types."""
    m = (metric or "ip").lower()
    if m in ("ip", "inner_product", "dot"):
        return "IP"
    if m in ("cosine",):
        # Milvus >=2.4 supports COSINE; if your server is older, normalize vectors and use IP.
        return "COSINE"
    return "L2"  # default


class MilvusBackend(VectorBackend):
    """
    Milvus backend with configurable index:
      - Supports HNSW and IVF families (IVF_FLAT, IVF_SQ8, IVF_PQ)
      - Uses explicit PK (id: INT64), label: INT64 (optional), embedding: FLOAT_VECTOR
      - Metric: IP / COSINE / L2
    """
    name = "milvus"   # e.g., use "milvus" or "milvus.hnsw" in your configs

    def __init__(self):
        self.dim = None
        self.metric = "IP"
        self.params: Dict[str, Any] = {}
        self.collection: Collection | None = None
        self.collection_name = "bench"
        self.normalize_for_cosine = False  # If server doesn't support COSINE, we normalize + use IP.

    # -----------------------
    # Required interface impl
    # -----------------------
    def init(self, dim: int, metric: str, **params):
        """
        params:
          - host (default "localhost")
          - port (default 19530)
          - uri  (alternative to host/port; optional)
          - collection (default "bench")
          - index_type: "HNSW" | "IVF_FLAT" | "IVF_SQ8" | "IVF_PQ" (default "HNSW")
          - index params (per type):
              HNSW: M (16), efConstruction (100), efSearch (64)
              IVF_*: nlist (1024), nprobe (16)
          - metric: l2|ip|cosine (in config; but we receive normalized metric here)
        """
        self.dim = int(dim)
        self.params = params or {}
        self.collection_name = str(self.params.get("collection", "bench"))

        # Connect
        uri = self.params.get("uri")
        if uri:
            connections.connect(alias="default", uri=uri)
        else:
            host = self.params.get("host", "localhost")
            port = int(self.params.get("port", 19530))
            connections.connect(alias="default", host=host, port=port)

        # Metric handling
        m = _metric_for(metric)
        self.metric = m

        # Server-side COSINE support varies; if unsupported, normalize vectors and use IP
        if m == "COSINE":
            # We’ll still try to use COSINE. If your server is older and rejects it,
            # you can set use_cosine=false & rely on normalization + IP instead.
            self.normalize_for_cosine = False
        else:
            self.normalize_for_cosine = (metric.lower() == "cosine" and m != "COSINE")

        # (Re)create collection schema
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
        )
        label_field = FieldSchema(
            name="label", dtype=DataType.INT64, is_primary=False, auto_id=False
        )
        emb_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema = CollectionSchema(
            fields=[id_field, label_field, emb_field],
            description="Benchmark collection",
            enable_dynamic_field=False,
        )
        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using="default",
            shards_num=int(self.params.get("shards_num", 2)),
        )

        # Build index
        index_type = str(self.params.get("index_type", "HNSW")).upper()
        if index_type == "HNSW":
            idx_params = {
                "M": int(self.params.get("M", self.params.get("m", 16))),
                "efConstruction": int(self.params.get("ef_construct", 100)),
            }
        elif index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            idx_params = {
                "nlist": int(self.params.get("nlist", 1024)),
            }
            if index_type == "IVF_PQ":
                # optional PQ params; Milvus may default if omitted
                m = int(self.params.get("pq_m", self.params.get("m", 8)))
                idx_params["m"] = m
                idx_params["nbits"] = int(self.params.get("nbits", 8))
        else:
            raise ValueError(f"Unsupported index_type: {index_type}")

        self.collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": self.metric,
                "params": idx_params,
            },
        )
        self.collection.load()

    def train(self, X_train: np.ndarray):
        """Milvus builds the index on create_index; no separate train step."""
        return

    def _prep_vecs(self, X: np.ndarray) -> np.ndarray:
        X = X.astype("float32", copy=False)
        if self.normalize_for_cosine:
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
            X = X / n
        return X

    def upsert(self, ids: np.ndarray, X: np.ndarray):
        # Milvus has no single "upsert"; we do a delete-then-insert for overlapping IDs.
        ids = np.asarray(ids, dtype="int64")
        X = self._prep_vecs(np.asarray(X, dtype="float32"))
        labels = np.zeros((ids.shape[0],), dtype="int64")  # optional label placeholder

        # Best-effort delete existing IDs (fast path)
        try:
            expr = f"id in {ids.tolist()}"
            self.collection.delete(expr)
        except Exception:
            pass

        # Insert
        entities = [ids.tolist(), labels.tolist(), X.tolist()]
        self.collection.insert(entities)
        self.collection.flush()

    def delete(self, ids: Sequence[int]):
        arr = np.asarray(list(ids), dtype="int64")
        expr = f"id in {arr.tolist()}"
        self.collection.delete(expr)
        self.collection.flush()

    def search(self, Q: np.ndarray, topk: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        Q = self._prep_vecs(np.asarray(Q, dtype="float32"))
        # Search params depend on index type; pick defaults + allow overrides
        sp: Dict[str, Any] = {}
        index_desc = self.collection.indexes[0].params.get("index_type", "HNSW").upper()

        if index_desc == "HNSW":
            sp["ef"] = int(kwargs.get("ef_search", self.params.get("ef_search", 64)))
        else:  # IVF family
            sp["nprobe"] = int(kwargs.get("nprobe", self.params.get("nprobe", 16)))

        # Normalize vectors if needed (handled in _prep_vecs)
        res = self.collection.search(
            data=Q.tolist(),
            anns_field="embedding",
            param=sp,
            limit=int(topk),
            output_fields=["id", "label"],
            metric_type=self.metric,
        )
        # Convert to (D, I) where higher-is-better similarity
        nq = len(res)
        D = np.zeros((nq, topk), dtype="float32")
        I = -np.ones((nq, topk), dtype="int64")
        for i, hits in enumerate(res):
            for j, h in enumerate(hits):
                # Milvus returns distance; for L2: smaller is better; for IP/COSINE: larger is better internally,
                # but pymilvus returns "distance" as a number—interpret accordingly:
                dist = float(h.distance)
                if self.metric == "L2":
                    score = -dist  # convert to higher-is-better
                else:
                    # For IP/COSINE, pymilvus returns similarity in newer versions, but keep it robust:
                    score = dist
                D[i, j] = score
                I[i, j] = int(h.id)
        return D, I

    def stats(self) -> Dict[str, Any]:
        cnt = self.collection.num_entities if self.collection is not None else 0
        idx = None
        try:
            if self.collection and self.collection.indexes:
                idx = self.collection.indexes[0].params
        except Exception:
            pass
        return {
            "collection": self.collection_name,
            "metric": self.metric,
            "count": int(cnt),
            "index": idx,
        }

    def drop(self):
        try:
            if self.collection is not None:
                name = self.collection.name
                self.collection.release()
                utility.drop_collection(name)
        except Exception:
            pass
        self.collection = None