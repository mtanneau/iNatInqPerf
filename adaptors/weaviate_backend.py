    # adaptors/weaviate_backend.py
from typing import Dict, Any, Tuple, Sequence
import os
import numpy as np

try:
    import weaviate  # client v3.x
except Exception as e:
    raise RuntimeError(
        "weaviate-client is required for the Weaviate backend. "
        "Add 'weaviate-client>=3.25' to pyproject and `uv sync`."
    ) from e

from .base import VectorBackend

def _distance_for(metric: str) -> str:
    """Map our metric names to Weaviate distance names."""
    m = (metric or "ip").lower()
    if m in ("ip", "cosine"):
        return "cosine"
    if m in ("l2", "euclidean"):
        return "l2-squared"
    if m in ("dot", "inner_product"):
        return "dot"
    return "cosine"

def _camel(name: str) -> str:
    return "".join([p.capitalize() for p in name.split("_")])

class WeaviateHNSW(VectorBackend):
    """
    Weaviate backend using user-provided vectors (vectorizer='none'), HNSW index.
    - IDs are stored in property 'ext_id' (int), we use string UUIDs = str(ext_id) for object IDs.
    - Metric mapped via _distance_for; defaults to 'cosine'.
    """
    name = "weaviate.hnsw"

    def __init__(self):
        self.client = None
        self.class_name = "Bench"
        self.dim = None
        self.metric = "cosine"
        self.params: Dict[str, Any] = {}

    def init(self, dim: int, metric: str, **params):
        self.dim = dim
        self.metric = metric
        self.params = params or {}

        # Connection settings
        url = self.params.get("url") or self.params.get("host") or os.environ.get("WEAVIATE_URL") or "http://localhost:8080"
        api_key = self.params.get("api_key") or os.environ.get("WEAVIATE_API_KEY")
        additional_headers = None
        if api_key:
            additional_headers = {"Authorization": f"Bearer {api_key}"}

        # v3 client
        self.client = weaviate.Client(url=url, additional_headers=additional_headers)

        # Schema (vectorizer=none: we push vectors ourselves)
        self.class_name = _camel(self.params.get("collection", "bench"))
        distance = _distance_for(metric)
        m = int(self.params.get("m", 16))
        ef_construct = int(self.params.get("ef_construct", 100))

        # Recreate class
        try:
            if self.client.schema.contains({"class": self.class_name}):
                self.client.schema.delete_class(self.class_name)
        except Exception:
            pass

        class_obj = {
            "class": self.class_name,
            "vectorizer": "none",
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": distance,
                "efConstruction": ef_construct,
                "maxConnections": m,
            },
            "properties": [
                {"name": "ext_id", "dataType": ["int"]},
                {"name": "label", "dataType": ["int"]},
            ],
        }
        self.client.schema.create_class(class_obj)

        # Optional: set efSearch default
        ef_search = int(self.params.get("ef_search", 64))
        try:
            # Weaviate v1.20+ supports per-collection runtime settings via setter
            # Not all versions expose this; ignore if not supported.
            self.client.collections.get(self.class_name)  # type: ignore[attr-defined]
            # If above doesn't raise, you're on a v4 API; but since we use v3 GraphQL below,
            # we'll pass ef at search-time anyway.
        except Exception:
            pass

    def train(self, X_train: np.ndarray):
        """No training required for HNSW."""
        return

    def upsert(self, ids: np.ndarray, X: np.ndarray):
        # Weaviate v3 batch API
        X = X.astype("float32", copy=False)
        with self.client.batch as batch:
            batch.configure(batch_size=512, dynamic=True, timeout_retries=3)
            for ext_id, vec in zip(ids.tolist(), X):
                props = {"ext_id": int(ext_id)}  # label optional; omit if unknown
                # Use the ext_id string as deterministic UUID
                batch.add_data_object(
                    data_object=props,
                    class_name=self.class_name,
                    uuid=str(int(ext_id)),
                    vector=vec.tolist(),
                )

    def delete(self, ids: Sequence[int]):
        # Delete by filter on ext_id (v3)
        where = {
            "path": ["ext_id"],
            "operator": "ContainsAny",
            "valueInt": [int(x) for x in ids],
        }
        try:
            self.client.batch.delete_objects(
                class_name=self.class_name,
                where=where,
                output="minimal",
            )
        except Exception:
            # fallback: per-id delete
            for x in ids:
                try:
                    self.client.data_object.delete(uuid=str(int(x)), class_name=self.class_name)
                except Exception:
                    pass

    def search(self, Q: np.ndarray, topk: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        Q = Q.astype("float32", copy=False)
        ef = int(kwargs.get("ef_search", self.params.get("ef_search", 64)))
        D = np.zeros((Q.shape[0], topk), dtype="float32")
        I = -np.ones((Q.shape[0], topk), dtype="int64")

        for qi in range(Q.shape[0]):
            near = {"vector": Q[qi].tolist(), "ef": ef}
            res = (
                self.client.query
                .get(self.class_name, ["ext_id"])
                .with_near_vector(near)
                .with_additional(["id", "distance"])
                .with_limit(topk)
                .do()
            )
            hits = res.get("data", {}).get("Get", {}).get(self.class_name, []) or []
            for j, h in enumerate(hits[:topk]):
                ext_id = h.get("ext_id", None)
                dist = float(h.get("_additional", {}).get("distance", 0.0))
                # Weaviate returns distance: smaller is better. Convert to "higher is better".
                I[qi, j] = int(ext_id) if ext_id is not None else -1
                D[qi, j] = -dist
        return D, I

    def stats(self) -> Dict[str, Any]:
        # Use Aggregate to get count
        try:
            agg = self.client.query.aggregate(self.class_name).with_meta_count().do()
            cnt = agg.get("data", {}).get("Aggregate", {}).get(self.class_name, [{}])[0].get("meta", {}).get("count", 0)
        except Exception:
            cnt = None
        return {
            "class": self.class_name,
            "metric": self.metric,
            "kind": "hnsw",
            "count": int(cnt) if cnt is not None else None,
        }

    def drop(self):
        try:
            self.client.schema.delete_class(self.class_name)
        except Exception:
            pass

BACKENDS = {
    "weaviate.hnsw": WeaviateHNSW
}