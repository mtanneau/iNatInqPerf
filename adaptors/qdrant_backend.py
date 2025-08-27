import numpy as np
from typing import Dict, Any, Tuple, Sequence
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, HnswConfig, PointStruct, Filter, FieldCondition, Match

from .base import VectorBackend

def _qdrant_distance(metric: str):
    return Distance.COSINE if metric.lower() in ("ip","cosine") else Distance.EUCLID

class QdrantHNSW(VectorBackend):
    name = "qdrant.hnsw"
    def __init__(self):
        self.client = None
        self.collection = "bench"
        self.dim = None
        self.metric = "cosine"
        self.params = {}

    def init(self, dim: int, metric: str, **params):
        self.dim = dim
        self.metric = metric
        self.params = params
        host = params.get("host","localhost")
        port = int(params.get("port",6333))
        self.collection = params.get("collection","bench")
        self.client = QdrantClient(host=host, port=port, prefer_grpc=False)

        # create/overwrite
        self.drop()
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=_qdrant_distance(metric)),
            hnsw_config=HnswConfig(
                m=int(params.get("m", 16)),
                ef_construct=int(params.get("ef_construct", 100))
            )
        )

    def train(self, X_train: np.ndarray):
        # HNSW doesn't require training
        return

    def upsert(self, ids: np.ndarray, X: np.ndarray):
        points = [PointStruct(id=int(i), vector=v.tolist(), payload={}) for i, v in zip(ids, X)]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete(self, ids: Sequence[int]):
        self.client.delete(collection_name=self.collection, points_selector=list(map(int, ids)), wait=True)

    def search(self, Q: np.ndarray, topk: int, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        ef = int(kwargs.get("ef_search", self.params.get("ef_search", 64)))
        # qdrant searches one query at a time
        D = np.zeros((Q.shape[0], topk), dtype="float32")
        I = -np.ones((Q.shape[0], topk), dtype="int64")
        for qi, qv in enumerate(Q):
            res = self.client.search(
                collection_name=self.collection,
                query_vector=qv.tolist(),
                limit=topk,
                search_params={"hnsw_ef": ef}
            )
            for j, pt in enumerate(res):
                I[qi, j] = int(pt.id)
                # qdrant returns higher=better when cosine; approximate to similarity score
                D[qi, j] = float(pt.score)
        return D, I

    def stats(self) -> Dict[str,Any]:
        info = self.client.get_collection(self.collection)
        return {"collection": self.collection, "status": info.status, "vectors_count": info.vectors_count}

    def drop(self):
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass

BACKENDS = {
  "qdrant.hnsw": QdrantHNSW,
}