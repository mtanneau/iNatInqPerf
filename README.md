# VectorDB-agnostic Benchmark

This project provides a **modular benchmark pipeline** for experimenting with different vector databases (FAISS, Qdrant, …).  
It runs end-to-end:

1. **Download** → Hugging Face dataset (optionally export images + manifest)  
2. **Embed** → Generate CLIP embeddings for images  
3. **Build** → Construct indexes with multiple VectorDB backends  
4. **Search** → Profile queries (latency + Recall@K vs exact baseline)  
5. **Update** → Test insertions & deletions (index maintenance)

All steps are run with **uv** as the package manager.

---

## Quick start (full pipeline)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv venv .venv && source .venv/bin/activate
uv sync

# Run a small end-to-end benchmark (FAISS IVF+PQ backend)
uv run python bench/bench.py run-all --size small --backend faiss.ivfpq
```

---

## Step 1: Download Dataset

Fetch a dataset from Hugging Face, slice by size, and optionally export JPEGs with a manifest.

```bash
# Small slice (200 samples, exports JPEGs)
uv run python bench/bench.py download --size small --out_dir data/raw --export-images

# Large (full train split only)
uv run python bench/bench.py download --size large --out_dir data/raw

# XL (train+val+test)
uv run python bench/bench.py download --size xlarge --out_dir data/raw

# XXL (all data)
uv run python bench/bench.py download --size xxlarge --out_dir data/raw
```

### Options
- `--size` : `small`, `large`, `xlarge`, `xxlarge`
- `--out_dir` : output folder (default: `data/raw`)
- `--export-images` : save JPEGs + `manifest.csv`
- `--no-export-images` : keep HF Arrow dataset only

**Output structure:**
```
data/raw/
  dataset_info.json
  state.json
  data-00000-of-00001.arrow
  images/
    00000000.jpg
    00000001.jpg
    ...
  images/manifest.csv   # [index,filename,label]
```

---

## Step 2: Embed Images

Generate CLIP embeddings and save them into a Hugging Face dataset.

```bash
# Default model + batch size
uv run python bench/bench.py embed --raw_dir data/raw --emb_dir data/emb

# Override CLIP model & batch size
uv run python bench/bench.py embed --raw_dir data/raw --emb_dir data/emb --model_id openai/clip-vit-large-patch14 --batch 32
```

**Outputs:**
- `data/emb/` — temporary embeddings
- `data/emb_hf/` — HF dataset with `{id, label, embedding}`

---

## Step 3: Build Index

Construct an index on a chosen backend.

```bash
# FAISS Flat (exact baseline)
uv run python bench/bench.py build --backend faiss.flat --hf_dir data/emb_hf

# FAISS IVF+PQ (ANN)
uv run python bench/bench.py build --backend faiss.ivfpq --hf_dir data/emb_hf

# Qdrant HNSW (requires running dockerized Qdrant)
./scripts/qdrant_up.sh
uv run python bench/bench.py build --backend qdrant.hnsw --hf_dir data/emb_hf
```

**Supported backends:**
- `faiss.flat` (exact)
- `faiss.ivfpq` (IVF + OPQ + PQ)
- `qdrant.hnsw` (HNSW graph)

---

## Step 4: Search Profiling

Profile query latency and compute Recall@K vs FAISS Flat baseline.

```bash
uv run python bench/bench.py search --backend faiss.ivfpq --hf_dir data/emb_hf --topk 10 --queries bench/queries.txt
```

**Outputs:**
- Latency statistics (avg, p50, p95)
- Recall@K vs baseline
- JSON metrics in `.results/`

---

## Step 5: Index Update

Simulate real-time usage: insert (upsert) and delete vectors.

```bash
uv run python bench/bench.py update --backend faiss.ivfpq --hf_dir data/emb_hf
```

Configurable counts via `configs/bench.yaml`:
```yaml
update:
  add_count: 50
  delete_count: 30
```

---

## Profiling with py-spy

Use `py-spy` to record flamegraphs during any step:

```bash
bash scripts/pyspy_run.sh search-faiss -- python bench/bench.py search --backend faiss.ivfpq --hf_dir data/emb_hf --topk 10 --queries bench/queries.txt
```

Outputs:
- `.results/search-faiss.svg` (flamegraph)
- `.results/search-faiss.speedscope.json`

---

## Adding a New Backend

Each backend adapter implements the `VectorBackend` interface (`adapters/base.py`):

```python
class VectorBackend(ABC):
    def init(self, dim: int, metric: str, **params): ...
    def train(self, X_train: np.ndarray): ...
    def upsert(self, ids: np.ndarray, X: np.ndarray): ...
    def search(self, Q: np.ndarray, topk: int, **kwargs): ...
    def delete(self, ids: Sequence[int]): ...
    def stats(self) -> Dict[str, Any]: ...
    def drop(self): ...
```

### Steps:
1. Write an adapter in `adapters/` (e.g., `weaviate_backend.py`).
2. Register it in `ALL_BACKENDS` in `bench/bench.py`.
3. Add parameters under `backends:` in `configs/bench.yaml`.
4. Run `build`, `search`, and `update` with `--backend <your.backend>`.

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs a **small benchmark** on PRs:
- Download → Embed → Build → Search → Update
- Uploads `.results/*.json` and optional py-spy flamegraphs

Mark it as a **required check** in branch protection to ensure changes pass benchmarks before merge.
