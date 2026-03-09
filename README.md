# CosmoHoloScale 🌌

**A holographic-cosmological vector memory engine**  
Proof-of-concept adaptive vector store that grows capacity like an expanding universe and compresses historical data onto reconstructible “event horizons” via truncated SVD.

- Entropy-triggered expansion (differential-entropy heuristic)  
- Golden-ratio scaling with mild acceleration  
- Deliberate lossy compression of older vectors into low-rank holographic summaries  
- Approximate nearest-neighbor queries (KDTree + cosine or Euclidean)

**Current status:** v0.1 — working research prototype, **not production-scale yet**.


## Quick start

```bash
pip install numpy scipy
```

```python
import numpy as np
from cosmoholoscale import CosmoHoloScale

mem = CosmoHoloScale(dim=384, use_cosine=True)   # e.g. for sentence embeddings

# Add vectors (single or batch)
mem.add_vector(np.random.randn(384))
mem.add_batch([np.random.randn(384) for _ in range(100)])

# Query
results = mem.query(np.random.randn(384), top_k=5)
for vec, dist in results:
    print(f"distance: {dist:.4f}")

print(mem.get_status())

```

Run `python cosmoholoscale.py` for a full random-vector demo that shows expansions in action.

## How it works (mental model)

1. **You add vectors** — they go into live memory.  
2. **Every addition**, we compute a lightweight differential-entropy score (the “information density” of the current cloud).  
3. **When density crosses the cosmic threshold**, the engine triggers **dark-energy mode**:
   - Capacity grows by the golden ratio (with gentle acceleration)  
   - The oldest ~28 % of vectors are SVD-compressed into a single summary vector + stored low-rank “horizon” (projections + components)  
4. **Queries** search both live vectors **and** the last few holographic horizons. Old data is approximate but still reachable.

Result: the memory can grow forever without ever saying “out of memory.”


## Design notes / Caveats

### Cosine / Euclidean semantics
- When `use_cosine=True` (default), the KDTree is built on **unit-normalized** vectors.  
  Returned distances = **1 − cosine similarity** (cosine distance on the unit sphere).  
- When `use_cosine=False`, plain Euclidean distance on raw vectors is used.

### Lossy holographic compression
Older data is **intentionally compressed** via truncated SVD.  
The tail of the stream is replaced by a mean vector + stored low-rank projections.  
- This is deliberately lossy — exact retrieval of individual old vectors is sacrificed for unbounded capacity.  
- Queries can still access approximate reconstructions (with a small distance bonus).  

**Trade-off:** better long-term retention vs. perfect short-term recall.

### Performance notes
- KDTree rebuilds are currently eager (fine up to ~5–10k vectors).  
- No dedicated ANN index yet (HNSW/IVF/FAISS) — roadmap item.

### Serialization warning
`save()` / `load()` use Python’s `pickle` for convenience.  
**Do not load files from untrusted sources** — pickle can execute arbitrary code.  
(The internal `to_state()` / `from_state()` API makes swapping formats easy later.)


## Roadmap (community-driven ideas, not promises)
- Attach user-defined payloads / IDs → return `(id, vector, distance)`  
- Track reconstruction error (MSE or cosine fidelity) after each compression  
- Lazy / incremental index rebuilds  
- Optional FAISS / hnswlib backend for million-scale performance  
- Richer horizon strategies (multiple representatives, adaptive rank, etc.)

## Contributing
Contributions, benchmarks, wild ideas, or PRs are very welcome!  
Open an issue or PR — happy to collaborate.

**MIT licensed** — fork, experiment, integrate, and build cool things.

---

Made with curiosity and a love for the universe 🌌  
Feedback / offers / collab → open issues 
