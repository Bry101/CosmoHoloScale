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


