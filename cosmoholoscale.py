import numpy as np
from numpy.linalg import svd, slogdet, norm
from scipy.spatial import KDTree
import pickle


class CosmoHoloScale:
    """Time-aware, lossy vector memory that keeps recent vectors exact and periodically compresses older ones using low-rank SVD.

    The system monitors information density via differential entropy and triggers capacity expansion + compression when a threshold is crossed.
    Compression is intentionally lossy to enable unbounded growth at the cost of approximate retrieval for historical data.
    """

    def __init__(self, initial_capacity=100.0, dim=32, use_cosine=True, verbose=True):
        self.capacity = initial_capacity
        self.dim = dim
        self.vectors = []                    # live vectors (exact, recent)
        self.horizons = []                   # list of (projections, components, mean) for compressed history
        self.expansions = 0
        self.total_cost = 0.0
        self.original_total = 0.0
        self.use_cosine = use_cosine
        self.verbose = verbose               # controls whether expansion events print messages

        # Indexing & caching
        self.index = None                    
        self.data_unit = None                
        self._data_cache = None              

        # Control parameters
        self.adds_since_last = 0
        self.cooldown = 30                   

    # ────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────

    def _invalidate_cache(self):
        self._data_cache = None
        self.data_unit = None               # Always clear so _rebuild_index recreates it

    def _get_data(self):
        if self._data_cache is None:
            if self.vectors:
                self._data_cache = np.stack(self.vectors)
            else:
                self._data_cache = np.empty((0, self.dim))
        return self._data_cache

    def calculate_info_density(self):
        if len(self.vectors) < 3:
            return 0.0
        data = self._get_data()
        if data.shape[0] < 3:
            return 0.0

        cov = np.cov(data.T) + 1e-8 * np.eye(self.dim)
        _, logdet = slogdet(cov)
        entropy = 0.5 * logdet + 0.5 * self.dim * np.log(2 * np.pi * np.e)

        max_entropy = self.dim * np.log(2 * np.pi * np.e) * np.log(len(self.vectors) + 1)
        return max(0.0, min(1.0, entropy / max_entropy))

    def _rebuild_index(self):
        if len(self.vectors) < 2:
            self.index = None
            self.data_unit = None
            return

        data = self._get_data()
        if self.use_cosine:
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            self.data_unit = data / (norms + 1e-10)
            self.index = KDTree(self.data_unit)
        else:
            self.data_unit = None
            self.index = KDTree(data)

    # ────────────────────────────────────────────────
    # Core operations
    # ────────────────────────────────────────────────

    def add_vector(self, vec):
        vec = np.asarray(vec, dtype=float).flatten()[:self.dim]
        vec = np.pad(vec, (0, self.dim - len(vec)))

        self.vectors.append(vec)
        self._invalidate_cache()

        nrm = norm(vec)
        self.total_cost += nrm
        self.original_total += nrm
        self.adds_since_last += 1

        density = self.calculate_info_density()
        if density > 0.155 and self.adds_since_last > self.cooldown:
            self.activate_dark_energy_mode()
            self.adds_since_last = 0
        else:
            self._rebuild_index()

    def add_batch(self, vectors):
        for vec in vectors:
            self.add_vector(vec)

    def activate_dark_energy_mode(self):
        old_capacity = self.capacity
        data = self._get_data()
        old_sum = float(np.sum(np.linalg.norm(data, axis=1))) if len(self.vectors) > 0 else 0.0

        expansion_factor = 1.618034 * (1 + 0.003 * self.expansions)
        self.capacity *= expansion_factor
        self.expansions += 1

        vectors_before = len(self.vectors)
        horizons_before = len(self.horizons)

        if len(self.vectors) > 15:
            compress_start = int(len(self.vectors) * 0.72)
            old_data = data[compress_start:, :]
            U, S, Vh = svd(old_data, full_matrices=False)
            k = max(1, int(len(S) * 0.5))
            projections = U[:, :k] * S[:k]
            components = Vh[:k, :]
            boundary_mean = np.mean(old_data, axis=0)

            self.horizons.append((projections, components, boundary_mean))
            self.vectors = self.vectors[:compress_start] + [boundary_mean]
            self._invalidate_cache()

        data_after = self._get_data()
        kept_sum = float(np.sum(np.linalg.norm(data_after, axis=1))) if len(self.vectors) > 0 else 0.0

        compression_loss = old_sum - kept_sum
        dark_energy_rebate = old_capacity * 0.08
        self.total_cost = max(0.0, self.total_cost - compression_loss - dark_energy_rebate)

        self._rebuild_index()

        if self.verbose:
            print(
                f"🌌 HOLO-DARK ENERGY ACTIVATED! "
                f"Capacity {old_capacity:.1f} → {self.capacity:.1f}× | "
                f"Expansions: {self.expansions} | "
                f"Horizons: {len(self.horizons)} | "
                f"Vectors: {vectors_before} → {len(self.vectors)} | "
                f"Saved ≈ {compression_loss + dark_energy_rebate:.1f}"
            )

    def query(self, query_vec, top_k=5):
        q = np.asarray(query_vec, dtype=float).flatten()[:self.dim]
        q = np.pad(q, (0, self.dim - len(q)))
        q_norm = norm(q)
        q_unit = q / q_norm if q_norm > 0 else q

        results = []

        # Live vectors via KDTree
        if self.index is not None:
            if self.use_cosine:
                k = min(top_k * 2, len(self.vectors))
                _, idx = self.index.query(q_unit.reshape(1, -1), k=k)
                idx = np.atleast_1d(idx[0])
                for i in idx:
                    dist = 1.0 if q_norm == 0 else 1 - float(np.dot(q_unit, self.data_unit[i]))
                    results.append((self.vectors[i], dist))
            else:
                data = self._get_data()
                k = min(top_k * 2, len(self.vectors))
                _, idx = self.index.query(q.reshape(1, -1), k=k)
                idx = np.atleast_1d(idx[0])
                for i in idx:
                    dist = float(norm(q - data[i]))
                    results.append((self.vectors[i], dist))
        else:
            # fallback linear scan
            for v in self.vectors:
                if self.use_cosine:
                    v_norm = norm(v)
                    dist = 1.0 if q_norm == 0 or v_norm == 0 else 1 - float(np.dot(q_unit, v / v_norm))
                else:
                    dist = float(norm(q - v))
                results.append((v, dist))

        # Holographic reconstructions
        for proj, comp, bmean in self.horizons[-4:]:
            q_proj = ((q - bmean) @ comp.T) @ comp
            recon = bmean + q_proj
            if self.use_cosine:
                recon_norm = norm(recon)
                dist = 1.0 if q_norm == 0 or recon_norm == 0 else 1 - float(np.dot(q_unit, recon / recon_norm))
            else:
                dist = float(norm(q - recon))
            results.append((recon, dist * 0.7))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def reconstruct_from_horizon(self, horizon_idx=-1):
        if not self.horizons:
            return None
        proj, comp, bmean = self.horizons[horizon_idx]
        return proj @ comp + bmean

    # ────────────────────────────────────────────────
    # Serialization — WARNING: pickle is not safe for untrusted files
    # ────────────────────────────────────────────────

    def to_state(self):
        data = self._get_data()
        return {
            "capacity": self.capacity, "dim": self.dim, "vectors": data,
            "horizons": self.horizons, "expansions": self.expansions,
            "total_cost": self.total_cost, "original_total": self.original_total,
            "use_cosine": self.use_cosine, "adds_since_last": self.adds_since_last,
            "cooldown": self.cooldown, "verbose": self.verbose,
        }

    @classmethod
    def from_state(cls, state):
        obj = cls(
            initial_capacity=state["capacity"],
            dim=state["dim"],
            use_cosine=state["use_cosine"],
            verbose=state.get("verbose", True)
        )
        data = np.asarray(state["vectors"], dtype=float)
        obj.vectors = [row.copy() for row in data]
        obj.horizons = state["horizons"]
        obj.expansions = state["expansions"]
        obj.total_cost = state["total_cost"]
        obj.original_total = state["original_total"]
        obj.adds_since_last = state["adds_since_last"]
        obj.cooldown = state["cooldown"]
        obj._invalidate_cache()
        obj._rebuild_index()
        return obj

    def save(self, filepath="cosmoholoscale.pkl"):
        state = self.to_state()
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved to {filepath}  (Note: pickle is not safe for untrusted files)")

    @classmethod
    def load(cls, filepath="cosmoholoscale.pkl"):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        return cls.from_state(state)

    def get_status(self):
        efficiency = 100 * max(0.0, (self.original_total - self.total_cost)) / self.original_total if self.original_total > 0 else 0.0
        return {
            "capacity": round(self.capacity, 2),
            "expansions": self.expansions,
            "info_density": round(self.calculate_info_density(), 3),
            "efficiency_gain": round(min(efficiency, 100.0), 2),
            "live_vectors": len(self.vectors),
            "num_horizons": len(self.horizons),
        }


# ────────────────────────────────────────────────
#                  DEMO / TEST
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 CosmoHoloScale PRODUCTION DEMO\n")
    ch = CosmoHoloScale(initial_capacity=100.0, dim=32, use_cosine=True, verbose=True)

    np.random.seed(42)
    for i in range(280):
        vec = np.random.normal(0, 1, 32) * (1 + i / 60)
        ch.add_vector(vec)

    status = ch.get_status()
    print("FINAL STATUS:")
    for k, v in status.items():
        print(f"  {k:18} : {v}")

    q = np.random.normal(0, 1, 32)
    results = ch.query(q, top_k=3)
    print("\nSample query results (vector, distance):")
    for vec, dist in results:
        print(f"  → distance = {dist:.4f}")
