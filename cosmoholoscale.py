import numpy as np
from numpy.linalg import svd, slogdet, norm
from scipy.spatial import KDTree
import pickle
from typing import List, Tuple, Dict, Union, Optional


class CosmoHoloScale:
    """Time-aware, lossy vector memory that keeps recent vectors exact and periodically compresses older ones using low-rank SVD.

    The system monitors information density via differential entropy and triggers capacity expansion + compression when a threshold is crossed.
    Compression is intentionally lossy to enable unbounded growth at the cost of approximate retrieval for historical data.
    """

    def __init__(
        self,
        initial_capacity: float = 100.0,
        dim: int = 32,
        use_cosine: bool = True,
        verbose: bool = True,
    ):
        self.capacity: float = initial_capacity
        self.dim: int = dim
        self.vectors: List[np.ndarray] = []  # live vectors (exact, recent)
        self.horizons: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (projections, components, mean) for compressed history
        self.expansions: int = 0
        self.total_cost: float = 0.0          # cumulative 'energy' (norm sum) of all added vectors
        self.original_total: float = 0.0      # tracks total 'energy' before any compression savings
        self.use_cosine: bool = use_cosine
        self.verbose: bool = verbose

        # Indexing & caching
        self.index: Optional[KDTree] = None
        self.data_unit: Optional[np.ndarray] = None
        self._data_cache: Optional[np.ndarray] = None

        # Control parameters
        self.adds_since_last: int = 0
        self.cooldown: int = 30

    # ────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────

    def _invalidate_cache(self) -> None:
        self._data_cache = None
        self.data_unit = None

    def _get_data(self) -> np.ndarray:
        if self._data_cache is None:
            if self.vectors:
                self._data_cache = np.stack(self.vectors)
            else:
                self._data_cache = np.empty((0, self.dim))
        return self._data_cache

    def calculate_info_density(self) -> float:
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

    def _rebuild_index(self) -> None:
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

    def add_vector(self, vec: Union[np.ndarray, List[float]]) -> None:
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

    def add_batch(self, vectors: Union[List[np.ndarray], np.ndarray]) -> None:
        for vec in vectors:
            self.add_vector(vec)

    def activate_dark_energy_mode(self) -> None:
        old_capacity = self.capacity
        data = self._get_data()
        old_sum = float(np.sum(np.linalg.norm(data, axis=1))) if len(self.vectors) > 0 else 0.0

        expansion_factor = 1.618034 * (1 + 0.003 * self.expansions)
        self.capacity *= expansion_factor
        self.expansions += 1

        vectors_before = len(self.vectors)
        horizons_before = len(self.horizons)

        if len(self.vectors) > 15:
            compress_start = int(len(self.vectors) * 0.72)  # keep ~72% as exact, compress the oldest ~28%
            old_data = data[compress_start:, :]
            U, S, Vh = svd(old_data, full_matrices=False)
            k = max(1, int(len(S) * 0.5))  # keep top 50% singular values for a low-rank approximation
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

    def query(
        self, query_vec: Union[np.ndarray, List[float]], top_k: int = 5
    ) -> List[Tuple[np.ndarray, float, Optional[str]]]:
        """Query nearest neighbors, optionally tagging source ('live' or 'horizon').

        Returns list of (vector, distance, source) where source is None for live vectors,
        or 'horizon' for reconstructed ones.
        """
        q = np.asarray(query_vec, dtype=float).flatten()[:self.dim]
        q = np.pad(q, (0, self.dim - len(q)))
        q_norm = norm(q)
        q_unit = q / q_norm if q_norm > 0 else q

        results: List[Tuple[np.ndarray, float, Optional[str]]] = []

        # Live vectors via KDTree
        if self.index is not None:
            if self.use_cosine:
                k = min(top_k * 2, len(self.vectors))
                _, idx = self.index.query(q_unit.reshape(1, -1), k=k)
                idx = np.atleast_1d(idx[0])
                for i in idx:
                    dist = 1.0 if q_norm == 0 else 1 - float(np.dot(q_unit, self.data_unit[i]))
                    results.append((self.vectors[i], dist, None))
            else:
                data = self._get_data()
                k = min(top_k * 2, len(self.vectors))
                _, idx = self.index.query(q.reshape(1, -1), k=k)
                idx = np.atleast_1d(idx[0])
                for i in idx:
                    dist = float(norm(q - data[i]))
                    results.append((self.vectors[i], dist, None))
        else:
            # fallback linear scan
            for v in self.vectors:
                if self.use_cosine:
                    v_norm = norm(v)
                    dist = 1.0 if q_norm == 0 or v_norm == 0 else 1 - float(np.dot(q_unit, v / v_norm))
                else:
                    dist = float(norm(q - v))
                results.append((v, dist, None))

        # Holographic reconstructions
        for proj, comp, bmean in self.horizons[-4:]:
            q_proj = ((q - bmean) @ comp.T) @ comp
            recon = bmean + q_proj
            if self.use_cosine:
                recon_norm = norm(recon)
                dist = 1.0 if q_norm == 0 or recon_norm == 0 else 1 - float(np.dot(q_unit, recon / recon_norm))
            else:
                dist = float(norm(q - recon))
            results.append((recon, dist * 0.7, "horizon"))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def reconstruct_from_horizon(self, horizon_idx: int = -1) -> Optional[np.ndarray]:
        if not self.horizons:
            return None
        proj, comp, bmean = self.horizons[horizon_idx]
        return proj @ comp + bmean

    # ────────────────────────────────────────────────
    # Serialization — WARNING: pickle is not safe for untrusted files
    # ────────────────────────────────────────────────

    def to_state(self) -> Dict:
        data = self._get_data()
        return {
            "capacity": self.capacity,
            "dim": self.dim,
            "vectors": data,
            "horizons": self.horizons,
            "expansions": self.expansions,
            "total_cost": self.total_cost,
            "original_total": self.original_total,
            "use_cosine": self.use_cosine,
            "adds_since_last": self.adds_since_last,
            "cooldown": self.cooldown,
            "verbose": self.verbose,
        }

    @classmethod
    def from_state(cls, state: Dict) -> 'CosmoHoloScale':
        obj = cls(
            initial_capacity=state["capacity"],
            dim=state["dim"],
            use_cosine=state["use_cosine"],
            verbose=state.get("verbose", True),
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

    def save(self, filepath: str = "cosmoholoscale.pkl") -> None:
        state = self.to_state()
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved to {filepath}  (Note: pickle is not safe for untrusted files)")

    @classmethod
    def load(cls, filepath: str = "cosmoholoscale.pkl") -> 'CosmoHoloScale':
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        return cls.from_state(state)

    def get_status(self) -> Dict[str, Union[float, int]]:
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
    print("\nSample query results (vector, distance, source):")
    for vec, dist, source in results:
        print(f"  → distance = {dist:.4f}, source = {source}")
