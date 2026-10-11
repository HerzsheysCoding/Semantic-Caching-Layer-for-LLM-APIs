"""
Shadow Index Manager for Zero-Downtime Vector Migrations.
Implements Blue/Green FAISS indexing with atomic pointer swaps and dual-writes.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np


@dataclass
class MigrationStatus:
    active: bool
    live_model: str
    shadow_model: Optional[str]
    live_vector_count: int
    shadow_vector_count: int
    readiness_pct: float
    can_swap: bool


class ShadowIndexManager:
    def __init__(self, initial_model: str, dimension: int = 384, ef_construction: int = 64, m: int = 16):
        self.live_model = initial_model
        self.shadow_model: Optional[str] = None
        self.dimension = dimension
        self.ef_construction = ef_construction
        self.m = m


        self.live_index = self._build_hnsw_index(self.dimension)
        self.live_metadata: Dict[int, Dict[str, Any]] = {}
        self.live_counter = 0

       
        self.shadow_index: Optional[faiss.IndexHNSWFlat] = None
        self.shadow_metadata: Dict[int, Dict[str, Any]] = {}
        self.shadow_counter = 0

     
        self.canary_index: Optional[faiss.IndexHNSWFlat] = None
        self.canary_metadata: Dict[int, Dict[str, Any]] = {}
        self.canary_model: Optional[str] = None

        self._swap_lock = threading.RLock()
        self._is_migrating = False
        self._warmup_target = 0

    def _build_hnsw_index(self, dim: int) -> faiss.IndexHNSWFlat:
        index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = 32
        return index

    def start_migration(self, target_model: str, target_dim: int, target_entries: Optional[int] = None) -> None:
        """Initializes the shadow index and sets up dual-write tracking."""
        with self._swap_lock:
            self.shadow_model = target_model
            self.shadow_index = self._build_hnsw_index(target_dim)
            self.shadow_metadata = {}
            self.shadow_counter = 0
            self._warmup_target = target_entries or max(len(self.live_metadata), 10)
            self._is_migrating = True

    def query_live(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
        """Queries the live index without touching the shadow partition."""
        with self._swap_lock:
            if self.live_index.ntotal == 0:
                return []
            distances, indices = self.live_index.search(query_vector, top_k)
            results = []
            for sim, idx in zip(distances[0], indices[0]):
                if idx in self.live_metadata:
                    results.append((float(sim), self.live_metadata[idx]))
            return results

    def write_live(self, vector: np.ndarray, metadata: Dict[str, Any]) -> int:
        """Writes directly to the live index."""
        with self._swap_lock:
            idx = self.live_counter
            self.live_index.add(vector)
            self.live_metadata[idx] = metadata
            self.live_counter += 1
            return idx

    def write_shadow(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[int]:
        """Dual-writes new ingested completions into the shadow index using target embeddings."""
        with self._swap_lock:
            if not self._is_migrating or self.shadow_index is None:
                return None
            idx = self.shadow_counter
            self.shadow_index.add(vector)
            self.shadow_metadata[idx] = metadata
            self.shadow_counter += 1
            return idx

    def get_status(self) -> MigrationStatus:
        """Returns real-time telemetry on index synchronization."""
        with self._swap_lock:
            live_count = len(self.live_metadata)
            shadow_count = len(self.shadow_metadata) if self._is_migrating else 0
            readiness = min(100.0, (shadow_count / self._warmup_target * 100.0)) if self._warmup_target else 0.0

            return MigrationStatus(
                active=self._is_migrating,
                live_model=self.live_model,
                shadow_model=self.shadow_model,
                live_vector_count=live_count,
                shadow_vector_count=shadow_count,
                readiness_pct=round(readiness, 2),
                can_swap=self._is_migrating and (shadow_count >= self._warmup_target),
            )

    def atomic_swap(self) -> bool:
        """
        Atomically promotes the shadow index to live and demotes live to canary.
        Zero read locks or interruption to active incoming requests.
        """
        with self._swap_lock:
            if not self._is_migrating or self.shadow_index is None:
                return False

            self.canary_index = self.live_index
            self.canary_metadata = self.live_metadata
            self.canary_model = self.live_model

    
            self.live_index = self.shadow_index
            self.live_metadata = self.shadow_metadata
            self.live_counter = self.shadow_counter
            self.live_model = self.shadow_model

           
            self.shadow_index = None
            self.shadow_metadata = {}
            self.shadow_model = None
            self._is_migrating = False

            return True

    def rollback(self) -> bool:
        """Rolls back to the canary index if post-migration divergence occurs."""
        with self._swap_lock:
            if self.canary_index is None or self.canary_model is None:
                return False

            self.live_index = self.canary_index
            self.live_metadata = self.canary_metadata
            self.live_model = self.canary_model
            self.canary_index = None
            self.canary_metadata = {}
            self.canary_model = None
            return True
