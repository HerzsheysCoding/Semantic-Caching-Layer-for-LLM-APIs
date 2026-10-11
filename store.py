import hashlib
import threading
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from schemas import CachedEntry, CacheMetadata


class DualTierCacheStore:
    def __init__(self, embedding_dim: int, max_entries: int = 100_000):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.lock = threading.RLock()

        self._l1_table: Dict[str, str] = {}
        self._entries: Dict[str, CachedEntry] = {}
        self._index_ids: List[str] = []
        self._embeddings_matrix: Optional[np.ndarray] = None

    @staticmethod
    def compute_exact_key(query: str, meta: CacheMetadata) -> str:
        payload = f"{meta.tenant_id}|{meta.model}|{meta.system_prompt_hash}|{meta.temperature}|{query.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_expired(self, entry: CachedEntry, current_time: Optional[float] = None) -> bool:
        t = current_time or time.time()
        expiry = entry.metadata.created_at.timestamp() + entry.metadata.ttl_seconds
        return t > expiry

    def _evict(self, entry_id: str, exact_key: str) -> None:
        self._entries.pop(entry_id, None)
        self._l1_table.pop(exact_key, None)
        if entry_id in self._index_ids:
            idx = self._index_ids.index(entry_id)
            self._index_ids.pop(idx)
            self._embeddings_matrix = np.delete(self._embeddings_matrix, idx, axis=0)
            if len(self._index_ids) == 0:
                self._embeddings_matrix = None

    def _evict_oldest(self) -> None:
        if not self._index_ids:
            return
        oldest_id = self._index_ids[0]
        oldest_entry = self._entries.get(oldest_id)
        if oldest_entry:
            key = self.compute_exact_key(oldest_entry.query, oldest_entry.metadata)
            self._evict(oldest_id, key)

    def get_l1_exact(self, exact_key: str) -> Optional[CachedEntry]:
        with self.lock:
            entry_id = self._l1_table.get(exact_key)
            if not entry_id:
                return None

            entry = self._entries.get(entry_id)
            if entry and self._is_expired(entry):
                self._evict(entry_id, exact_key)
                return None
            return entry

    def search_l2_semantic(
        self,
        query_vector: np.ndarray,
        meta: CacheMetadata,
        threshold: float,
    ) -> Tuple[Optional[CachedEntry], float]:
        with self.lock:
            if self._embeddings_matrix is None or len(self._index_ids) == 0:
                return None, 0.0

            scores = np.dot(self._embeddings_matrix, query_vector)
            sorted_indices = np.argsort(scores)[::-1]
            top_score = float(scores[sorted_indices[0]]) if len(sorted_indices) > 0 else 0.0

            now = time.time()
            for idx in sorted_indices:
                score = float(scores[idx])
                if score < threshold:
                    break

                entry_id = self._index_ids[idx]
                entry = self._entries[entry_id]

                if (
                    entry.metadata.tenant_id != meta.tenant_id
                    or entry.metadata.model != meta.model
                    or entry.metadata.system_prompt_hash != meta.system_prompt_hash
                ):
                    continue

                if self._is_expired(entry, now):
                    continue

                return entry, score

            return None, top_score

    def insert(self, entry: CachedEntry) -> None:
        with self.lock:
            exact_key = self.compute_exact_key(entry.query, entry.metadata)

            if len(self._entries) >= self.max_entries:
                self._evict_oldest()

            self._l1_table[exact_key] = entry.entry_id
            self._entries[entry.entry_id] = entry

            vector = np.array(entry.embedding, dtype=np.float32).reshape(1, -1)
            if self._embeddings_matrix is None or len(self._index_ids) == 0:
                self._embeddings_matrix = vector
                self._index_ids = [entry.entry_id]
            else:
                self._embeddings_matrix = np.vstack([self._embeddings_matrix, vector])
                self._index_ids.append(entry.entry_id)