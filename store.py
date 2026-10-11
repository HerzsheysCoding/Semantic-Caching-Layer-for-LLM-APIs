import hashlib
import threading
import time
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
from schemas import CachedEntry, CacheMetadata


class DualTierCacheStore:
    def __init__(self, embedding_dim: int, max_entries: int = 100_000):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.lock = threading.RLock()

        self._l1_table: Dict[str, str] = {}
        self._entries: Dict[str, CachedEntry] = {}

        quantizer = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        self._faiss_index = faiss.IndexIDMap2(quantizer)

        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: Dict[int, str] = {}
        self._id_counter: int = 0

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
        int_id = self._id_to_int.pop(entry_id, None)
        if int_id is not None:
            self._int_to_id.pop(int_id, None)

    def _evict_oldest(self) -> None:
        if not self._entries:
            return
        oldest_id = next(iter(self._entries))
        oldest_entry = self._entries[oldest_id]
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
        top_k: int = 5,
    ) -> Tuple[Optional[CachedEntry], float]:
        with self.lock:
            if self._faiss_index.ntotal == 0 or len(self._entries) == 0:
                return None, 0.0

            q_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
            search_k = min(top_k, self._faiss_index.ntotal)
            scores, indices = self._faiss_index.search(q_vector, search_k)

            now = time.time()
            top_score = 0.0

            for score, int_id in zip(scores[0], indices[0]):
                if int_id == -1:
                    break

                entry_id = self._int_to_id.get(int(int_id))
                if not entry_id:
                    continue

                entry = self._entries.get(entry_id)
                if not entry:
                    continue

                if self._is_expired(entry, now):
                    exact_key = self.compute_exact_key(entry.query, entry.metadata)
                    self._evict(entry_id, exact_key)
                    continue

                if top_score == 0.0:
                    top_score = float(score)

                if score < threshold:
                    break

                if (
                    entry.metadata.tenant_id != meta.tenant_id
                    or entry.metadata.model != meta.model
                    or entry.metadata.system_prompt_hash != meta.system_prompt_hash
                ):
                    continue

                return entry, float(score)

            return None, top_score

    def insert(self, entry: CachedEntry) -> None:
        with self.lock:
            exact_key = self.compute_exact_key(entry.query, entry.metadata)

            if len(self._entries) >= self.max_entries:
                self._evict_oldest()

            int_id = self._id_counter
            self._id_counter += 1

            self._l1_table[exact_key] = entry.entry_id
            self._entries[entry.entry_id] = entry
            self._id_to_int[entry.entry_id] = int_id
            self._int_to_id[int_id] = entry.entry_id

            vector = np.ascontiguousarray(np.array([entry.embedding], dtype=np.float32))
            id_array = np.array([int_id], dtype=np.int64)
            self._faiss_index.add_with_ids(vector, id_array)