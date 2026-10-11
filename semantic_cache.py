import hashlib
import time
import uuid
from typing import Callable, Optional
from embedder import EmbeddingEngine
from schemas import CacheConfig, CacheLookupResult, CacheMetadata, CachedEntry
from store import DualTierCacheStore


class SemanticCache:
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.embedder = EmbeddingEngine(
            model_name=self.config.embedding_model,
            device=self.config.device,
        )
        self.store = DualTierCacheStore(
            embedding_dim=self.embedder.dimension,
            max_entries=self.config.max_entries,
        )
        self.cost_per_1k_tokens = 0.01

    def _hash_system_prompt(self, system_prompt: Optional[str]) -> str:
        if not system_prompt:
            return "none"
        return hashlib.sha256(system_prompt.strip().encode("utf-8")).hexdigest()[:16]

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> CacheLookupResult:
        start_t = time.perf_counter()

        meta = CacheMetadata(
            tenant_id=tenant_id,
            model=model,
            system_prompt_hash=self._hash_system_prompt(system_prompt),
            temperature=temperature,
            ttl_seconds=self.config.ttl_seconds,
        )

        exact_key = self.store.compute_exact_key(prompt, meta)
        l1_entry = self.store.get_l1_exact(exact_key)
        if l1_entry:
            latency = (time.perf_counter() - start_t) * 1000
            return CacheLookupResult(
                hit=True,
                tier="L1_EXACT",
                response=l1_entry.response,
                similarity_score=1.0,
                latency_ms=round(latency, 2),
                cost_saved_usd=self._calculate_saved_cost(l1_entry.token_count),
            )

        query_vector = self.embedder.embed_query(prompt)
        l2_entry, similarity = self.store.search_l2_semantic(
            query_vector=query_vector,
            meta=meta,
            threshold=self.config.similarity_threshold,
        )

        latency = (time.perf_counter() - start_t) * 1000

        if l2_entry:
            return CacheLookupResult(
                hit=True,
                tier="L2_SEMANTIC",
                response=l2_entry.response,
                similarity_score=round(similarity, 4),
                latency_ms=round(latency, 2),
                cost_saved_usd=self._calculate_saved_cost(l2_entry.token_count),
            )

        return CacheLookupResult(
            hit=False,
            tier=None,
            response=None,
            similarity_score=round(similarity, 4),
            latency_ms=round(latency, 2),
            cost_saved_usd=0.0,
        )

    def write(
        self,
        prompt: str,
        response: str,
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
        model: str = "gpt-4o",
        temperature: float = 0.0,
        token_count: int = 0,
    ) -> None:
        meta = CacheMetadata(
            tenant_id=tenant_id,
            model=model,
            system_prompt_hash=self._hash_system_prompt(system_prompt),
            temperature=temperature,
            ttl_seconds=self.config.ttl_seconds,
        )
        embedding = self.embedder.embed_query(prompt).tolist()
        entry = CachedEntry(
            entry_id=str(uuid.uuid4()),
            query=prompt,
            embedding=embedding,
            response=response,
            metadata=meta,
            token_count=token_count,
        )
        self.store.insert(entry)

    def _calculate_saved_cost(self, tokens: int) -> float:
        if tokens <= 0:
            tokens = 350
        return round((tokens / 1000.0) * self.cost_per_1k_tokens, 5)

    def wrap(
        self,
        llm_callable: Callable[[str], str],
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
        model: str = "gpt-4o",
    ) -> Callable[[str], str]:
        def wrapper(prompt: str) -> str:
            cached = self.query(
                prompt=prompt,
                system_prompt=system_prompt,
                tenant_id=tenant_id,
                model=model,
            )
            if cached.hit:
                return cached.response

            fresh_response = llm_callable(prompt)
            self.write(
                prompt=prompt,
                response=fresh_response,
                system_prompt=system_prompt,
                tenant_id=tenant_id,
                model=model,
            )
            return fresh_response

        return wrapper
