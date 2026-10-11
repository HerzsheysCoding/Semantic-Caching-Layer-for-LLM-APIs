import time
from semantic_cache import SemanticCache
from schemas import CacheConfig


def mock_expensive_llm(prompt: str) -> str:
    time.sleep(1.2)
    return f"Synthesized answer to query: '{prompt}'"


if __name__ == "__main__":
    config = CacheConfig(
        similarity_threshold=0.88,
        ttl_seconds=3600,
        embedding_model="all-MiniLM-L6-v2",
    )
    cache = SemanticCache(config)
    cached_llm = cache.wrap(mock_expensive_llm, system_prompt="You are an expert.")

    queries = [
        "How do I configure SSL termination on NGINX?",
        "How do I configure SSL termination on NGINX?",
        "Can you explain setting up SSL termination in Nginx?",
        "What is the average speed of an unladen European swallow?",
    ]

    for q in queries:
        t0 = time.perf_counter()
        lookup = cache.query(q, system_prompt="You are an expert.")

        if lookup.hit:
            res = lookup.response
            elapsed = (time.perf_counter() - t0) * 1000
            print(
                f"[HIT - {lookup.tier}] '{q}'\n"
                f"  Sim Score: {lookup.similarity_score} | Latency: {elapsed:.2f}ms (Saved: {lookup.cost_saved_usd}$)\n"
            )
        else:
            res = cached_llm(q)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[MISS] '{q}'\n  Executed Upstream | Latency: {elapsed:.2f}ms\n")
