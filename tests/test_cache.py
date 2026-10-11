import asyncio
import sys
import time
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas import CacheConfig
from semantic_cache import SemanticCache
from server import app


@pytest.fixture
def cache():
    config = CacheConfig(similarity_threshold=0.82, ttl_seconds=2)
    return SemanticCache(config)


def test_l1_exact_hit(cache):
    query = "What is the capital of France?"
    cache.write(query, "Paris", system_prompt="System 1", tenant_id="t1")

    res = cache.query(query, system_prompt="System 1", tenant_id="t1")
    assert res.hit is True
    assert res.tier == "L1_EXACT"
    assert res.response == "Paris"
    assert res.similarity_score == 1.0


def test_l2_semantic_hit(cache):
    cache.write("How do I configure SSL termination on NGINX?", "NGINX SSL guide", system_prompt="Sys", tenant_id="t1")

    res = cache.query("Can you explain setting up SSL termination in Nginx?", system_prompt="Sys", tenant_id="t1")
    assert res.hit is True
    assert res.tier == "L2_SEMANTIC"
    assert res.response == "NGINX SSL guide"
    assert res.similarity_score >= 0.82


def test_semantic_miss_on_unrelated(cache):
    cache.write("How do I configure SSL termination on NGINX?", "NGINX SSL guide", system_prompt="Sys", tenant_id="t1")

    res = cache.query("What is the distance to the Moon?", system_prompt="Sys", tenant_id="t1")
    assert res.hit is False
    assert res.similarity_score < 0.82


def test_cross_tenant_isolation(cache):
    query = "What is our company confidential revenue?"
    cache.write(query, "$10M", tenant_id="tenant_alpha")

    res = cache.query(query, tenant_id="tenant_beta")
    assert res.hit is False


def test_system_prompt_isolation(cache):
    query = "Summarize the project."
    cache.write(query, "Summary A", system_prompt="Role A")

    res = cache.query(query, system_prompt="Role B")
    assert res.hit is False


def test_ttl_expiration(cache):
    query = "Temporary token"
    cache.write(query, "token123")

    immediate = cache.query(query)
    assert immediate.hit is True

    time.sleep(2.1)
    expired = cache.query(query)
    assert expired.hit is False


@pytest.mark.asyncio
async def test_fastapi_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Assistant"},
                {"role": "user", "content": "How do I build a Docker image?"},
            ],
            "temperature": 0.0,
            "tenant_id": "test_org",
        }
        res1 = await ac.post("/v1/chat/completions", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["cache_status"] == "MISS"

        await asyncio.sleep(0.1)

        res2 = await ac.post("/v1/chat/completions", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cache_status"] == "HIT-L1_EXACT"