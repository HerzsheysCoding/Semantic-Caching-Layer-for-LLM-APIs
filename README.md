# Enterprise Semantic Caching Gateway for LLM APIs

A dual-tier semantic caching reverse proxy for Large Language Model (LLM) APIs. Built with **FastAPI**, **FAISS (HNSW)**, and **Sentence-Transformers**, this gateway cuts query latency by up to **99.9%** and reduces token spend via deterministic exact-match shortcutting and dense vector similarity search.

Compatible with the OpenAI Chat Completions API specification (`POST /v1/chat/completions`) with full support for Server-Sent Events (SSE) streaming.

---

## Architecture Overview

                           Incoming LLM Request
                                    │
                                    ▼
                  ┌───────────────────────────────────┐
                  │  L1 Deterministic Hash (SHA-256)  │  < 0.1ms
                  └─────────────────┬─────────────────┘
                                    │
                    HIT ────────────┴──────────── MISS
                     │                             │
                     ▼                             ▼
             Immediate Response      ┌───────────────────────────┐
                                     │ Dense Vector Embedding    │  ~10-15ms
                                     │ (all-MiniLM-L6-v2)        │
                                     └─────────────┬─────────────┘
                                                   │
                                                   ▼
                                     ┌───────────────────────────┐
                                     │ L2 Vector Search (FAISS)  │  ~2-5ms
                                     │ (HNSW Cosine Sim ≥ 0.82)  │
                                     └─────────────┬─────────────┘
                                                   │
                                   HIT ────────────┴──────────── MISS
                                    │                             │
                                    ▼                             ▼
                            Cached Response             Upstream Inference (1-2s)
                                                                  │
                                                                  ▼
                                                        Async Cache Population

### Key Engineering Features

* **Dual-Tier Matching Pipeline:**
  * **Tier 1 (Exact Match):** Computes a deterministic SHA-256 digest of prompt tokens, model parameters, tenant metadata, and system prompts. Returns hits in **< 0.1ms**.
  * **Tier 2 (Semantic Match):** Generates dense unit-sphere normalized vector embeddings using `all-MiniLM-L6-v2` and performs Approximate Nearest Neighbor (ANN) search via **FAISS HNSW** in **< 20ms**.
* **Zero Cross-Tenant Leakage:** Partition keys strictly isolate requests by `tenant_id`, `model_name`, `temperature`, and `system_prompt_hash`.
* **Streaming Reverse Proxy:** Fully supports Server-Sent Events (`stream=True`) for streaming responses back to clients while asynchronously persisting completions to cache in background tasks.
* **Tombstone Invalidation:** Graph-safe TTL-based expiration and LRU cache eviction using soft-delete tombstoning to preserve HNSW index integrity.
* **Economic Telemetry:** Tracks estimated dollar savings ($USD) and token savings per query.

---

## Benchmark Results

Empirical performance recorded running against mock upstream LLM latency (~1,250ms):

| Request Type | Query Example | Tier Hit | Similarity Score | Latency | Latency Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cold Request** | `"How do I configure SSL termination on NGINX?"` | *None (Miss)* | `0.0000` | 1,250.51 ms | Baseline |
| **Exact Duplicate** | `"How do I configure SSL termination on NGINX?"` | **L1 Exact** | `1.0000` | **0.08 ms** | **99.99%** |
| **Paraphrased Query** | `"Can you explain setting up SSL termination in Nginx?"` | **L2 Semantic** | `0.8647` | **14.67 ms** | **98.83%** |
| **Semantic Divergence** | `"What is the average speed of an unladen swallow?"` | *None (Miss)* | `0.0951` | 1,243.99 ms | Rejected |

---

## Project Structure

```text
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration with health check
├── requirements.txt            # Locked production dependencies
├── schemas.py                  # Pydantic models for cache & OpenAI API schemas
├── embedder.py                 # Dense vector embedding engine
├── store.py                    # Thread-safe dual-tier store backed by FAISS HNSW
├── semantic_cache.py           # Core orchestrator and interceptor logic
├── server.py                   # FastAPI reverse proxy with SSE streaming
├── main.py                     # Standalone verification and benchmark script
└── tests/
    └── test_cache.py           # Pytest integration & unit test suite
Getting Started
Local Setup
Clone the repository:

Bash
git clone [https://github.com/HerzsheysCoding/Semantic-Caching-Layer-for-LLM-APIs.git](https://github.com/HerzsheysCoding/Semantic-Caching-Layer-for-LLM-APIs.git)
cd Semantic-Caching-Layer-for-LLM-APIs
Install dependencies:

Bash
pip install -r requirements.txt
Run the standalone benchmark:

Bash
python main.py
Run the automated test suite:

Bash
python -m pytest tests/ -v
API Usage
Start the Gateway Server
Bash
uvicorn server:app --host 0.0.0.0 --port 8000
Standard Chat Completion (POST /v1/chat/completions)
Bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "tenant_id": "org_production",
    "messages": [
      {"role": "system", "content": "You are a DevOps assistant."},
      {"role": "user", "content": "How do I configure SSL termination on NGINX?"}
    ]
  }'
JSON Response
JSON
{
  "id": "chatcmpl-a1b2c3d4e5f6",
  "object": "chat.completion",
  "created": 1728645000,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Synthesized upstream completion for query: 'How do I configure SSL termination on NGINX?'"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 20,
    "total_tokens": 31
  },
  "cache_status": "HIT-L2_SEMANTIC",
  "cache_latency_ms": 14.67
}
Docker Deployment
Build and start the containerized service:

Bash
docker-compose up --build -d
Verify service health:

Bash
curl http://localhost:8000/health
JSON
{"status": "healthy", "entries_cached": 0}
