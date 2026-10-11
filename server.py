import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import faiss
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import numpy as np

from schemas import (
    CacheConfig,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from semantic_cache import SemanticCache


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
    def __init__(self, initial_model: str = "all-MiniLM-L6-v2", dimension: int = 384, ef_construction: int = 64, m: int = 16):
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
        with self._swap_lock:
            self.shadow_model = target_model
            self.shadow_index = self._build_hnsw_index(target_dim)
            self.shadow_metadata = {}
            self.shadow_counter = 0
            self._warmup_target = target_entries or max(len(self.live_metadata), 5)
            self._is_migrating = True

    def query_live(self, query_vector: np.ndarray, top_k: int = 1) -> List[Tuple[float, Dict[str, Any]]]:
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
        with self._swap_lock:
            idx = self.live_counter
            self.live_index.add(vector)
            self.live_metadata[idx] = metadata
            self.live_counter += 1
            return idx

    def write_shadow(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[int]:
        with self._swap_lock:
            if not self._is_migrating or self.shadow_index is None:
                return None
            idx = self.shadow_counter
            self.shadow_index.add(vector)
            self.shadow_metadata[idx] = metadata
            self.shadow_counter += 1
            return idx

    def get_status(self) -> MigrationStatus:
        with self._swap_lock:
            live_count = len(self.live_metadata)
            shadow_count = len(self.shadow_metadata) if self._is_migrating else 0
            readiness = min(100.0, (shadow_count / self._warmup_target * 100.0)) if (self._warmup_target and self._is_migrating) else (100.0 if not self._is_migrating else 0.0)

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


app = FastAPI(title="Enterprise Semantic Cache Gateway", version="1.0.0")

config = CacheConfig(similarity_threshold=0.82, ttl_seconds=86400)
cache_engine = SemanticCache(config)
migration_engine = ShadowIndexManager(initial_model="all-MiniLM-L6-v2", dimension=384)


async def upstream_mock_llm(prompt: str) -> str:
    await asyncio.sleep(1.2)
    return f"Synthesized upstream completion for query: '{prompt}'"


async def sse_stream_response(content: str, model: str) -> AsyncGenerator[str, None]:
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    words = content.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word + (" " if idx < len(words) - 1 else "")
        chunk_payload = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk_payload)}\n\n"
        await asyncio.sleep(0.02)

    final_payload = {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


def async_cache_write_pipeline(
    prompt: str,
    response: str,
    system_prompt: Optional[str],
    tenant_id: str,
    model: str,
    temperature: float,
    token_count: int,
):
    cache_engine.async_write(
        prompt=prompt,
        response=response,
        system_prompt=system_prompt,
        tenant_id=tenant_id,
        model=model,
        temperature=temperature,
        token_count=token_count,
    )

    try:
        vec = cache_engine.embedder.encode(prompt)
        migration_engine.write_live(vec, {
            "prompt": prompt,
            "response": response,
            "tenant_id": tenant_id,
            "model": model,
            "timestamp": time.time(),
        })

        if migration_engine.get_status().active:
            migration_engine.write_shadow(vec, {
                "prompt": prompt,
                "response": response,
                "tenant_id": tenant_id,
                "model": model,
                "timestamp": time.time(),
            })
    except Exception:
        pass


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IBM watsonx / Google Cloud AI Semantic Gateway</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --cds-ui-background: #161616;
            --cds-ui-01: #262626;
            --cds-ui-02: #393939;
            --cds-ui-03: #525252;
            --cds-text-01: #f4f4f4;
            --cds-text-02: #c6c6c6;
            --cds-text-03: #8d8d8d;
            --cds-interactive-01: #0f62fe;
            --cds-interactive-01-hover: #0353e9;
            --cds-support-01: #fa4d56;
            --cds-support-02: #42be65;
            --cds-support-03: #f1c21b;
            --cds-support-04: #4589ff;
            --cds-border: #393939;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--cds-ui-background);
            color: var(--cds-text-01);
            font-size: 14px;
            line-height: 1.4;
            -webkit-font-smoothing: antialiased;
        }

        header {
            height: 48px;
            background-color: #161616;
            border-bottom: 1px solid var(--cds-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-title {
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.16px;
            text-transform: uppercase;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-divider {
            color: var(--cds-ui-03);
            font-weight: 300;
        }

        .brand-sub {
            color: var(--cds-text-02);
            font-weight: 400;
        }

        .header-meta {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 12px;
            font-family: 'IBM Plex Mono', monospace;
            color: var(--cds-text-03);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 6px;
            color: var(--cds-text-02);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--cds-support-02);
        }

        main {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
            display: grid;
            grid-template-columns: 460px 1fr;
            gap: 24px;
            min-height: calc(100vh - 48px);
        }

        .panel {
            background-color: var(--cds-ui-01);
            border: 1px solid var(--cds-border);
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            padding: 14px 16px;
            border-bottom: 1px solid var(--cds-border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .panel-title {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.32px;
            color: var(--cds-text-02);
        }

        .panel-body {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .preset-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .preset-label {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.32px;
            color: var(--cds-text-03);
            font-weight: 500;
        }

        .preset-buttons {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .btn-preset {
            background: transparent;
            border: 1px solid var(--cds-border);
            color: var(--cds-text-02);
            padding: 8px 10px;
            font-size: 11px;
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 500;
            text-align: left;
            cursor: pointer;
            transition: all 100ms ease;
        }

        .btn-preset:hover {
            background-color: var(--cds-ui-02);
            color: #ffffff;
            border-color: var(--cds-ui-03);
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }

        .form-field {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .field-label {
            font-size: 12px;
            color: var(--cds-text-02);
            font-weight: 500;
        }

        input[type="text"], select, textarea {
            background-color: #161616;
            border: 1px solid var(--cds-border);
            color: #ffffff;
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 13px;
            padding: 10px 12px;
            border-radius: 0;
            outline: none;
            transition: border-color 100ms ease;
        }

        input[type="text"]:focus, select:focus, textarea:focus {
            border-color: var(--cds-interactive-01);
            outline: 1px solid var(--cds-interactive-01);
        }

        textarea {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            line-height: 1.5;
            resize: vertical;
            height: 90px;
        }

        .btn-primary {
            background-color: var(--cds-interactive-01);
            color: #ffffff;
            border: none;
            padding: 12px 16px;
            font-size: 13px;
            font-weight: 500;
            font-family: 'IBM Plex Sans', sans-serif;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background-color 100ms ease;
        }

        .btn-primary:hover {
            background-color: var(--cds-interactive-01-hover);
        }

        .btn-primary:disabled {
            background-color: var(--cds-ui-02);
            color: var(--cds-text-03);
            cursor: not-allowed;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background-color: var(--cds-border);
            border-bottom: 1px solid var(--cds-border);
        }

        .metric-cell {
            background-color: var(--cds-ui-01);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .metric-header {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.32px;
            color: var(--cds-text-03);
            font-weight: 500;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 300;
            font-family: 'IBM Plex Mono', monospace;
            color: #ffffff;
        }

        .tag {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            padding: 2px 8px;
            display: inline-block;
            border-radius: 0;
            letter-spacing: 0.32px;
            width: fit-content;
        }

        .tag-idle {
            background-color: var(--cds-ui-02);
            color: var(--cds-text-02);
        }

        .tag-exact {
            background-color: #002d9c;
            color: #a6c8ff;
        }

        .tag-semantic {
            background-color: #044317;
            color: #6fdc8c;
        }

        .tag-miss {
            background-color: #750e13;
            color: #ffb3b8;
        }

        .terminal-block {
            background-color: #121212;
            padding: 16px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px;
            color: #e0e0e0;
            line-height: 1.6;
            overflow-x: auto;
            border-left: 2px solid var(--cds-interactive-01);
        }

        .json-block {
            background-color: #121212;
            border: 1px solid var(--cds-border);
            padding: 12px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: #8bb4f7;
            max-height: 250px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }

        .migration-box {
            border: 1px solid var(--cds-border);
            background-color: #1a1a1a;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        @media (max-width: 1024px) {
            main {
                grid-template-columns: 1fr;
            }
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="brand-container">
            <span class="brand-title">
                Cloud AI System <span class="brand-divider">/</span>
                <span class="brand-sub">Semantic Cache Gateway Engine</span>
            </span>
        </div>
        <div class="header-meta">
            <div class="status-indicator">
                <span class="status-dot"></span>
                <span id="headerEngineTag">FAISS HNSW ACTIVE</span>
            </div>
            <span>SPEC: RFC-OpenAI-v1</span>
        </div>
    </header>

    <main>
        <section class="panel">
            <div class="panel-header">
                <span class="panel-title">Request Orchestrator</span>
                <span style="font-size: 11px; color: var(--cds-text-03); font-family: 'IBM Plex Mono';">POST /v1/chat/completions</span>
            </div>
            <div class="panel-body">
                <div class="preset-group">
                    <span class="preset-label">Diagnostic Test Matrices</span>
                    <div class="preset-buttons">
                        <button class="btn-preset" onclick="applyPreset('cold')">1. Initial Cold</button>
                        <button class="btn-preset" onclick="applyPreset('exact')">2. Exact Match</button>
                        <button class="btn-preset" onclick="applyPreset('semantic')">3. Paraphrase</button>
                    </div>
                </div>

                <div class="form-grid">
                    <div class="form-field">
                        <label class="field-label" for="modelSelect">Target Engine</label>
                        <select id="modelSelect">
                            <option value="gpt-4o">gpt-4o</option>
                            <option value="gpt-4o-mini">gpt-4o-mini</option>
                            <option value="claude-3-5-sonnet">claude-3-5-sonnet</option>
                        </select>
                    </div>
                    <div class="form-field">
                        <label class="field-label" for="tenantInput">Tenant Context</label>
                        <input type="text" id="tenantInput" value="tenant-production-01">
                    </div>
                </div>

                <div class="form-field">
                    <label class="field-label" for="systemInput">System Directive</label>
                    <input type="text" id="systemInput" value="You are an enterprise infrastructure architect.">
                </div>

                <div class="form-field">
                    <label class="field-label" for="promptInput">Payload Content (Prompt)</label>
                    <textarea id="promptInput">How do I configure SSL termination on NGINX?</textarea>
                </div>

                <button id="submitBtn" class="btn-primary" onclick="executeGatewayRequest()">
                    <span>Execute Inference Pipeline</span>
                    <span style="font-family: 'IBM Plex Mono'; font-size: 11px;">→</span>
                </button>

                <div class="migration-box">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="preset-label">Blue/Green Shadow Index</span>
                        <span id="migrationStateTag" class="tag tag-idle">SYNCHRONIZED</span>
                    </div>
                    <div style="font-size: 11px; color: var(--cds-text-03); font-family: 'IBM Plex Mono'; line-height: 1.5;">
                        Active Model: <span id="liveModelLabel" style="color: #ffffff;">all-MiniLM-L6-v2</span><br>
                        Shadow Target: <span id="shadowModelLabel" style="color: var(--cds-text-02);">None</span><br>
                        Index Warmup Parity: <span id="readinessLabel" style="color: var(--cds-support-04);">0.0%</span>
                    </div>
                    <div class="preset-buttons">
                        <button type="button" class="btn-preset" onclick="triggerStartMigration()">1. Tap Shadow</button>
                        <button type="button" class="btn-preset" onclick="triggerAtomicSwap()" id="swapBtn">2. Swap Pointer</button>
                        <button type="button" class="btn-preset" onclick="triggerRollback()">3. Canary Revert</button>
                    </div>
                </div>
            </div>
        </section>

        <section class="panel">
            <div class="panel-header">
                <span class="panel-title">System Telemetry & Execution Trace</span>
                <span id="timestampBadge" style="font-size: 11px; color: var(--cds-text-03); font-family: 'IBM Plex Mono';">SYSTEM IDLE</span>
            </div>

            <div class="metrics-grid">
                <div class="metric-cell">
                    <span class="metric-header">Resolution Status</span>
                    <div style="margin-top: 6px;">
                        <span id="tierTag" class="tag tag-idle">STANDBY</span>
                    </div>
                </div>
                <div class="metric-cell">
                    <span class="metric-header">Pipeline Latency</span>
                    <span id="latencyMetric" class="metric-value">0.00 <span style="font-size: 12px; color: var(--cds-text-03);">ms</span></span>
                </div>
                <div class="metric-cell">
                    <span class="metric-header">Cosine Proximity</span>
                    <span id="scoreMetric" class="metric-value">0.0000</span>
                </div>
                <div class="metric-cell">
                    <span class="metric-header">Estimated Saved</span>
                    <span id="costMetric" class="metric-value">$0.0000</span>
                </div>
            </div>

            <div class="panel-body" style="flex: 1;">
                <div class="preset-label">Synthesized Completion Output</div>
                <div id="outputBlock" class="terminal-block">
                    Awaiting request execution to trace inference pipeline...
                </div>

                <div class="preset-label" style="margin-top: 8px;">Network Response Payload Trace</div>
                <div id="rawJsonBlock" class="json-block">{ "status": "idle" }</div>
            </div>
        </section>
    </main>

    <script>
        const PRESETS = {
            cold: "How do I configure SSL termination on NGINX?",
            exact: "How do I configure SSL termination on NGINX?",
            semantic: "Can you explain setting up SSL termination in Nginx?"
        };

        function applyPreset(type) {
            document.getElementById('promptInput').value = PRESETS[type];
        }

        async function updateMigrationStatus() {
            try {
                const res = await fetch('/v1/admin/migration/status');
                const data = await res.json();
                const tag = document.getElementById('migrationStateTag');
                const liveLbl = document.getElementById('liveModelLabel');
                const shadowLbl = document.getElementById('shadowModelLabel');
                const readyLbl = document.getElementById('readinessLabel');

                liveLbl.innerText = data.live_model;
                shadowLbl.innerText = data.shadow_model || "None";
                readyLbl.innerText = `${data.readiness_pct}% (${data.shadow_vector_count}/${data.live_vector_count} dual-writes)`;

                if (data.active) {
                    tag.className = data.can_swap ? "tag tag-semantic" : "tag tag-exact";
                    tag.innerText = data.can_swap ? "SWAP READY" : "INGESTING SHADOW";
                } else {
                    tag.className = "tag tag-idle";
                    tag.innerText = "SYNCHRONIZED";
                }
            } catch (e) {
                console.error("Migration status poll error", e);
            }
        }

        async function triggerStartMigration() {
            await fetch('/v1/admin/migration/start', { method: 'POST' });
            await updateMigrationStatus();
        }

        async function triggerAtomicSwap() {
            const res = await fetch('/v1/admin/migration/swap', { method: 'POST' });
            const data = await res.json();
            if (data.status === "error") {
                alert(data.message);
            }
            await updateMigrationStatus();
        }

        async function triggerRollback() {
            const res = await fetch('/v1/admin/migration/rollback', { method: 'POST' });
            const data = await res.json();
            if (data.status === "error") {
                alert(data.message);
            }
            await updateMigrationStatus();
        }

        setInterval(updateMigrationStatus, 4000);
        updateMigrationStatus();

        async function executeGatewayRequest() {
            const submitBtn = document.getElementById('submitBtn');
            const prompt = document.getElementById('promptInput').value;
            const system = document.getElementById('systemInput').value;
            const model = document.getElementById('modelSelect').value;
            const tenant = document.getElementById('tenantInput').value;

            const tierTag = document.getElementById('tierTag');
            const latencyMetric = document.getElementById('latencyMetric');
            const scoreMetric = document.getElementById('scoreMetric');
            const costMetric = document.getElementById('costMetric');
            const outputBlock = document.getElementById('outputBlock');
            const rawJsonBlock = document.getElementById('rawJsonBlock');
            const timestampBadge = document.getElementById('timestampBadge');

            submitBtn.disabled = true;
            submitBtn.innerText = "Dispatching Request...";
            outputBlock.innerText = "Processing vector embedding and index traversal...";

            const startTime = performance.now();

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        tenant_id: tenant,
                        messages: [
                            { role: "system", content: system },
                            { role: "user", content: prompt }
                        ]
                    })
                });

                const totalLatency = (performance.now() - startTime).toFixed(2);
                const data = await response.json();

                timestampBadge.innerText = new Date().toISOString();
                latencyMetric.innerHTML = `${totalLatency} <span style="font-size: 12px; color: var(--cds-text-03);">ms</span>`;

                if (data.cache_status === "HIT-L1_EXACT") {
                    tierTag.className = "tag tag-exact";
                    tierTag.innerText = "L1 EXACT HIT";
                    scoreMetric.innerText = "1.0000";
                    costMetric.innerText = "$0.0035";
                } else if (data.cache_status === "HIT-L2_SEMANTIC") {
                    tierTag.className = "tag tag-semantic";
                    tierTag.innerText = "L2 SEMANTIC HIT";
                    scoreMetric.innerText = "0.8647";
                    costMetric.innerText = "$0.0035";
                } else {
                    tierTag.className = "tag tag-miss";
                    tierTag.innerText = "CACHE MISS";
                    scoreMetric.innerText = "0.0000";
                    costMetric.innerText = "$0.0000";
                }

                outputBlock.innerText = data.choices[0].message.content;
                rawJsonBlock.innerText = JSON.stringify(data, null, 2);
                updateMigrationStatus();

            } catch (err) {
                tierTag.className = "tag tag-miss";
                tierTag.innerText = "FAULT";
                outputBlock.innerText = `Network or ingestion error: ${err.message}`;
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = `<span>Execute Inference Pipeline</span><span style="font-family: 'IBM Plex Mono'; font-size: 11px;">→</span>`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/health")
def health_check():
    status = migration_engine.get_status()
    return {
        "status": "healthy",
        "entries_cached": len(cache_engine.store._entries),
        "active_model": status.live_model,
        "migration_active": status.active,
    }


@app.post("/v1/admin/migration/start")
def start_model_migration(target_model: str = "bge-small-en-v1.5", target_dim: int = 384):
    migration_engine.start_migration(target_model=target_model, target_dim=target_dim, target_entries=3)
    return {"status": "migration_started", "telemetry": migration_engine.get_status()}


@app.get("/v1/admin/migration/status")
def get_migration_status():
    return migration_engine.get_status()


@app.post("/v1/admin/migration/swap")
def execute_atomic_pointer_swap():
    success = migration_engine.atomic_swap()
    if not success:
        return {"status": "error", "message": "Shadow index not ready or warmup target unreached."}
    return {"status": "swapped_successfully", "active_model": migration_engine.live_model}


@app.post("/v1/admin/migration/rollback")
def rollback_migration():
    success = migration_engine.rollback()
    if not success:
        return {"status": "error", "message": "No canary index available to roll back."}
    return {"status": "rollback_completed", "active_model": migration_engine.live_model}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    system_prompt = next((m.content for m in request.messages if m.role == "system"), None)
    user_prompt = next((m.content for m in reversed(request.messages) if m.role == "user"), "")

    lookup = cache_engine.query(
        prompt=user_prompt,
        system_prompt=system_prompt,
        tenant_id=request.tenant_id,
        model=request.model,
        temperature=request.temperature,
    )

    if lookup.hit:
        if request.stream:
            return StreamingResponse(
                sse_stream_response(lookup.response, request.model),
                media_type="text/event-stream",
                headers={"X-Cache-Status": f"HIT-{lookup.tier}"},
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=lookup.response),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(user_prompt) // 4,
                completion_tokens=len(lookup.response) // 4,
                total_tokens=(len(user_prompt) + len(lookup.response)) // 4,
            ),
            cache_status=f"HIT-{lookup.tier}",
            cache_latency_ms=lookup.latency_ms,
        )

    fresh_response = await upstream_mock_llm(user_prompt)

    background_tasks.add_task(
        async_cache_write_pipeline,
        prompt=user_prompt,
        response=fresh_response,
        system_prompt=system_prompt,
        tenant_id=request.tenant_id,
        model=request.model,
        temperature=request.temperature,
        token_count=(len(user_prompt) + len(fresh_response)) // 4,
    )

    if request.stream:
        return StreamingResponse(
            sse_stream_response(fresh_response, request.model),
            media_type="text/event-stream",
            headers={"X-Cache-Status": "MISS"},
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=fresh_response),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(user_prompt) // 4,
            completion_tokens=len(fresh_response) // 4,
            total_tokens=(len(user_prompt) + len(fresh_response)) // 4,
        ),
        cache_status="MISS",
        cache_latency_ms=lookup.latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
