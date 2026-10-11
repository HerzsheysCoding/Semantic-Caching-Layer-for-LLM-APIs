import asyncio
import json
import time
import uuid
from typing import AsyncGenerator
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from schemas import (
    CacheConfig,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from semantic_cache import SemanticCache

app = FastAPI(title="Enterprise Semantic Cache Gateway", version="1.0.0")

config = CacheConfig(similarity_threshold=0.82, ttl_seconds=86400)
cache_engine = SemanticCache(config)


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


@app.get("/health")
def health_check():
    return {"status": "healthy", "entries_cached": len(cache_engine.store._entries)}


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
            usage=UsageInfo(prompt_tokens=len(user_prompt) // 4, completion_tokens=len(lookup.response) // 4, total_tokens=(len(user_prompt) + len(lookup.response)) // 4),
            cache_status=f"HIT-{lookup.tier}",
            cache_latency_ms=lookup.latency_ms,
        )

    fresh_response = await upstream_mock_llm(user_prompt)

    background_tasks.add_task(
        cache_engine.async_write,
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
        usage=UsageInfo(prompt_tokens=len(user_prompt) // 4, completion_tokens=len(fresh_response) // 4, total_tokens=(len(user_prompt) + len(fresh_response)) // 4),
        cache_status="MISS",
        cache_latency_ms=lookup.latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)