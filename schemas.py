import time
from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    similarity_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=86400)
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    max_entries: int = 100_000
    cost_per_1k_tokens: float = 0.01


class CacheMetadata(BaseModel):
    tenant_id: str = "default"
    model: str = "gpt-4o"
    system_prompt_hash: str
    temperature: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int


class CachedEntry(BaseModel):
    entry_id: str
    query: str
    embedding: List[float]
    response: str
    metadata: CacheMetadata
    token_count: int = 0


class CacheLookupResult(BaseModel):
    hit: bool
    tier: Optional[str] = None
    response: Optional[str] = None
    similarity_score: float = 0.0
    latency_ms: float
    cost_saved_usd: float = 0.0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[ChatMessage]
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    tenant_id: str = "default"


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    cache_status: str
    cache_latency_ms: float