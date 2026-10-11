from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class CacheConfig(BaseModel):
    similarity_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    ttl_seconds: int = Field(default=86400)
    embedding_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    max_entries: int = 100_000


class CacheMetadata(BaseModel):
    tenant_id: str = "default"
    model: str = "gpt-4o"
    system_prompt_hash: str
    temperature: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
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
