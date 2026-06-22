from __future__ import annotations

import redis

from app.core.config import get_settings


class RedisCache:
    def __init__(self) -> None:
        self.client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True)

    def get(self, key: str) -> str | None:
        try:
            return self.client.get(key)
        except redis.RedisError:
            return None

    def set(self, key: str, value: str, ttl_seconds: int = 30) -> None:
        try:
            self.client.set(key, value, ex=ttl_seconds)
        except redis.RedisError:
            return
