"""Decayed feature counters for ranking signals (Implementation Guide, Section 7.2).

Provides :class:`DecayedCounter` — a Redis-backed counter with exponential
time decay, falling back to an in-memory implementation when Redis is
unavailable. Counters track signals such as query→product co-click counts,
position-based CTR, and impression counts used by the ranker.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

try:  # pragma: no cover - optional dependency
    import redis as redis_pkg
except ImportError:  # pragma: no cover - optional dependency
    redis_pkg = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class _InMemoryCounter:
    """In-memory fallback counter with exponential decay."""

    def __init__(self, decay_half_life_s: float) -> None:
        self.decay_half_life_s = decay_half_life_s
        self._values: dict[str, float] = defaultdict(float)
        self._ts: dict[str, float] = {}

    def _decay(self, key: str) -> None:
        now = time.time()
        last = self._ts.get(key, now)
        if now > last:
            elapsed = now - last
            factor = 0.5 ** (elapsed / self.decay_half_life_s)
            self._values[key] *= factor
            self._ts[key] = now

    def increment(self, key: str, amount: float = 1.0) -> None:
        self._decay(key)
        self._values[key] += amount
        self._ts[key] = time.time()

    def get(self, key: str) -> float:
        self._decay(key)
        return self._values.get(key, 0.0)

    def set(self, key: str, value: float) -> None:
        self._values[key] = value
        self._ts[key] = time.time()

    def keys(self) -> list[str]:
        return list(self._values.keys())


class DecayedCounter:
    """A counter with exponential time decay.

    Redis is used when available; otherwise an in-memory fallback is used.
    Values decay by half every ``decay_half_life_s`` seconds, so recent
    signals dominate. Safe to use from multiple threads (single process).
    """

    def __init__(
        self,
        decay_half_life_s: float = 3600.0,
        redis_url: str | None = None,
        key_prefix: str = "fs:",
    ) -> None:
        self.decay_half_life_s = decay_half_life_s
        self.key_prefix = key_prefix
        self._redis: Any = None
        self._memory = _InMemoryCounter(decay_half_life_s)
        if redis_url:
            self._try_connect_redis(redis_url)

    def _try_connect_redis(self, redis_url: str) -> None:
        if redis_pkg is None:  # pragma: no cover - redis not installed
            logger.warning("Redis package not installed; using in-memory fallback")
            self._redis = None
            return
        try:  # pragma: no cover - Redis path
            self._redis = redis_pkg.from_url(redis_url, decode_responses=True)
            self._redis.ping()
            logger.info("DecayedCounter connected to Redis at %s", redis_url)
        except Exception:  # pragma: no cover - Redis unavailable
            logger.warning("Redis unavailable (%s); using in-memory fallback", redis_url)
            self._redis = None

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def increment(self, key: str, amount: float = 1.0) -> None:
        """Increment counter for ``key`` by ``amount`` and apply decay."""
        if self._redis is not None:  # pragma: no cover - Redis path
            try:
                self._redis.hincrbyfloat("counters", self._key(key), amount)
            except Exception:  # pragma: no cover - best effort
                logger.exception("Redis increment failed; falling back to memory")
                self._memory.increment(key, amount)
        else:
            self._memory.increment(key, amount)

    def get(self, key: str) -> float:
        """Return the decayed value for ``key``."""
        if self._redis is not None:  # pragma: no cover - Redis path
            try:
                raw = self._redis.hget("counters", self._key(key))
                if raw is None:
                    return 0.0
                return float(raw)
            except Exception:  # pragma: no cover - best effort
                logger.exception("Redis get failed; falling back to memory")
                return self._memory.get(key)
        return self._memory.get(key)

    def set(self, key: str, value: float) -> None:
        """Set counter for ``key`` to ``value`` (resets decay clock)."""
        if self._redis is not None:  # pragma: no cover - Redis path
            try:
                self._redis.hset("counters", self._key(key), value)
            except Exception:  # pragma: no cover - best effort
                logger.exception("Redis set failed; falling back to memory")
                self._memory.set(key, value)
        else:
            self._memory.set(key, value)

    def keys(self) -> list[str]:
        """Return all known counter keys."""
        if self._redis is not None:  # pragma: no cover - Redis path
            try:
                return list(self._redis.hkeys("counters"))
            except Exception:  # pragma: no cover - best effort
                logger.exception("Redis keys failed; falling back to memory")
                return self._memory.keys()
        return self._memory.keys()


class FeatureStore:
    """Named counters for ranking features.

    Groups related counters under named namespaces (e.g. "query:clicks",
    "query:impressions", "product:impressions") and exposes convenience
    methods for common ranking signals.
    """

    def __init__(self, decay_half_life_s: float = 3600.0, redis_url: str | None = None):
        self.counters = DecayedCounter(
            decay_half_life_s=decay_half_life_s,
            redis_url=redis_url,
        )

    def query_clicks(self, query: str, product_id: str) -> float:
        return self.counters.get(f"qc:{query}:{product_id}")

    def increment_query_clicks(self, query: str, product_id: str, amount: float = 1.0) -> None:
        self.counters.increment(f"qc:{query}:{product_id}", amount)

    def query_impressions(self, query: str, product_id: str) -> float:
        return self.counters.get(f"qi:{query}:{product_id}")

    def increment_query_impressions(self, query: str, product_id: str, amount: float = 1.0) -> None:
        self.counters.increment(f"qi:{query}:{product_id}", amount)

    def product_impressions(self, product_id: str) -> float:
        return self.counters.get(f"pi:{product_id}")

    def increment_product_impressions(self, product_id: str, amount: float = 1.0) -> None:
        self.counters.increment(f"pi:{product_id}", amount)

    def product_clicks(self, product_id: str) -> float:
        return self.counters.get(f"pc:{product_id}")

    def increment_product_clicks(self, product_id: str, amount: float = 1.0) -> None:
        self.counters.increment(f"pc:{product_id}", amount)

    def ctr(self, query: str, product_id: str) -> float:
        """Click-through rate for query→product (clicks / impressions)."""
        clicks = self.query_clicks(query, product_id)
        impressions = self.query_impressions(query, product_id)
        if impressions <= 0:
            return 0.0
        return clicks / impressions

    def global_ctr(self, product_id: str) -> float:
        """Global click-through rate for a product."""
        clicks = self.product_clicks(product_id)
        impressions = self.product_impressions(product_id)
        if impressions <= 0:
            return 0.0
        return clicks / impressions


__all__ = [
    "DecayedCounter",
    "FeatureStore",
]
