"""
Production Observability, Metrics Collector & Token-Bucket Rate Limiter.
Provides Prometheus / JSON performance metrics and DoS mitigation for Tiger's Day API.
"""

import time
import asyncio
from collections import deque
from typing import Dict, List, Tuple
import numpy as np

class TokenBucketRateLimiter:
    """
    Token-bucket rate limiter per IP address.
    Default: 60 tokens burst capacity, 1 token/sec refill rate (60 req/min).
    """
    def __init__(self, capacity: int = 60, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets: Dict[str, Tuple[float, float]] = {} # ip -> (tokens, last_update)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_ip: str) -> Tuple[bool, int, float]:
        """
        Returns (is_allowed, remaining_tokens, retry_after_seconds).
        """
        async with self._lock:
            now = time.time()
            if client_ip not in self.buckets:
                self.buckets[client_ip] = (float(self.capacity - 1), now)
                return True, self.capacity - 1, 0.0

            tokens, last_update = self.buckets[client_ip]
            elapsed = now - last_update
            # Refill tokens based on elapsed time
            tokens = min(float(self.capacity), tokens + elapsed * self.refill_rate)

            if tokens >= 1.0:
                tokens -= 1.0
                self.buckets[client_ip] = (tokens, now)
                return True, int(tokens), 0.0
            else:
                retry_after = (1.0 - tokens) / self.refill_rate
                self.buckets[client_ip] = (tokens, now)
                return False, 0, max(0.1, retry_after)


class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.total_errors = 0
        self.mcts_sims_total = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.latencies: deque = deque(maxlen=1000)  # Rolling latency buffer with O(1) eviction
        self.max_latency_samples = 1000

    def record_request(self, latency_ms: float, is_error: bool = False):
        self.total_requests += 1
        if is_error:
            self.total_errors += 1
        self.latencies.append(latency_ms)

    def record_mcts_sims(self, count: int):
        self.mcts_sims_total += count

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def get_summary(self, active_rooms: int = 0) -> dict:
        now = time.time()
        uptime = now - self.start_time

        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_cache) if total_cache > 0 else 0.0

        if self.latencies:
            arr = np.array(self.latencies)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            avg_latency = float(np.mean(arr))
        else:
            p50, p95, p99, avg_latency = 0.0, 0.0, 0.0, 0.0

        return {
            "uptime_seconds": round(uptime, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "mcts_simulations_total": self.mcts_sims_total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(cache_hit_rate * 100, 2),
            "active_rooms": active_rooms,
            "latency_ms": {
                "avg": round(avg_latency, 2),
                "p50": round(p50, 2),
                "p95": round(p95, 2),
                "p99": round(p99, 2)
            }
        }

global_metrics = MetricsCollector()
global_rate_limiter = TokenBucketRateLimiter(capacity=60, refill_rate=1.0)
