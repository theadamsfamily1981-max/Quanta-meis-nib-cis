"""
PGU (Proof Generation Unit) with enhanced caching.
Production policy with LRU cache, hit rate tracking, and safety modes.
Gate: p95 ≤ 200 ms, cache hit-rate ≥ 50%
"""
import re, hashlib, json, time
from collections import OrderedDict
from typing import Optional, Dict
from z3 import Solver, parse_smt2_string, is_const, Z3_OP_TRUE, Z3_OP_FALSE
import warnings


def canonical_key(ast):
    """
    Generate canonical key for Z3 AST with α-equivalence.
    Renames variables to x1, x2, ... for structural matching.
    """
    syms = []
    def walk(a):
        try: kids = a.children()
        except Exception: kids = []
        if is_const(a) and a.decl().kind() not in (Z3_OP_TRUE, Z3_OP_FALSE):
            s = str(a.decl().name())
            if s not in syms: syms.append(s)
        for k in kids: walk(k)
    walk(ast)
    rename = {old: f"x{idx+1}" for idx, old in enumerate(syms)}
    sexpr = ast.sexpr()
    for old, new in rename.items():
        sexpr = re.sub(rf"\b{re.escape(old)}\b", new, sexpr)
    return hashlib.sha1(sexpr.encode()).hexdigest()


class LRUCache:
    """LRU (Least Recently Used) cache with size limit."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: str):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)

    def __len__(self):
        return len(self.cache)


class PGUCache:
    """
    Production PGU cache with substitution-aware caching.

    Modes:
    - 'safety': Hard-block on timeout (for safety-critical applications)
    - 'pretrain': Warn on timeout, continue (for pre-training)
    - 'inference': Fast fallback on timeout
    """

    def __init__(self, timeout_ms: int = 120,
                 fallback_timeout_ms: int = 180,
                 max_cache_size: int = 10000,
                 mode: str = "inference"):
        """
        Args:
            timeout_ms: Primary solver timeout (default: 120ms)
            fallback_timeout_ms: Fallback timeout (default: 180ms)
            max_cache_size: Maximum cache entries (LRU)
            mode: 'safety', 'pretrain', or 'inference'
        """
        self.timeout = timeout_ms
        self.fallback_timeout = fallback_timeout_ms
        self.mode = mode

        # LRU cache with capacity
        self.cache = LRUCache(capacity=max_cache_size)

        # Statistics
        self.num_queries = 0
        self.num_cache_hits = 0
        self.num_cache_misses = 0
        self.num_timeouts = 0
        self.latencies_ms = []

    def check(self, smt2: str, use_fallback: bool = True) -> Dict:
        """
        Check SMT2 formula with caching.

        Args:
            smt2: SMT2 formula string
            use_fallback: Use fallback timeout if primary times out

        Returns:
            dict with keys: cached, ms, res, hit_rate
        """
        self.num_queries += 1

        # Parse and get canonical key
        try:
            ast = parse_smt2_string(smt2)
            key = canonical_key(ast)
        except Exception as e:
            return {
                "cached": False,
                "ms": 0.0,
                "res": "error",
                "error": str(e),
                "hit_rate": self.get_hit_rate()
            }

        # Check cache
        cached_result = self.cache.get(key)
        if cached_result is not None:
            self.num_cache_hits += 1
            return {
                "cached": True,
                "ms": 0.0,
                "res": cached_result,
                "hit_rate": self.get_hit_rate()
            }

        # Cache miss - run solver
        self.num_cache_misses += 1

        # Primary attempt
        s = Solver()
        s.set("timeout", self.timeout)
        t0 = time.perf_counter()

        try:
            res = s.check(ast)
            ms = (time.perf_counter() - t0) * 1000.0
            result_str = str(res)

            # Check for timeout (unknown result)
            if result_str == "unknown" and use_fallback:
                # Try fallback with longer timeout
                s2 = Solver()
                s2.set("timeout", self.fallback_timeout)
                t1 = time.perf_counter()
                res = s2.check(ast)
                ms = (time.perf_counter() - t1) * 1000.0
                result_str = str(res)

                if result_str == "unknown":
                    self.num_timeouts += 1
                    if self.mode == "safety":
                        raise TimeoutError(f"Z3 solver timed out after {self.fallback_timeout}ms")
                    elif self.mode == "pretrain":
                        warnings.warn(f"Z3 solver timed out after {self.fallback_timeout}ms")

        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            result_str = "error"
            if self.mode == "safety":
                raise

        # Cache the result
        self.cache.put(key, result_str)
        self.latencies_ms.append(ms)

        return {
            "cached": False,
            "ms": ms,
            "res": result_str,
            "hit_rate": self.get_hit_rate()
        }

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        if self.num_queries == 0:
            return 0.0
        return self.num_cache_hits / self.num_queries

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency in ms."""
        if not self.latencies_ms:
            return 0.0
        import numpy as np
        return np.percentile(self.latencies_ms, 95)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "num_queries": self.num_queries,
            "num_cache_hits": self.num_cache_hits,
            "num_cache_misses": self.num_cache_misses,
            "num_timeouts": self.num_timeouts,
            "hit_rate": self.get_hit_rate(),
            "p95_latency_ms": self.get_p95_latency(),
            "cache_size": len(self.cache),
            "max_cache_size": self.cache.capacity,
        }

    def reset_stats(self):
        """Reset statistics (keeps cache)."""
        self.num_queries = 0
        self.num_cache_hits = 0
        self.num_cache_misses = 0
        self.num_timeouts = 0
        self.latencies_ms = []

    def clear_cache(self):
        """Clear cache and statistics."""
        self.cache = LRUCache(capacity=self.cache.capacity)
        self.reset_stats()
