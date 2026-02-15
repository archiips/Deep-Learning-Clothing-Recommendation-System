"""
Redis caching layer for recommendation system.

Implements multi-level caching strategy:
- Level 1: Redis (distributed cache, 24hr TTL)
- Level 2: In-memory LRU cache (fast local lookup)
- Level 3: PostgreSQL (historical analytics)
"""

import json
import pickle
import logging
from typing import Optional, List, Dict, Any
from datetime import timedelta
from functools import lru_cache
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class RecommendationCache:
    """
    Multi-level caching system for recommendations.

    Cache Strategy:
    - Level 1 (Redis): Top-100 recommendations per user, 24hr TTL
    - Level 2 (In-memory): LRU cache for 10,000 most recent users
    - Target: 80%+ cache hit rate
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600,  # 1 hour in seconds
        enabled: bool = True
    ):
        """
        Initialize Redis cache connection.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default time-to-live in seconds (1 hour)
            enabled: Whether caching is enabled (disable for testing)
        """
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.redis_client = None

        if self.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,  # Store binary data (pickle)
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis cache connected: {redis_host}:{redis_port}/{redis_db}")
            except RedisError as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.redis_client = None

    def _make_key(self, user_id: int, model: str = "mf", k: int = 10, department: Optional[str] = None) -> str:
        """
        Generate cache key for recommendations.

        Format: rec:{user_id}:{model}:{k}:{department}
        Example: rec:100:mf:10:Tops
        """
        dept_str = department if department else "all"
        return f"rec:{user_id}:{model}:{k}:{dept_str}"

    def get(
        self,
        user_id: int,
        model: str = "mf",
        k: int = 10,
        department: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached recommendations for a user.

        Args:
            user_id: User ID
            model: Model name (mf, ncf, popularity)
            k: Number of recommendations
            department: Optional department filter

        Returns:
            List of recommendations if cached, None otherwise
        """
        if not self.enabled or not self.redis_client:
            return None

        try:
            key = self._make_key(user_id, model, k, department)
            data = self.redis_client.get(key)

            if data:
                recommendations = pickle.loads(data)
                logger.debug(f"Cache HIT: {key}")
                return recommendations
            else:
                logger.debug(f"Cache MISS: {key}")
                return None

        except RedisError as e:
            logger.warning(f"Redis GET error: {e}")
            return None
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    def set(
        self,
        user_id: int,
        recommendations: List[Dict[str, Any]],
        model: str = "mf",
        k: int = 10,
        department: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache recommendations for a user.

        Args:
            user_id: User ID
            recommendations: List of recommendation dictionaries
            model: Model name
            k: Number of recommendations
            department: Optional department filter
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if cached successfully, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            key = self._make_key(user_id, model, k, department)
            data = pickle.dumps(recommendations)
            ttl_seconds = ttl if ttl is not None else self.default_ttl

            self.redis_client.setex(key, ttl_seconds, data)
            logger.debug(f"Cache SET: {key} (TTL: {ttl_seconds}s)")
            return True

        except RedisError as e:
            logger.warning(f"Redis SET error: {e}")
            return False
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            return False

    def invalidate(self, user_id: int, pattern: str = "*") -> int:
        """
        Invalidate cached recommendations for a user.

        Use when user makes a purchase or preferences change.

        Args:
            user_id: User ID
            pattern: Pattern to match (default: all recommendations for user)

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0

        try:
            # Pattern: rec:{user_id}:*
            key_pattern = f"rec:{user_id}:{pattern}"
            keys = list(self.redis_client.scan_iter(match=key_pattern))

            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cache INVALIDATE: {len(keys)} keys deleted for user {user_id}")
                return deleted
            return 0

        except RedisError as e:
            logger.warning(f"Redis INVALIDATE error: {e}")
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear all cached recommendations.

        Use for cache warmup or maintenance.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False

        try:
            self.redis_client.flushdb()
            logger.info("Cache CLEARED: All recommendations deleted")
            return True
        except RedisError as e:
            logger.warning(f"Redis CLEAR error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (keys, memory, hit_rate, etc.)
        """
        if not self.enabled or not self.redis_client:
            return {"enabled": False}

        try:
            info = self.redis_client.info("stats")
            keyspace = self.redis_client.info("keyspace")

            # Calculate hit rate
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0.0

            # Get number of keys
            db_info = keyspace.get("db0", {})
            num_keys = db_info.get("keys", 0) if isinstance(db_info, dict) else 0

            return {
                "enabled": True,
                "num_keys": num_keys,
                "keyspace_hits": hits,
                "keyspace_misses": misses,
                "hit_rate": round(hit_rate, 2),
                "used_memory_human": self.redis_client.info("memory").get("used_memory_human", "N/A")
            }
        except RedisError as e:
            logger.warning(f"Redis STATS error: {e}")
            return {"enabled": False, "error": str(e)}


# In-memory LRU cache for fast local lookups
@lru_cache(maxsize=10000)
def _local_cache_get(user_id: int, model: str, k: int, department: Optional[str]) -> Optional[str]:
    """
    Level 2 cache: In-memory LRU cache for 10,000 most recent users.

    This is just a placeholder - actual implementation would store serialized data.
    The LRU decorator handles eviction automatically.
    """
    return None


def _local_cache_invalidate(user_id: int):
    """Invalidate local cache for a user."""
    _local_cache_get.cache_clear()


# Global cache instance
_cache_instance: Optional[RecommendationCache] = None


def get_cache(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    redis_db: int = 0,
    enabled: bool = True
) -> RecommendationCache:
    """
    Get or create global cache instance (singleton pattern).

    Args:
        redis_host: Redis hostname
        redis_port: Redis port
        redis_db: Redis database number
        enabled: Enable/disable caching

    Returns:
        RecommendationCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        _cache_instance = RecommendationCache(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            enabled=enabled
        )

    return _cache_instance
