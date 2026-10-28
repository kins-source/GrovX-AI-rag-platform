from hashlib import md5
from typing import Optional
from utils.logger import logger

class QueryCache:
    """
    A simple in-memory cache for user queries.
    Designed to be easily swappable with a Redis implementation for production.
    """
    def __init__(self):
        self._cache = {}

    def _generate_key(self, query: str, context: str = "") -> str:
        cache_input = f"{query.lower().strip()}\0{context}"
        return md5(cache_input.encode("utf-8")).hexdigest()

    def get(self, query: str, context: str = "") -> Optional[str]:
        key = self._generate_key(query, context)
        res = self._cache.get(key)
        if res:
            logger.info(f"Cache HIT for query: '{query}'")
        else:
            logger.info(f"Cache MISS for query: '{query}'")
        return res

    def set(self, query: str, response: str, context: str = ""):
        key = self._generate_key(query, context)
        self._cache[key] = response
        logger.debug(f"Cached response for query: '{query}'")

    def clear(self):
        self._cache.clear()

# Global singleton
query_cache = QueryCache()
