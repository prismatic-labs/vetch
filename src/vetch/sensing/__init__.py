"""Grid sensing and caching.

This package provides:
- grid.py: Electricity Maps API client
- cache.py: File-based cache with cross-process locking
- memory_cache.py: In-memory per-process cache
- global_averages.json: Static fallback values

Cache hierarchy:
1. In-memory cache (per-process, ~0.01ms)
2. File cache (cross-process, ~1ms)
3. API fetch (with file locking coordination)
4. Static fallback (regional or global average)
"""

from vetch.sensing.cache import CachedIntensity, FileCache, get_file_cache
from vetch.sensing.grid import GridIntensity, get_carbon_intensity
from vetch.sensing.memory_cache import MemoryCache, get_grid_cache

__all__ = [
    "CachedIntensity",
    "FileCache",
    "get_file_cache",
    "GridIntensity",
    "get_carbon_intensity",
    "MemoryCache",
    "get_grid_cache",
]
