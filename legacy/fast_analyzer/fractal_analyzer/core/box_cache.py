"""
Box counting result caching for FastFractalAnalyzer.

Provides intelligent caching of box counting results to avoid recomputation
during sliding window analysis and grid optimization.
"""

import hashlib
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import time

from .data_types import SegmentArray


@dataclass
class CacheEntry:
    """Single cache entry for box counting results."""
    result: int
    timestamp: float
    access_count: int = 0

    def __post_init__(self):
        self.access_count = 0


class BoxCountCache:
    """
    LRU cache for box counting results with intelligent key generation.

    Caches results based on segment data hash, box size, and grid offset
    to avoid recomputation during analysis phases.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries (1 hour default)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.segment_hashes: Dict[int, str] = {}  # Cache segment array hashes

    def _generate_segments_hash(self, segments: SegmentArray) -> str:
        """
        Generate a hash for segment array for cache key generation.

        Uses object ID for efficiency during single session, falls back to
        content hash for cross-session consistency.
        """
        segments_id = id(segments)

        if segments_id in self.segment_hashes:
            return self.segment_hashes[segments_id]

        # Generate content-based hash for the segments
        # Use a sample of segments for efficiency on large datasets
        if segments.n_segments <= 1000:
            # Hash all segments for small datasets
            content = segments.segments.tobytes()
        else:
            # Hash a representative sample for large datasets
            sample_indices = range(0, segments.n_segments, segments.n_segments // 500)
            sample_segments = segments.segments[sample_indices]
            content = sample_segments.tobytes()

        content_hash = hashlib.md5(content).hexdigest()[:16]

        # Cache the hash for this session
        self.segment_hashes[segments_id] = content_hash

        return content_hash

    def _generate_cache_key(self, segments: SegmentArray, box_size: float,
                          offset_x: float = 0.0, offset_y: float = 0.0) -> str:
        """Generate cache key for box counting parameters."""
        segments_hash = self._generate_segments_hash(segments)

        # Create parameter signature
        # Use limited precision to allow for reasonable cache hits
        box_size_rounded = f"{box_size:.6f}"
        offset_x_rounded = f"{offset_x:.6f}"
        offset_y_rounded = f"{offset_y:.6f}"

        key = f"{segments_hash}_{box_size_rounded}_{offset_x_rounded}_{offset_y_rounded}"
        return key

    def get(self, segments: SegmentArray, box_size: float,
            offset_x: float = 0.0, offset_y: float = 0.0) -> Optional[int]:
        """
        Get cached box count result if available.

        Args:
            segments: Line segments
            box_size: Box size
            offset_x, offset_y: Grid offsets

        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_cache_key(segments, box_size, offset_x, offset_y)

        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            return None

        # Update access statistics
        entry.access_count += 1

        return entry.result

    def put(self, segments: SegmentArray, box_size: float, result: int,
            offset_x: float = 0.0, offset_y: float = 0.0):
        """
        Store box count result in cache.

        Args:
            segments: Line segments
            box_size: Box size
            result: Box count result
            offset_x, offset_y: Grid offsets
        """
        key = self._generate_cache_key(segments, box_size, offset_x, offset_y)

        # Evict expired entries and maintain size limit
        self._cleanup_cache()

        # Store new entry
        self.cache[key] = CacheEntry(
            result=result,
            timestamp=time.time()
        )

    def _cleanup_cache(self):
        """Clean up expired entries and enforce size limits."""
        current_time = time.time()

        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]

        # Enforce size limit using LRU eviction
        if len(self.cache) >= self.max_size:
            # Sort by last access time (timestamp) and access count
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda item: (item[1].timestamp, item[1].access_count)
            )

            # Remove oldest entries
            entries_to_remove = len(self.cache) - self.max_size + 1
            for i in range(entries_to_remove):
                key_to_remove = sorted_entries[i][0]
                del self.cache[key_to_remove]

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.segment_hashes.clear()

    def get_stats(self) -> dict:
        """Get cache performance statistics."""
        if not self.cache:
            return {
                'size': 0,
                'max_size': self.max_size,
                'utilization': 0.0,
                'total_access_count': 0
            }

        total_access_count = sum(entry.access_count for entry in self.cache.values())
        avg_access_count = total_access_count / len(self.cache)

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'total_access_count': total_access_count,
            'avg_access_count': avg_access_count,
            'segment_hashes_cached': len(self.segment_hashes)
        }