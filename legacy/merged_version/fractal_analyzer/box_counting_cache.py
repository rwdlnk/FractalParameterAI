# box_counting_cache.py - Module-level cache fix

import numpy as np
import hashlib
from typing import Tuple, List, Dict, Any
import os

# Module-level cache to ensure it's shared across all instances
_GLOBAL_CACHE = {}
_GLOBAL_HITS = 0
_GLOBAL_MISSES = 0

class BoxCountingCache:
    """Cache for box counting results to avoid redundant calculations."""
    
    def __init__(self):
        # No local instance state - just use the module-level globals
        pass
    
    def get_key(self, segments, min_box_size, max_box_size, box_size_factor):
        """Generate a unique key for the cache based on fractal characteristics."""
        # First check if the segments are all empty
        if not segments:
            return f"empty_segments_{min_box_size}_{max_box_size}_{box_size_factor}"
        
        # Create a simpler representation based on key characteristics
        segment_count = len(segments)
        
        # For very large segment sets, we need a more efficient approach
        if segment_count > 10000:
            # Use a sample of segments and the total count
            sample_size = 50
            sample_step = max(1, segment_count // sample_size)
            
            # Sample points across the entire set
            sample_points = []
            for i in range(0, segment_count, sample_step):
                if i < segment_count:
                    (x1, y1), (x2, y2) = segments[i]
                    # Use fixed precision to avoid floating point issues
                    sample_points.append(f"{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}")
            
            # Get bounds
            min_x = min(min(s[0][0], s[1][0]) for s in segments)
            max_x = max(max(s[0][0], s[1][0]) for s in segments)
            min_y = min(min(s[0][1], s[1][1]) for s in segments)
            max_y = max(max(s[0][1], s[1][1]) for s in segments)
            
            # Create a key with enough unique information but without the full segments
            key_parts = [
                f"count:{segment_count}",
                f"bounds:{min_x:.8f},{min_y:.8f},{max_x:.8f},{max_y:.8f}",
                f"boxparams:{min_box_size}_{max_box_size}_{box_size_factor}",
                f"samples:{';'.join(sample_points)}"
            ]
        else:
            # For smaller segment sets, we can use the entire data
            # But still maintain a consistent format
            segments_str = []
            for (x1, y1), (x2, y2) in segments:
                segments_str.append(f"{x1:.6f},{y1:.6f},{x2:.6f},{y2:.6f}")
            
            key_parts = [
                f"count:{segment_count}",
                f"boxparams:{min_box_size}_{max_box_size}_{box_size_factor}",
                f"segmentdata:{';'.join(segments_str)}"
            ]
        
        # Create a deterministic key by combining all parts
        key_string = "|".join(key_parts)
        
        # Use a consistent hashing approach - SHA-256
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, segments, min_box_size, max_box_size, box_size_factor):
        """Get cached results if available."""
        global _GLOBAL_CACHE, _GLOBAL_HITS, _GLOBAL_MISSES
        
        # Generate cache key
        key = self.get_key(segments, min_box_size, max_box_size, box_size_factor)
        
        # Try to get from cache
        result = _GLOBAL_CACHE.get(key)
        
        if result is not None:
            _GLOBAL_HITS += 1
            print(f"Box counting cache hit (hits: {_GLOBAL_HITS}, misses: {_GLOBAL_MISSES})")
        else:
            _GLOBAL_MISSES += 1
        
        return result
    
    def store(self, segments, min_box_size, max_box_size, box_size_factor, results):
        """Store results in cache."""
        global _GLOBAL_CACHE
        
        # Generate key and store
        key = self.get_key(segments, min_box_size, max_box_size, box_size_factor)
        _GLOBAL_CACHE[key] = results
        
        # Debug output can be enabled by setting environment variable
        if os.environ.get('DEBUG_CACHE', '0') == '1':
            print(f"Stored result with key: {key[:10]}... (len={len(_GLOBAL_CACHE)})")
        
        return results
    
    def stats(self):
        """Return cache statistics."""
        global _GLOBAL_CACHE, _GLOBAL_HITS, _GLOBAL_MISSES
        
        total = _GLOBAL_HITS + _GLOBAL_MISSES
        hit_rate = _GLOBAL_HITS / total if total > 0 else 0.0
        
        return {
            'hits': _GLOBAL_HITS,
            'misses': _GLOBAL_MISSES,
            'total': total,
            'hit_rate': hit_rate,
            'entries': len(_GLOBAL_CACHE)
        }
