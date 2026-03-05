# fractal_generator_cache.py

from typing import Tuple, List, Dict, Any

class FractalGeneratorCache:
    """Cache for generated fractal segments to avoid redundant generation."""
    
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, fractal_type, level):
        """Generate a unique key for the cache."""
        return (fractal_type, level)
    
    def get(self, fractal_type, level):
        """Get cached fractal if available."""
        key = self.get_key(fractal_type, level)
        result = self.cache.get(key)
        if result is not None:
            self.hits += 1
            print(f"Fractal generation cache hit: {fractal_type} level {level} (hits: {self.hits}, misses: {self.misses})")
        else:
            self.misses += 1
        return result
    
    def store(self, fractal_type, level, points, segments):
        """Store generated fractal in cache."""
        key = self.get_key(fractal_type, level)
        self.cache[key] = (points, segments)
        return (points, segments)
    
    def stats(self):
        """Return cache statistics."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total': self.hits + self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'entries': len(self.cache)
        }
