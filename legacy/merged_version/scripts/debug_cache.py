#!/usr/bin/env python3
"""
Debug script to test the box counting cache more directly.
"""

import os
import time
from fractal_analyzer import FractalAnalyzer

def test_cache_direct():
    """Test the box counting cache directly."""
    # Enable debug output
    os.environ['DEBUG_CACHE'] = '1'
    
    print("=== TESTING BOX COUNTING CACHE ===")
    print("Creating first analyzer...")
    analyzer1 = FractalAnalyzer('koch')
    
    # Generate a koch curve
    print("\nGenerating koch curve (level 4)...")
    _, segments = analyzer1.generate_fractal('koch', level=4)
    
    # Perform box counting
    print("\nPerforming first box counting...")
    start_time = time.time()
    box_sizes, box_counts, _ = analyzer1.box_counter.box_counting_optimized(
        segments, min_box_size=0.001, max_box_size=0.5, box_size_factor=1.5
    )
    print(f"First box counting time: {time.time() - start_time:.2f} seconds")
    
    # Print cache stats
    print("\nCache stats after first counting:")
    print(analyzer1.box_counter.cache.stats())
    
    # Create a new analyzer
    print("\nCreating second analyzer...")
    analyzer2 = FractalAnalyzer('koch')
    
    # Generate the same fractal
    print("\nGenerating same koch curve (level 4)...")
    _, segments = analyzer2.generate_fractal('koch', level=4)
    
    # Perform box counting again - should be cached
    print("\nPerforming second box counting (should be cached)...")
    start_time = time.time()
    box_sizes2, box_counts2, _ = analyzer2.box_counter.box_counting_optimized(
        segments, min_box_size=0.001, max_box_size=0.5, box_size_factor=1.5
    )
    print(f"Second box counting time: {time.time() - start_time:.2f} seconds")
    
    # Print cache stats again
    print("\nCache stats after second counting:")
    print(analyzer2.box_counter.cache.stats())
    
    # Check if results match
    print("\nResults from both calculations match:", 
          all(box_sizes == box_sizes2) and all(box_counts == box_counts2))

if __name__ == "__main__":
    test_cache_direct()

