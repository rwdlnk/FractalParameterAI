#!/usr/bin/env python3
"""
Test script for quick box visualization.
"""

from fractal_analyzer import FractalAnalyzer
import os

def main():
    """Run quick visualizations for all fractal types."""
    output_dir = "./quick_viz_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a single analyzer first to check cache stats if available
    analyzer = FractalAnalyzer('koch')
    
    # Print initial cache statistics if available
    try:
        if hasattr(analyzer.box_counter, 'cache') and hasattr(analyzer.box_counter.cache, 'stats'):
            box_stats = analyzer.box_counter.cache.stats()
            print("\n===== INITIAL CACHE STATISTICS =====")
            print(f"Box counting cache: {box_stats['hits']} hits, {box_stats['misses']} misses, " 
                  f"{box_stats['hit_rate']*100:.1f}% hit rate")
            
        if hasattr(analyzer, 'generator_cache') and hasattr(analyzer.generator_cache, 'stats'):
            gen_stats = analyzer.generator_cache.stats()
            print(f"Fractal generator cache: {gen_stats['hits']} hits, {gen_stats['misses']} misses, "
                  f"{gen_stats['hit_rate']*100:.1f}% hit rate")
    except Exception as e:
        print(f"Could not get cache statistics: {str(e)}")
    
    # For each fractal
    for fractal_type in ['koch', 'sierpinski', 'dragon', 'hilbert', 'minkowski']:
        try:
            # Set level based on complexity
            level = 5 if fractal_type == 'hilbert' else 6
            
            # Create visualizer and run quick visualization
            analyzer = FractalAnalyzer(fractal_type)
            
            # Make sure the output directory exists
            fractal_dir = os.path.join(output_dir, fractal_type)
            os.makedirs(fractal_dir, exist_ok=True)
            
            # Run the quick visualization
            curve_file, box_file = analyzer.visualizer.create_quick_box_visualization(
                fractal_type, 
                fractal_dir, 
                level=level
            )
            
            print(f"Created visualizations for {fractal_type}:")
            print(f"  Curve: {curve_file}")
            print(f"  Box overlay: {box_file}")
            
        except Exception as e:
            print(f"Error creating visualization for {fractal_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print final cache statistics if available
    try:
        if hasattr(analyzer.box_counter, 'cache') and hasattr(analyzer.box_counter.cache, 'stats'):
            box_stats = analyzer.box_counter.cache.stats()
            print("\n===== FINAL CACHE STATISTICS =====")
            print(f"Box counting cache: {box_stats['hits']} hits, {box_stats['misses']} misses, " 
                  f"{box_stats['hit_rate']*100:.1f}% hit rate")
            
        if hasattr(analyzer, 'generator_cache') and hasattr(analyzer.generator_cache, 'stats'):
            gen_stats = analyzer.generator_cache.stats()
            print(f"Fractal generator cache: {gen_stats['hits']} hits, {gen_stats['misses']} misses, "
                  f"{gen_stats['hit_rate']*100:.1f}% hit rate")
    except Exception as e:
        print(f"Could not get cache statistics: {str(e)}")

if __name__ == "__main__":
    main()
