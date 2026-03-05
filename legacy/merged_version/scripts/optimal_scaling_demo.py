#!/usr/bin/env python3
"""
Demo script for optimal scaling region selection in fractal dimension calculation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from fractal_analyzer import FractalAnalyzer

def test_optimal_scaling(fractal_type='koch', level=9, output_dir='./scaling_test'):
    """Test the optimal scaling region selection on standard fractals."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing optimal scaling region selection on {fractal_type} (level {level})...")
    
    # Create fractal analyzer
    analyzer = FractalAnalyzer(fractal_type)
    
    # Generate fractal
    _, segments = analyzer.generate_fractal(fractal_type, level=level)
    print(f"Generated {fractal_type} with {len(segments)} segments")
    
    # Calculate fractal dimension with optimal scaling
    print("\n--- Using Optimal Scaling Region Selection ---")
    fd_opt, error_opt, box_sizes, box_counts, bbox, intercept_opt, r_squared_opt, opt_region, local_slopes = (
        analyzer.calculate_fractal_dimension(
            segments, 
            min_box_size=0.001, 
            box_size_factor=1.5,
            use_optimal_scaling=True,
            window_width=3,
            sigma_thresh=0.15,
            min_region_length=3,
            debug=True
        )
    )
    
    # Calculate fractal dimension with traditional approach for comparison
    print("\n--- Using Traditional Approach (Full Range) ---")
    fd_trad, error_trad, _, _, _, intercept_trad, r_squared_trad, _, _ = (
        analyzer.calculate_fractal_dimension(
            segments,
            min_box_size=0.001,
            box_size_factor=1.5,
            use_optimal_scaling=False
        )
    )
    
    # Compare results
    print("\n--- Results Comparison ---")
    print(f"Theoretical dimension: {analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type, 'Unknown'):.6f}")
    print(f"Optimal scaling: D = {fd_opt:.6f} ± {error_opt:.6f} (R² = {r_squared_opt:.6f})")
    print(f"Traditional: D = {fd_trad:.6f} ± {error_trad:.6f} (R² = {r_squared_trad:.6f})")
    
    if opt_region:
        print(f"Optimal scaling region: indices {opt_region[0]} to {opt_region[1]}")
        print(f"Box sizes in optimal region: {box_sizes[opt_region[0]]:.6f} to {box_sizes[opt_region[1]]:.6f}")
    
    # Visualize results - Generate detailed plot
    output_file = os.path.join(output_dir, f"{fractal_type}_level{level}_optimal_scaling.png")
    
    # Use the scaling_selector's plot_analysis method if available
    if hasattr(analyzer.box_counter, 'scaling_selector'):
        analyzer.box_counter.scaling_selector.plot_analysis(
            box_sizes, box_counts, opt_region, filename=output_file
        )
    else:
        # Fall back to the visualizer's plot_loglog method
        analyzer.visualizer.plot_loglog(
            box_sizes, box_counts, fd_opt, error_opt, intercept_opt,
            r_squared=r_squared_opt, optimal_region=opt_region, local_slopes=local_slopes,
            custom_filename=output_file
        )
    
    print(f"Plot saved to {output_file}")
    
    return fd_opt, fd_trad, opt_region

def test_multiple_fractals():
    """Test optimal scaling on multiple fractal types."""
    output_dir = "./scaling_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different fractals
    fractals = [
        ('koch', 9),
        ('sierpinski', 8),
        ('dragon', 9),
        ('hilbert', 7),
        ('minkowski', 7)
    ]
    
    # Store results for comparison
    results = []
    
    for fractal_type, level in fractals:
        print(f"\n{'='*50}")
        print(f"Testing {fractal_type.upper()} (level {level})")
        print(f"{'='*50}\n")
        
        try:
            fd_opt, fd_trad, opt_region = test_optimal_scaling(
                fractal_type, level, output_dir
            )
            
            # Get theoretical dimension if available
            analyzer = FractalAnalyzer(fractal_type)
            theoretical = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type, np.nan)
            
            # Store results
            results.append({
                'type': fractal_type,
                'level': level,
                'theoretical': theoretical,
                'optimal': fd_opt,
                'traditional': fd_trad,
                'diff_opt': abs(fd_opt - theoretical) if not np.isnan(theoretical) else np.nan,
                'diff_trad': abs(fd_trad - theoretical) if not np.isnan(theoretical) else np.nan
            })
            
        except Exception as e:
            print(f"Error processing {fractal_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    print("\n\n" + "="*80)
    print(" FRACTAL DIMENSION COMPARISON SUMMARY ".center(80, "="))
    print("="*80)
    print(f"{'Fractal Type':<15} {'Level':^5} {'Theoretical':^12} {'Optimal':^12} {'Traditional':^12} {'Improv.':^10}")
    print("-"*80)
    
    for r in results:
        if not np.isnan(r.get('diff_opt', np.nan)) and not np.isnan(r.get('diff_trad', np.nan)):
            if r['diff_trad'] > 0:
                improvement = (r['diff_trad'] - r['diff_opt']) / r['diff_trad'] * 100
            else:
                improvement = 0.0
        else:
            improvement = np.nan
            
        print(f"{r['type']:<15} {r['level']:^5} {r['theoretical']:^12.6f} {r['optimal']:^12.6f} "
              f"{r['traditional']:^12.6f} {improvement:^10.2f}%")
    
    print("="*80)
    print(f"Note: Improvement shows how much closer the optimal scaling result is to the theoretical value.")
    
if __name__ == "__main__":
    test_multiple_fractals()

