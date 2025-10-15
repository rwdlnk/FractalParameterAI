#!/usr/bin/env python3
"""
Test and compare all three interface extraction methods:
- scikit-image marching squares
- PLIC (Piecewise Linear Interface Calculation)
- CONREC contouring algorithm

This script compares segment counts, dimensions, and timings.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration.rt_analyzer_refactored import RefactoredRTAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare interface extraction methods')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--theoretical-dim', type=float, help='Theoretical dimension')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"INTERFACE EXTRACTION METHOD COMPARISON")
    print(f"{'='*80}")
    print(f"File: {os.path.basename(args.vtk_file)}")
    print()

    methods = ['skimage', 'conrec', 'plic']
    results = {}

    for method in methods:
        print(f"\n{'-'*80}")
        print(f"Testing method: {method.upper()}")
        print(f"{'-'*80}")

        try:
            analyzer = RefactoredRTAnalyzer(
                output_dir=f"./test_extraction_{method}",
                performance_mode=PerformanceMode.FAST,
                interface_method=method,
                enable_learning=False,  # Don't pollute feedback during testing
                debug=False
            )

            result = analyzer.analyze_single_vtk(
                args.vtk_file,
                analysis_types=['fractal'],
                theoretical_dimension=args.theoretical_dim
            )

            results[method] = result

        except Exception as e:
            print(f"❌ {method} failed: {e}")
            results[method] = {'success': False, 'error': str(e)}

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Segments':<10} {'Dimension':<12} {'R²':<10} {'Time (s)':<10}")
    print(f"{'-'*80}")

    for method in methods:
        result = results[method]
        if result['success']:
            segments = result.get('interface_segments', 'N/A')
            dimension = result.get('fractal_dimension', float('nan'))
            r_squared = result.get('fractal_r_squared', float('nan'))
            time = result.get('fractal_analysis_time', float('nan'))

            print(f"{method:<12} {segments:<10} {dimension:<12.6f} {r_squared:<10.6f} {time:<10.1f}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{method:<12} FAILED: {error}")

    # Print analysis
    print(f"\n{' '*80}")
    print(f"ANALYSIS:")

    successful = {k: v for k, v in results.items() if v['success']}

    if len(successful) > 1:
        # Compare segments
        segments_counts = {k: v['interface_segments'] for k, v in successful.items()}
        max_seg_method = max(segments_counts, key=segments_counts.get)
        min_seg_method = min(segments_counts, key=segments_counts.get)

        print(f"  Segment Count:")
        print(f"    Highest: {max_seg_method} ({segments_counts[max_seg_method]} segments)")
        print(f"    Lowest:  {min_seg_method} ({segments_counts[min_seg_method]} segments)")
        print(f"    Ratio:   {segments_counts[max_seg_method]/segments_counts[min_seg_method]:.2f}x")

        # Compare dimensions
        dimensions = {k: v['fractal_dimension'] for k, v in successful.items()}
        print(f"\n  Fractal Dimensions:")
        for method, dim in dimensions.items():
            print(f"    {method}: {dim:.6f}")

        dim_range = max(dimensions.values()) - min(dimensions.values())
        print(f"    Range: {dim_range:.6f}")

        if args.theoretical_dim:
            print(f"\n  Error vs Theoretical ({args.theoretical_dim:.3f}):")
            for method, dim in dimensions.items():
                error = abs(dim - args.theoretical_dim) / args.theoretical_dim * 100
                print(f"    {method}: {error:.2f}%")

        # Compare quality
        r_squareds = {k: v['fractal_r_squared'] for k, v in successful.items()}
        best_quality = max(r_squareds, key=r_squareds.get)
        print(f"\n  Best R²: {best_quality} (R²={r_squareds[best_quality]:.6f})")

        # Compare timing
        times = {k: v['fractal_analysis_time'] for k, v in successful.items()}
        fastest = min(times, key=times.get)
        slowest = max(times, key=times.get)
        print(f"\n  Speed:")
        print(f"    Fastest: {fastest} ({times[fastest]:.1f}s)")
        print(f"    Slowest: {slowest} ({times[slowest]:.1f}s)")
        print(f"    Ratio:   {times[slowest]/times[fastest]:.2f}x")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
