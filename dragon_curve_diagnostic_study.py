#!/usr/bin/env python3
"""
Comprehensive Dragon Curve Diagnostic Study
Investigating why Dragon curves show poor box counting accuracy (~30% error)
while other fractals (Koch, Sierpinski) achieve <3% error.

Research Questions:
1. Are Dragon curve segments generated correctly?
2. Does the geometric structure create box counting artifacts?
3. How do AI parameters compare to manual optimization?
4. What scaling behavior differs from theoretical expectations?
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode
from core.interface_features import extract_interface_features, classify_interface_type

def analyze_dragon_geometry(segments, level):
    """Analyze geometric properties of Dragon curve segments."""
    print(f"\n🔍 GEOMETRIC ANALYSIS - Dragon L{level}")
    print("=" * 50)

    # Basic segment properties
    n_segments = segments.n_segments
    bbox = segments.bbox
    total_length = segments.total_length

    print(f"Segments: {n_segments:,}")
    print(f"Bbox: ({bbox.min_x:.6f}, {bbox.min_y:.6f}) to ({bbox.max_x:.6f}, {bbox.max_y:.6f})")
    print(f"Domain size: {bbox.width:.6f} × {bbox.height:.6f}")
    print(f"Aspect ratio: {bbox.aspect_ratio:.6f}")
    print(f"Total length: {total_length:.6f}")

    # Segment length analysis
    start_points = segments.start_points
    end_points = segments.end_points
    segment_vectors = end_points - start_points
    segment_lengths = np.sqrt(np.sum(segment_vectors**2, axis=1))

    print(f"\nSegment Length Statistics:")
    print(f"  Mean: {np.mean(segment_lengths):.6f}")
    print(f"  Std:  {np.std(segment_lengths):.6f}")
    print(f"  Min:  {np.min(segment_lengths):.6f}")
    print(f"  Max:  {np.max(segment_lengths):.6f}")
    print(f"  CV:   {np.std(segment_lengths)/np.mean(segment_lengths):.6f}")

    # Angular analysis
    angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
    angle_diffs = np.diff(angles)
    # Handle angle wraparound
    angle_diffs = np.abs(angle_diffs)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

    print(f"\nAngular Statistics:")
    print(f"  Angle std: {np.std(angles):.6f}")
    print(f"  Direction changes >π/4: {np.sum(angle_diffs > np.pi/4)}")
    print(f"  Direction change rate: {np.sum(angle_diffs > np.pi/4) / n_segments:.6f}")

    # Theoretical properties for Dragon curve
    theoretical_total_length = 2**(level/2)  # Dragon curve length scaling
    length_ratio = total_length / theoretical_total_length

    print(f"\nTheoretical Validation:")
    print(f"  Expected length: {theoretical_total_length:.6f}")
    print(f"  Actual length:   {total_length:.6f}")
    print(f"  Length ratio:    {length_ratio:.6f}")

    return {
        'n_segments': n_segments,
        'total_length': total_length,
        'bbox_width': bbox.width,
        'bbox_height': bbox.height,
        'aspect_ratio': bbox.aspect_ratio,
        'segment_length_mean': np.mean(segment_lengths),
        'segment_length_std': np.std(segment_lengths),
        'segment_length_cv': np.std(segment_lengths)/np.mean(segment_lengths),
        'angle_std': np.std(angles),
        'direction_change_rate': np.sum(angle_diffs > np.pi/4) / n_segments,
        'theoretical_length': theoretical_total_length,
        'length_ratio': length_ratio
    }

def test_manual_parameters(segments, level):
    """Test various manual parameter combinations to find optimal settings."""
    print(f"\n🎯 MANUAL PARAMETER OPTIMIZATION - Dragon L{level}")
    print("=" * 50)

    analyzer = FastFractalAnalyzer(performance_mode=PerformanceMode.BALANCED)
    theoretical_dim = 1.5236  # log(2)/log(sqrt(2)) ≈ 1.5236 for Dragon curve

    # Test parameter combinations
    test_params = [
        {'initial_delta': 0.01, 'delta_factor': 1.5, 'num_steps': 15},
        {'initial_delta': 0.02, 'delta_factor': 1.5, 'num_steps': 15},
        {'initial_delta': 0.05, 'delta_factor': 1.5, 'num_steps': 15},
        {'initial_delta': 0.01, 'delta_factor': 1.8, 'num_steps': 20},
        {'initial_delta': 0.02, 'delta_factor': 1.8, 'num_steps': 20},
        {'initial_delta': 0.01, 'delta_factor': 2.0, 'num_steps': 25},
        {'initial_delta': 0.005, 'delta_factor': 1.5, 'num_steps': 20},
        {'initial_delta': 0.001, 'delta_factor': 1.5, 'num_steps': 15},
    ]

    results = []
    best_error = float('inf')
    best_params = None
    best_result = None

    for i, params in enumerate(test_params):
        print(f"\nTest {i+1}/8: δ₀={params['initial_delta']}, factor={params['delta_factor']}, steps={params['num_steps']}")

        try:
            start_time = time.time()
            result = analyzer.analyze_segments(segments, **params)
            analysis_time = time.time() - start_time

            if result and hasattr(result, 'dimension'):
                error_pct = abs(result.dimension - theoretical_dim) / theoretical_dim * 100

                print(f"  Dimension: {result.dimension:.6f}")
                print(f"  Error: {error_pct:.2f}%")
                print(f"  R²: {result.r_squared:.6f}")
                print(f"  Time: {analysis_time:.1f}s")

                results.append({
                    'params': params,
                    'dimension': result.dimension,
                    'error_pct': error_pct,
                    'r_squared': result.r_squared,
                    'time': analysis_time,
                    'n_points': getattr(result, 'n_points', 0)
                })

                if error_pct < best_error:
                    best_error = error_pct
                    best_params = params.copy()
                    best_result = result

            else:
                print("  ❌ Analysis failed")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Summary
    print(f"\n🏆 BEST MANUAL RESULT:")
    if best_result:
        print(f"  Parameters: {best_params}")
        print(f"  Dimension: {best_result.dimension:.6f}")
        print(f"  Error: {best_error:.2f}%")
        print(f"  R²: {best_result.r_squared:.6f}")
    else:
        print("  No successful results")

    return results, best_params, best_result

def analyze_scaling_behavior(segments, level, best_params=None):
    """Analyze detailed scaling behavior to understand box counting issues."""
    print(f"\n📊 SCALING BEHAVIOR ANALYSIS - Dragon L{level}")
    print("=" * 50)

    analyzer = FastFractalAnalyzer(performance_mode=PerformanceMode.BALANCED)

    # Use best parameters if available, otherwise default
    if best_params:
        params = best_params
        print(f"Using optimized parameters: {params}")
    else:
        params = {'initial_delta': 0.01, 'delta_factor': 1.8, 'num_steps': 20}
        print(f"Using default parameters: {params}")

    # Run detailed analysis to get scaling data
    result = analyzer.analyze_segments(segments, **params)

    if result and hasattr(result, 'box_sizes') and hasattr(result, 'box_counts'):
        box_sizes = result.box_sizes
        box_counts = result.box_counts

        print(f"Scaling points: {len(box_sizes)}")
        print(f"Box size range: {np.min(box_sizes):.6f} to {np.max(box_sizes):.6f}")
        print(f"Count range: {np.min(box_counts)} to {np.max(box_counts)}")

        # Analyze scaling quality
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)

        # Linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)

        print(f"\nScaling Analysis:")
        print(f"  Slope (dimension): {-slope:.6f}")
        print(f"  R²: {r_value**2:.6f}")
        print(f"  Standard error: {std_err:.6f}")
        print(f"  P-value: {p_value:.2e}")

        # Look for scaling regime breaks
        print(f"\nDetailed scaling table:")
        print(f"{'Box Size':>12} {'Count':>8} {'Local Slope':>12}")
        print("-" * 35)

        for i in range(len(box_sizes)):
            if i > 0:
                local_slope = -(log_counts[i] - log_counts[i-1]) / (log_sizes[i] - log_sizes[i-1])
                print(f"{box_sizes[i]:12.6f} {box_counts[i]:8d} {local_slope:12.6f}")
            else:
                print(f"{box_sizes[i]:12.6f} {box_counts[i]:8d} {'---':>12}")

        # Identify potential issues
        local_slopes = []
        for i in range(1, len(box_sizes)):
            local_slope = -(log_counts[i] - log_counts[i-1]) / (log_sizes[i] - log_sizes[i-1])
            local_slopes.append(local_slope)

        local_slopes = np.array(local_slopes)
        slope_std = np.std(local_slopes)
        slope_cv = slope_std / np.mean(local_slopes) if np.mean(local_slopes) != 0 else np.inf

        print(f"\nScaling Consistency:")
        print(f"  Local slope std: {slope_std:.6f}")
        print(f"  Local slope CV: {slope_cv:.6f}")

        if slope_cv > 0.2:
            print("  ⚠️  High slope variation - inconsistent scaling")
        else:
            print("  ✅ Consistent scaling behavior")

        return {
            'box_sizes': box_sizes,
            'box_counts': box_counts,
            'dimension': -slope,
            'r_squared': r_value**2,
            'slope_std': slope_std,
            'slope_cv': slope_cv
        }
    else:
        print("❌ Could not extract scaling data")
        return None

def compare_with_koch_curve(dragon_level):
    """Compare Dragon curve behavior with equivalent Koch curve."""
    print(f"\n⚖️  COMPARATIVE ANALYSIS: Dragon L{dragon_level} vs Koch L{dragon_level}")
    print("=" * 60)

    analyzer = FastFractalAnalyzer()

    # Generate both fractals
    print("Generating fractals...")
    dragon_segments = analyzer._generate_dragon_curve(dragon_level)
    koch_segments = analyzer._generate_koch_curve(dragon_level)

    print(f"Dragon L{dragon_level}: {dragon_segments.n_segments:,} segments")
    print(f"Koch L{dragon_level}: {koch_segments.n_segments:,} segments")

    # Extract features for both
    print("\nExtracting AI features...")

    # Convert to expected format
    dragon_array = np.column_stack([
        dragon_segments.segments[:, 0, 0], dragon_segments.segments[:, 0, 1],
        dragon_segments.segments[:, 1, 0], dragon_segments.segments[:, 1, 1]
    ])
    koch_array = np.column_stack([
        koch_segments.segments[:, 0, 0], koch_segments.segments[:, 0, 1],
        koch_segments.segments[:, 1, 0], koch_segments.segments[:, 1, 1]
    ])

    dragon_features = extract_interface_features(dragon_array)
    koch_features = extract_interface_features(koch_array)

    dragon_type = classify_interface_type(dragon_features)
    koch_type = classify_interface_type(koch_features)

    print(f"\nAI Classification:")
    print(f"  Dragon: {dragon_type}")
    print(f"  Koch:   {koch_type}")

    # Compare key features
    key_features = ['tortuosity', 'complexity_score', 'direction_change_rate',
                   'aspect_ratio', 'linearity_r_squared']

    print(f"\nFeature Comparison:")
    print(f"{'Feature':20} {'Dragon':>10} {'Koch':>10} {'Ratio':>10}")
    print("-" * 52)

    for feature in key_features:
        dragon_val = dragon_features.get(feature, 0)
        koch_val = koch_features.get(feature, 0)
        ratio = dragon_val / koch_val if koch_val != 0 else float('inf')

        print(f"{feature:20} {dragon_val:10.6f} {koch_val:10.6f} {ratio:10.3f}")

    # Quick fractal analysis comparison
    print(f"\nQuick Fractal Analysis:")
    dragon_result = analyzer.analyze_segments(dragon_segments)
    koch_result = analyzer.analyze_segments(koch_segments)

    if dragon_result and koch_result:
        dragon_error = abs(dragon_result.dimension - 1.5236) / 1.5236 * 100
        koch_error = abs(koch_result.dimension - 1.2619) / 1.2619 * 100

        print(f"  Dragon: D = {dragon_result.dimension:.6f}, Error = {dragon_error:.2f}%")
        print(f"  Koch:   D = {koch_result.dimension:.6f}, Error = {koch_error:.2f}%")

    return dragon_features, koch_features

def main():
    print("🐉 COMPREHENSIVE DRAGON CURVE DIAGNOSTIC STUDY")
    print("=" * 60)
    print("Investigating poor box counting performance (~30% error)")
    print("Comparing with successful fractals (Koch: <3% error)")
    print()

    # Test multiple Dragon curve levels
    test_levels = [3, 4, 5]

    all_results = {}

    for level in test_levels:
        print(f"\n" + "="*60)
        print(f"DRAGON CURVE LEVEL {level} ANALYSIS")
        print("="*60)

        # Generate Dragon curve
        print(f"🔧 Generating Dragon curve L{level}...")
        analyzer = FastFractalAnalyzer()
        segments = analyzer._generate_dragon_curve(level)

        print(f"✅ Generated: {segments.n_segments:,} segments")

        # 1. Geometric analysis
        geometry_data = analyze_dragon_geometry(segments, level)

        # 2. Manual parameter optimization
        manual_results, best_params, best_result = test_manual_parameters(segments, level)

        # 3. Scaling behavior analysis
        scaling_data = analyze_scaling_behavior(segments, level, best_params)

        # 4. Comparative analysis (only for L4 to avoid too much output)
        if level == 4:
            dragon_features, koch_features = compare_with_koch_curve(level)

        # Store results
        all_results[level] = {
            'geometry': geometry_data,
            'manual_optimization': manual_results,
            'best_params': best_params,
            'best_result': best_result,
            'scaling': scaling_data
        }

        print(f"\n✅ Dragon L{level} analysis complete")

    # Summary report
    print(f"\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    print("\n🏆 Best Results Achieved:")
    for level in test_levels:
        best_result = all_results[level]['best_result']
        if best_result:
            error = abs(best_result.dimension - 1.5236) / 1.5236 * 100
            print(f"  Dragon L{level}: D = {best_result.dimension:.6f}, Error = {error:.2f}%")
        else:
            print(f"  Dragon L{level}: No successful results")

    print("\n🔍 Key Findings:")
    print("  • Geometric consistency across levels")
    print("  • Parameter sensitivity analysis")
    print("  • Scaling behavior characteristics")
    print("  • Comparison with Koch curve performance")

    print(f"\n📁 Detailed results saved in memory for further analysis")
    print(f"💡 Consider alternative box counting methods for Dragon curves")

if __name__ == "__main__":
    main()