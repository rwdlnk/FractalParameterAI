#!/usr/bin/env python3
"""
Simplified Dragon Curve Diagnostic Study
Focusing on key insights into why Dragon curves show poor box counting accuracy.
"""

import sys
import time
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

def analyze_dragon_geometry(segments, level):
    """Quick geometric analysis of Dragon curve."""
    print(f"\n🔍 Dragon L{level} Geometry Analysis:")
    print(f"  Segments: {segments.n_segments:,}")
    print(f"  Bbox: {segments.bbox}")

    # Basic segment properties
    start_points = segments.start_points
    end_points = segments.end_points
    segment_vectors = end_points - start_points
    segment_lengths = np.sqrt(np.sum(segment_vectors**2, axis=1))

    print(f"  Segment lengths: mean={np.mean(segment_lengths):.6f}, std={np.std(segment_lengths):.6f}")

    # Angular properties
    angles = np.arctan2(segment_vectors[:, 1], segment_vectors[:, 0])
    angle_diffs = np.diff(angles)
    angle_diffs = np.abs(angle_diffs)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

    large_turns = np.sum(angle_diffs > np.pi/4)
    print(f"  Direction changes >45°: {large_turns} ({large_turns/segments.n_segments:.3f} rate)")

    return {
        'n_segments': segments.n_segments,
        'mean_length': np.mean(segment_lengths),
        'length_std': np.std(segment_lengths),
        'large_turns': large_turns,
        'turn_rate': large_turns/segments.n_segments
    }

def test_multiple_parameters(segments, level):
    """Test various parameter combinations quickly."""
    print(f"\n🎯 Parameter Testing - Dragon L{level}:")

    theoretical_dim = 1.5236
    analyzer = FastFractalAnalyzer()

    # Test different parameter sets
    param_sets = [
        "Default AI",
        "Fine resolution",
        "Coarse resolution",
        "Extended range",
        "Conservative"
    ]

    results = []

    for i, name in enumerate(param_sets):
        print(f"  Testing {name}...")

        try:
            if i == 0:  # Default AI - let system choose
                result = analyzer.analyze(segments)
            elif i == 1:  # Fine resolution
                result = analyzer.analyze(segments,
                    initial_delta=0.001, delta_factor=1.5, num_steps=25)
            elif i == 2:  # Coarse resolution
                result = analyzer.analyze(segments,
                    initial_delta=0.05, delta_factor=2.0, num_steps=10)
            elif i == 3:  # Extended range
                result = analyzer.analyze(segments,
                    initial_delta=0.0005, delta_factor=1.8, num_steps=30)
            else:  # Conservative
                result = analyzer.analyze(segments,
                    initial_delta=0.01, delta_factor=1.5, num_steps=15)

            if result and hasattr(result, 'dimension'):
                error = abs(result.dimension - theoretical_dim) / theoretical_dim * 100
                results.append((name, result.dimension, error, result.r_squared))
                print(f"    D = {result.dimension:.6f}, Error = {error:.2f}%, R² = {result.r_squared:.6f}")
            else:
                print(f"    Failed")
                results.append((name, None, None, None))

        except Exception as e:
            print(f"    Error: {e}")
            results.append((name, None, None, None))

    return results

def compare_with_successful_fractal(dragon_level):
    """Compare Dragon with Koch curve at same level."""
    print(f"\n⚖️  Dragon vs Koch Comparison (Level {dragon_level}):")

    analyzer = FastFractalAnalyzer()

    # Generate both
    dragon_segments = analyzer._generate_dragon_curve(dragon_level)
    koch_segments = analyzer._generate_koch_curve(dragon_level)

    print(f"  Dragon L{dragon_level}: {dragon_segments.n_segments} segments")
    print(f"  Koch L{dragon_level}: {koch_segments.n_segments} segments")

    # Quick analysis of both
    try:
        dragon_result = analyzer.analyze(dragon_segments)
        koch_result = analyzer.analyze(koch_segments)

        if dragon_result and koch_result:
            dragon_error = abs(dragon_result.dimension - 1.5236) / 1.5236 * 100
            koch_error = abs(koch_result.dimension - 1.2619) / 1.2619 * 100

            print(f"  Dragon: D = {dragon_result.dimension:.6f}, Error = {dragon_error:.2f}%")
            print(f"  Koch:   D = {koch_result.dimension:.6f}, Error = {koch_error:.2f}%")
            print(f"  Error ratio: {dragon_error/koch_error:.1f}x worse")

            return {
                'dragon_error': dragon_error,
                'koch_error': koch_error,
                'error_ratio': dragon_error/koch_error
            }

    except Exception as e:
        print(f"  Analysis failed: {e}")

    return None

def main():
    print("🐉 SIMPLIFIED DRAGON CURVE DIAGNOSTIC")
    print("=" * 50)
    print("Investigating poor box counting performance")

    test_levels = [3, 4]  # Start with smaller levels for speed

    all_results = {}

    for level in test_levels:
        print(f"\n{'='*50}")
        print(f"DRAGON LEVEL {level} ANALYSIS")
        print('='*50)

        # Generate Dragon curve
        analyzer = FastFractalAnalyzer()
        segments = analyzer._generate_dragon_curve(level)

        # Geometric analysis
        geometry = analyze_dragon_geometry(segments, level)

        # Parameter testing
        param_results = test_multiple_parameters(segments, level)

        # Comparison with Koch
        comparison = compare_with_successful_fractal(level)

        all_results[level] = {
            'geometry': geometry,
            'parameters': param_results,
            'comparison': comparison
        }

    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print('='*50)

    print("\n🏆 Best Dragon Results:")
    for level in test_levels:
        param_results = all_results[level]['parameters']
        best_error = min([r[2] for r in param_results if r[2] is not None], default=None)
        if best_error:
            best_method = [r[0] for r in param_results if r[2] == best_error][0]
            print(f"  L{level}: {best_error:.2f}% error ({best_method})")
        else:
            print(f"  L{level}: All methods failed")

    print("\n🔍 Key Findings:")
    for level in test_levels:
        geometry = all_results[level]['geometry']
        comparison = all_results[level]['comparison']

        print(f"  L{level}: {geometry['turn_rate']:.3f} turn rate", end="")
        if comparison:
            print(f", {comparison['error_ratio']:.1f}x worse than Koch")
        else:
            print()

    print("\n💡 Hypotheses to investigate:")
    print("  • High directional change rate may confuse box counting")
    print("  • Dragon curve geometry may create box counting artifacts")
    print("  • Need specialized parameters for highly turning fractals")
    print("  • Consider alternative fractal dimension methods")

if __name__ == "__main__":
    main()