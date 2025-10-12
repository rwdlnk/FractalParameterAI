#!/usr/bin/env python3
"""
Dragon Curve Grid Rotation Test
Testing the hypothesis that Dragon curves form box-like patterns that interfere
with regular grid detection, and that rotating the grid can improve accuracy.

Based on user insight: "dragon is a sequence of lines forming boxes or parts thereof
which might interfere with detecting box intersections from a regular grid.
What about rotating the grid or some such scheme?"
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.box_counting import VectorizedBoxCounter
from fractal_analyzer.core.data_types import SegmentArray, BoundingBox

def rotate_segments(segments, angle_deg):
    """Rotate all segments by given angle (degrees)."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Get segment data
    seg_data = segments.segments  # Shape: (n_segs, 2, 2)

    # Rotate start and end points
    rotated_data = np.zeros_like(seg_data)
    for i in range(2):  # start and end points
        points = seg_data[:, i, :]  # Shape: (n_segs, 2)
        rotated_points = np.dot(points, rotation_matrix.T)
        rotated_data[:, i, :] = rotated_points

    # Create new SegmentArray with rotated data
    return SegmentArray(rotated_data)

def test_dragon_rotation_sensitivity(level=4, rotation_angles=None):
    """Test Dragon curve sensitivity to grid rotation."""
    print(f"🐉 DRAGON CURVE ROTATION SENSITIVITY TEST - Level {level}")
    print("=" * 60)

    if rotation_angles is None:
        # Test every 5 degrees from 0 to 90 (due to symmetry)
        rotation_angles = np.arange(0, 91, 5)

    print(f"Testing {len(rotation_angles)} rotation angles: {rotation_angles}")
    print(f"Hypothesis: Grid alignment affects box counting accuracy")
    print()

    # Generate Dragon curve
    analyzer = FastFractalAnalyzer()
    original_segments = analyzer._generate_dragon_curve(level)
    theoretical_dim = 2.0  # Dragon curve theoretical dimension

    print(f"Dragon L{level}: {original_segments.n_segments:,} segments")
    print(f"Original bbox: ({original_segments.bbox.min_x:.3f}, {original_segments.bbox.min_y:.3f}) to "
          f"({original_segments.bbox.max_x:.3f}, {original_segments.bbox.max_y:.3f})")
    print()

    # Test different rotation angles
    results = []
    best_error = float('inf')
    best_angle = None
    best_result = None

    print("Rotation Test Results:")
    print("Angle (°) | Dimension | Error (%) |   R²    | Time (s)")
    print("-" * 52)

    for angle in rotation_angles:
        try:
            # Rotate segments
            if angle == 0:
                rotated_segments = original_segments
            else:
                rotated_segments = rotate_segments(original_segments, angle)

            # Analyze rotated segments
            start_time = time.time()
            result = analyzer.analyze_segments(
                rotated_segments,
                initial_delta=0.01,  # Use consistent parameters
                delta_factor=1.8,
                num_steps=20
            )
            analysis_time = time.time() - start_time

            if result and hasattr(result, 'dimension'):
                error_pct = abs(result.dimension - theoretical_dim) / theoretical_dim * 100

                print(f"{angle:7.1f} | {result.dimension:9.6f} | {error_pct:8.2f} | {result.r_squared:7.4f} | {analysis_time:7.1f}")

                results.append({
                    'angle': angle,
                    'dimension': result.dimension,
                    'error_pct': error_pct,
                    'r_squared': result.r_squared,
                    'analysis_time': analysis_time,
                    'bbox_width': rotated_segments.bbox.width,
                    'bbox_height': rotated_segments.bbox.height
                })

                if error_pct < best_error:
                    best_error = error_pct
                    best_angle = angle
                    best_result = result
            else:
                print(f"{angle:7.1f} | {'FAILED':>9} | {'---':>8} | {'---':>7} | {'---':>7}")

        except Exception as e:
            print(f"{angle:7.1f} | {'ERROR':>9} | {'---':>8} | {'---':>7} | {'---':>7}")

    # Analysis of results
    print()
    print("🔍 ROTATION SENSITIVITY ANALYSIS")
    print("=" * 40)

    if results:
        errors = [r['error_pct'] for r in results]
        dimensions = [r['dimension'] for r in results]
        angles = [r['angle'] for r in results]

        min_error = np.min(errors)
        max_error = np.max(errors)
        error_range = max_error - min_error
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        print(f"Error Statistics:")
        print(f"  Minimum error: {min_error:.2f}% at {angles[np.argmin(errors)]:.1f}°")
        print(f"  Maximum error: {max_error:.2f}% at {angles[np.argmax(errors)]:.1f}°")
        print(f"  Error range: {error_range:.2f}% ({error_range/mean_error*100:.1f}% relative)")
        print(f"  Mean error: {mean_error:.2f} ± {std_error:.2f}%")

        print(f"\nDimension Statistics:")
        print(f"  Minimum: {np.min(dimensions):.6f}")
        print(f"  Maximum: {np.max(dimensions):.6f}")
        print(f"  Range: {np.max(dimensions) - np.min(dimensions):.6f}")
        print(f"  Mean: {np.mean(dimensions):.6f} ± {np.std(dimensions):.6f}")

        # Improvement assessment
        baseline_error = errors[0]  # 0° rotation
        improvement_pct = (baseline_error - min_error) / baseline_error * 100

        print(f"\n🎯 Grid Rotation Assessment:")
        print(f"  Baseline (0°): {baseline_error:.2f}% error")
        print(f"  Best angle: {best_angle:.1f}° with {min_error:.2f}% error")
        print(f"  Improvement: {improvement_pct:.1f}% reduction in error")

        if improvement_pct > 10:
            print("  ✅ SIGNIFICANT IMPROVEMENT - Grid rotation helps!")
        elif improvement_pct > 5:
            print("  ✅ MODERATE IMPROVEMENT - Grid rotation has some benefit")
        elif improvement_pct > 1:
            print("  🔄 MINOR IMPROVEMENT - Small but measurable benefit")
        else:
            print("  ❌ NO SIGNIFICANT IMPROVEMENT - Grid rotation doesn't help much")

        # Check for systematic patterns
        # Look for optimal angles (e.g., 45° for Dragon curves)
        if 45.0 in angles:
            angle_45_idx = angles.index(45.0)
            angle_45_error = errors[angle_45_idx]
            print(f"\n📐 45° Analysis (Dragon symmetry):")
            print(f"  45° rotation error: {angle_45_error:.2f}%")
            improvement_45 = (baseline_error - angle_45_error) / baseline_error * 100
            print(f"  Improvement over 0°: {improvement_45:.1f}%")

        return results, best_angle, best_result
    else:
        print("❌ No successful results to analyze")
        return [], None, None

def test_optimal_angle_detailed(level=4, optimal_angle=45.0):
    """Test the optimal angle in more detail with different parameters."""
    print(f"\n🔍 DETAILED ANALYSIS AT OPTIMAL ANGLE ({optimal_angle}°)")
    print("=" * 50)

    analyzer = FastFractalAnalyzer()
    original_segments = analyzer._generate_dragon_curve(level)
    rotated_segments = rotate_segments(original_segments, optimal_angle)
    theoretical_dim = 2.0

    # Test different parameter combinations at optimal angle
    test_params = [
        {'initial_delta': 0.005, 'delta_factor': 1.5, 'num_steps': 25},
        {'initial_delta': 0.01, 'delta_factor': 1.5, 'num_steps': 20},
        {'initial_delta': 0.01, 'delta_factor': 1.8, 'num_steps': 20},
        {'initial_delta': 0.02, 'delta_factor': 1.8, 'num_steps': 15},
        {'initial_delta': 0.005, 'delta_factor': 2.0, 'num_steps': 30},
    ]

    print(f"Testing {len(test_params)} parameter combinations at {optimal_angle}° rotation:")
    print("δ₀      | Factor | Steps | Dimension | Error (%) |   R²")
    print("-" * 55)

    best_error = float('inf')
    best_combo = None

    for params in test_params:
        result = analyzer.analyze_segments(rotated_segments, **params)

        if result and hasattr(result, 'dimension'):
            error_pct = abs(result.dimension - theoretical_dim) / theoretical_dim * 100

            print(f"{params['initial_delta']:7.3f} | {params['delta_factor']:6.1f} | "
                  f"{params['num_steps']:5d} | {result.dimension:9.6f} | "
                  f"{error_pct:8.2f} | {result.r_squared:5.3f}")

            if error_pct < best_error:
                best_error = error_pct
                best_combo = params.copy()
                best_combo['result'] = result
        else:
            print(f"{params['initial_delta']:7.3f} | {params['delta_factor']:6.1f} | "
                  f"{params['num_steps']:5d} | {'FAILED':>9} | {'---':>8} | {'---':>5}")

    if best_combo:
        print(f"\n🏆 Best combination at {optimal_angle}°:")
        print(f"  Parameters: δ₀={best_combo['initial_delta']}, "
              f"factor={best_combo['delta_factor']}, steps={best_combo['num_steps']}")
        print(f"  Dimension: {best_combo['result'].dimension:.6f}")
        print(f"  Error: {best_error:.2f}%")
        print(f"  R²: {best_combo['result'].r_squared:.6f}")

    return best_combo

def main():
    print("🐉 DRAGON CURVE GRID ROTATION HYPOTHESIS TEST")
    print("=" * 60)
    print("Testing if grid rotation improves Dragon curve box counting accuracy")
    print("Hypothesis: Dragon segments form box-like patterns that interfere with regular grids")
    print()

    # Test Dragon L4 (the problematic case we identified)
    level = 4

    # Test 1: Basic rotation sensitivity
    results, best_angle, best_result = test_dragon_rotation_sensitivity(level)

    if best_angle is not None:
        # Test 2: Detailed analysis at optimal angle
        best_combo = test_optimal_angle_detailed(level, best_angle)

        # Summary
        print(f"\n🎯 FINAL ASSESSMENT - Dragon L{level}")
        print("=" * 40)
        print(f"Grid rotation hypothesis: {'✅ CONFIRMED' if best_angle != 0 else '❌ NOT CONFIRMED'}")

        if best_angle != 0:
            print(f"Optimal rotation angle: {best_angle}°")
            baseline_error = results[0]['error_pct']  # 0° baseline
            improvement = (baseline_error - best_result.dimension + 2.0) / 2.0 * 100
            print(f"Error improvement: {baseline_error:.2f}% → {abs(best_result.dimension - 2.0)/2.0*100:.2f}%")

            if best_combo:
                print(f"Best overall result: {best_combo['result'].dimension:.6f} "
                      f"(error: {abs(best_combo['result'].dimension - 2.0)/2.0*100:.2f}%)")
        else:
            print("Grid rotation does not significantly improve Dragon curve analysis")
    else:
        print("❌ Could not complete rotation analysis")

if __name__ == "__main__":
    main()