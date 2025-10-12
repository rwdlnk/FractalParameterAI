#!/usr/bin/env python3
"""
Rotation-Optimized Box Counting for Dragon Curves
Implements pre-expanded grid strategy to test rotation angle as an optimizable parameter.
Based on the insight that grid alignment affects fractal dimension accuracy.
"""

import sys
import time
import math
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.box_counting import VectorizedBoxCounter
from fractal_analyzer.core.data_types import SegmentArray, BoundingBox

def rotate_segments(segments, angle_deg):
    """Rotate segments by given angle (degrees) around their center."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Get original bounding box center
    bbox = segments.bbox
    center_x = (bbox.min_x + bbox.max_x) / 2
    center_y = (bbox.min_y + bbox.max_y) / 2

    # Get segment data and rotate around center
    seg_data = segments.segments.copy()  # Shape: (n_segs, 2, 2)

    for i in range(2):  # Start and end points
        points = seg_data[:, i, :]  # Shape: (n_segs, 2)

        # Translate to origin
        points[:, 0] -= center_x
        points[:, 1] -= center_y

        # Rotate
        rotated_points = np.dot(points, rotation_matrix.T)

        # Translate back
        rotated_points[:, 0] += center_x
        rotated_points[:, 1] += center_y

        seg_data[:, i, :] = rotated_points

    return SegmentArray(seg_data)

def create_expanded_domain(original_bbox, expansion_factor=math.sqrt(2)):
    """Create expanded domain to accommodate any rotation angle."""
    center_x = (original_bbox.min_x + original_bbox.max_x) / 2
    center_y = (original_bbox.min_y + original_bbox.max_y) / 2

    half_width = original_bbox.width / 2 * expansion_factor
    half_height = original_bbox.height / 2 * expansion_factor

    return BoundingBox(
        min_x=center_x - half_width,
        min_y=center_y - half_height,
        max_x=center_x + half_width,
        max_y=center_y + half_height
    )

def rotation_optimized_box_count(segments, box_size, test_angles=None):
    """
    Count boxes with rotation optimization using pre-expanded domain.
    Tests multiple rotation angles and returns the minimum count.
    """
    if test_angles is None:
        test_angles = np.arange(0, 90, 5)  # Test every 5 degrees

    # Create pre-expanded domain (√2 factor for 45° rotation)
    original_bbox = segments.bbox
    expanded_domain = create_expanded_domain(original_bbox)

    print(f"Original domain: {original_bbox.width:.3f} × {original_bbox.height:.3f}")
    print(f"Expanded domain: {expanded_domain.width:.3f} × {expanded_domain.height:.3f}")
    print(f"Expansion factor: {expanded_domain.width/original_bbox.width:.3f}")

    box_counter = VectorizedBoxCounter()
    best_count = float('inf')
    best_angle = 0
    all_counts = []

    for angle in test_angles:
        # Rotate segments
        rotated_segments = rotate_segments(segments, angle)

        # Count boxes in expanded domain
        try:
            # Calculate grid parameters for expanded domain
            n_boxes_x = int(np.ceil(expanded_domain.width / box_size))
            n_boxes_y = int(np.ceil(expanded_domain.height / box_size))

            # Create grid starting from expanded domain corner
            occupied = box_counter._vectorized_intersection_test(
                rotated_segments, box_size, expanded_domain, n_boxes_x, n_boxes_y
            )

            count = int(np.sum(occupied))
            all_counts.append((angle, count))

            if count < best_count:
                best_count = count
                best_angle = angle

        except Exception as e:
            print(f"Error at angle {angle}°: {e}")
            continue

    return best_count, best_angle, all_counts

def rotation_optimized_fractal_analysis(segments, theoretical_dim=None,
                                       rotation_angles=None, verbose=True):
    """
    Complete fractal dimension analysis with rotation optimization.
    """
    if rotation_angles is None:
        rotation_angles = np.arange(0, 91, 5)  # 0° to 90° every 5°

    if verbose:
        print(f"🔄 ROTATION-OPTIMIZED FRACTAL ANALYSIS")
        print(f"Testing {len(rotation_angles)} rotation angles")
        print(f"Segments: {segments.n_segments:,}")
        print()

    # Test different box sizes with rotation optimization
    bbox = segments.bbox
    min_box_size = max(bbox.width, bbox.height) / 100  # Finest scale
    max_box_size = max(bbox.width, bbox.height) / 5    # Coarsest scale

    box_sizes = []
    box_counts = []
    optimal_angles = []

    # Generate box size sequence
    current_size = max_box_size
    size_factor = 1.5

    if verbose:
        print("Box Size    | Best Angle | Min Count | Max Count | Variation | Time (s)")
        print("-" * 70)

    while current_size >= min_box_size:
        start_time = time.time()

        # Find optimal rotation for this box size
        best_count, best_angle, all_counts = rotation_optimized_box_count(
            segments, current_size, rotation_angles
        )

        if best_count > 0:
            counts_only = [c for _, c in all_counts]
            min_count = min(counts_only)
            max_count = max(counts_only)
            variation = (max_count - min_count) / min_count * 100 if min_count > 0 else 0

            box_sizes.append(current_size)
            box_counts.append(best_count)
            optimal_angles.append(best_angle)

            elapsed = time.time() - start_time
            if verbose:
                print(f"{current_size:10.6f} | {best_angle:9.1f}° | {min_count:8d} | {max_count:8d} | {variation:7.1f}% | {elapsed:7.1f}")

        current_size /= size_factor

    if len(box_sizes) < 3:
        if verbose:
            print("❌ Insufficient data points for dimension calculation")
        return None

    # Calculate fractal dimension from log-log fit
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)

    # Linear regression: log(N) = -D * log(r) + const
    slope, intercept = np.polyfit(log_sizes, log_counts, 1)
    dimension = -slope

    # Calculate R-squared
    log_counts_pred = slope * log_sizes + intercept
    ss_res = np.sum((log_counts - log_counts_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if verbose:
        print(f"\n📊 ROTATION-OPTIMIZED RESULTS:")
        print(f"Fractal dimension: {dimension:.6f}")
        if theoretical_dim:
            error_pct = abs(dimension - theoretical_dim) / theoretical_dim * 100
            print(f"Theoretical: {theoretical_dim:.6f}")
            print(f"Error: {error_pct:.2f}%")
        print(f"R²: {r_squared:.6f}")
        print(f"Box sizes tested: {len(box_sizes)}")
        print(f"Optimal angles range: {min(optimal_angles):.1f}° to {max(optimal_angles):.1f}°")

    return {
        'dimension': dimension,
        'r_squared': r_squared,
        'box_sizes': np.array(box_sizes),
        'box_counts': np.array(box_counts),
        'optimal_angles': np.array(optimal_angles),
        'theoretical_dimension': theoretical_dim,
        'error_pct': abs(dimension - theoretical_dim) / theoretical_dim * 100 if theoretical_dim else None
    }

def compare_standard_vs_rotation_optimized(segments, theoretical_dim, fractal_name=""):
    """Compare standard box counting vs rotation-optimized approach."""
    print(f"⚖️  STANDARD vs ROTATION-OPTIMIZED COMPARISON")
    if fractal_name:
        print(f"Fractal: {fractal_name}")
    print("=" * 60)

    # Standard analysis (0° rotation only)
    print("\n📊 STANDARD BOX COUNTING (0° rotation):")
    standard_result = rotation_optimized_fractal_analysis(
        segments, theoretical_dim, rotation_angles=[0], verbose=True
    )

    print("\n" + "="*60)

    # Rotation-optimized analysis
    print("\n🔄 ROTATION-OPTIMIZED BOX COUNTING:")
    optimized_result = rotation_optimized_fractal_analysis(
        segments, theoretical_dim, verbose=True
    )

    # Summary comparison
    print(f"\n🎯 COMPARISON SUMMARY:")
    print("-" * 30)
    if standard_result and optimized_result:
        standard_error = standard_result['error_pct']
        optimized_error = optimized_result['error_pct']
        improvement = (standard_error - optimized_error) / standard_error * 100

        print(f"Standard error:    {standard_error:.2f}%")
        print(f"Optimized error:   {optimized_error:.2f}%")
        print(f"Improvement:       {improvement:.1f}% reduction")

        if improvement > 15:
            print("✅ SIGNIFICANT IMPROVEMENT")
        elif improvement > 5:
            print("✅ MODERATE IMPROVEMENT")
        elif improvement > 0:
            print("🔄 MINOR IMPROVEMENT")
        else:
            print("❌ NO IMPROVEMENT")

    return standard_result, optimized_result

def main():
    print("🔄 ROTATION-OPTIMIZED BOX COUNTING TEST")
    print("=" * 60)
    print("Testing if rotation angle optimization improves Dragon curve accuracy")
    print("Using pre-expanded grid strategy (√2 factor)")
    print()

    # Generate test fractals
    analyzer = FastFractalAnalyzer()

    # Test 1: Dragon Curve L4 (our problematic case)
    print("🐉 DRAGON CURVE L4 TEST")
    print("=" * 40)
    dragon_segments = analyzer._generate_dragon_curve(4)
    dragon_standard, dragon_optimized = compare_standard_vs_rotation_optimized(
        dragon_segments, 2.0, "Dragon L4"
    )

    # Test 2: Koch Curve L4 (for comparison)
    print("\n\n🌟 KOCH CURVE L4 TEST (comparison)")
    print("=" * 40)
    koch_segments = analyzer._generate_koch_curve(4)
    koch_standard, koch_optimized = compare_standard_vs_rotation_optimized(
        koch_segments, 1.2619, "Koch L4"
    )

    # Final assessment
    print(f"\n🏆 ROTATION OPTIMIZATION ASSESSMENT")
    print("=" * 50)

    if dragon_standard and dragon_optimized:
        dragon_improvement = (dragon_standard['error_pct'] - dragon_optimized['error_pct']) / dragon_standard['error_pct'] * 100
        print(f"Dragon curve improvement: {dragon_improvement:.1f}%")

    if koch_standard and koch_optimized:
        koch_improvement = (koch_standard['error_pct'] - koch_optimized['error_pct']) / koch_standard['error_pct'] * 100
        print(f"Koch curve improvement:   {koch_improvement:.1f}%")

    print(f"\n💡 Rotation optimization adds ~2x computational cost but may significantly")
    print(f"   improve accuracy for fractals with strong grid alignment artifacts.")

if __name__ == "__main__":
    main()