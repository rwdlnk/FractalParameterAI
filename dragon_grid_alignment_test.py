#!/usr/bin/env python3
"""
Dragon Curve Grid Alignment Test
Simple test to validate the hypothesis that Dragon curves create grid alignment
artifacts in box counting. Uses direct box counting with different grid offsets.

Based on user insight: "dragon is a sequence of lines forming boxes or parts thereof
which might interfere with detecting box intersections from a regular grid."
"""

import sys
import time
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.box_counting import VectorizedBoxCounter

def test_dragon_grid_alignment(level=4):
    """Test Dragon curve sensitivity to grid alignment."""
    print(f"🐉 DRAGON CURVE GRID ALIGNMENT TEST - Level {level}")
    print("=" * 60)
    print("Testing hypothesis: Dragon segments create box-like patterns")
    print("that interfere with regular grid box counting")
    print()

    # Generate Dragon curve
    analyzer = FastFractalAnalyzer()
    segments = analyzer._generate_dragon_curve(level)
    theoretical_dim = 2.0  # Dragon curve theoretical dimension

    print(f"Dragon L{level}: {segments.n_segments:,} segments")
    print(f"Bbox: ({segments.bbox.min_x:.3f}, {segments.bbox.min_y:.3f}) to "
          f"({segments.bbox.max_x:.3f}, {segments.bbox.max_y:.3f})")
    print(f"Domain size: {segments.bbox.width:.3f} × {segments.bbox.height:.3f}")
    print()

    # Test 1: Basic analysis with standard parameters
    print("📊 BASELINE ANALYSIS")
    print("-" * 30)

    start_time = time.time()
    baseline_result = analyzer.analyze_segments(
        segments,
        initial_delta=0.01,
        delta_factor=1.8,
        num_steps=20
    )
    baseline_time = time.time() - start_time

    if baseline_result and hasattr(baseline_result, 'dimension'):
        baseline_error = abs(baseline_result.dimension - theoretical_dim) / theoretical_dim * 100
        print(f"Baseline dimension: {baseline_result.dimension:.6f}")
        print(f"Theoretical: {theoretical_dim:.6f}")
        print(f"Error: {baseline_error:.2f}%")
        print(f"R²: {baseline_result.r_squared:.6f}")
        print(f"Time: {baseline_time:.1f}s")
    else:
        print("❌ Baseline analysis failed")
        return

    # Test 2: Direct box counting with different grid offsets
    print(f"\n🔍 GRID OFFSET SENSITIVITY TEST")
    print("-" * 40)

    box_counter = VectorizedBoxCounter()
    test_box_size = 0.02  # Fixed box size for comparison

    # Test different grid offsets
    offset_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    offset_results = []

    print("Offset | Box Count | Variation (%)")
    print("-" * 35)

    for offset_frac in offset_fractions:
        offset_x = offset_frac * test_box_size
        offset_y = offset_frac * test_box_size

        try:
            count = box_counter.count_boxes(
                segments,
                test_box_size,
                offset_x,
                offset_y
            )
            offset_results.append({
                'offset': offset_frac,
                'count': count,
                'offset_x': offset_x,
                'offset_y': offset_y
            })

            # Calculate variation from first result
            if offset_results:
                base_count = offset_results[0]['count']
                variation = (count - base_count) / base_count * 100 if base_count > 0 else 0
                print(f"{offset_frac:6.1f} | {count:9d} | {variation:9.1f}")
            else:
                print(f"{offset_frac:6.1f} | {count:9d} | {'---':>9}")

        except Exception as e:
            print(f"{offset_frac:6.1f} | {'ERROR':>9} | {'---':>9}")

    # Analyze offset sensitivity
    if offset_results:
        counts = [r['count'] for r in offset_results]
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        variation_range = (max_count - min_count) / mean_count * 100
        cv = std_count / mean_count * 100

        print(f"\n📈 Grid Offset Analysis:")
        print(f"  Min count: {min_count:,}")
        print(f"  Max count: {max_count:,}")
        print(f"  Mean: {mean_count:.1f} ± {std_count:.1f}")
        print(f"  Variation range: {variation_range:.1f}%")
        print(f"  Coefficient of variation: {cv:.1f}%")

        if cv > 10:
            print("  ⚠️  HIGH GRID SENSITIVITY - Confirms grid alignment hypothesis!")
        elif cv > 5:
            print("  ✅ MODERATE GRID SENSITIVITY - Some alignment effects")
        else:
            print("  ❌ LOW GRID SENSITIVITY - Grid alignment not major factor")

    # Test 3: Compare with Koch curve (known to work well)
    print(f"\n⚖️  COMPARATIVE ANALYSIS: Dragon vs Koch")
    print("-" * 45)

    koch_segments = analyzer._generate_koch_curve(level)
    print(f"Koch L{level}: {koch_segments.n_segments:,} segments")

    # Test same grid offsets on Koch curve
    koch_results = []
    for offset_frac in [0.0, 0.5]:  # Just test 0% and 50% offset
        offset_x = offset_frac * test_box_size
        offset_y = offset_frac * test_box_size

        try:
            count = box_counter.count_boxes(
                koch_segments,
                test_box_size,
                offset_x,
                offset_y
            )
            koch_results.append(count)
        except:
            koch_results.append(0)

    if len(koch_results) >= 2 and koch_results[0] > 0:
        koch_variation = abs(koch_results[1] - koch_results[0]) / koch_results[0] * 100
        dragon_variation = abs(offset_results[-1]['count'] - offset_results[0]['count']) / offset_results[0]['count'] * 100

        print(f"Grid sensitivity comparison (0% vs 50% offset):")
        print(f"  Dragon L{level}: {dragon_variation:.1f}% variation")
        print(f"  Koch L{level}: {koch_variation:.1f}% variation")
        print(f"  Dragon/Koch ratio: {dragon_variation/koch_variation:.1f}x" if koch_variation > 0 else "  Dragon/Koch ratio: ∞")

        if dragon_variation > 2 * koch_variation:
            print("  ✅ DRAGON SHOWS HIGHER GRID SENSITIVITY than Koch")
        else:
            print("  ❌ Dragon and Koch show similar grid sensitivity")

    # Test 4: Fine grid analysis at problematic box size
    print(f"\n🔬 FINE-SCALE GRID ANALYSIS")
    print("-" * 35)

    # Test the box size where Dragon curves typically fail
    problematic_size = segments.bbox.width / 20  # Scale to fractal size

    print(f"Testing box size: {problematic_size:.6f}")
    print(f"(≈ {segments.bbox.width/problematic_size:.0f} boxes across width)")

    fine_offsets = np.linspace(0, 0.9, 10)
    fine_counts = []

    for offset_frac in fine_offsets:
        offset_x = offset_frac * problematic_size
        offset_y = offset_frac * problematic_size

        try:
            count = box_counter.count_boxes(
                segments,
                problematic_size,
                offset_x,
                offset_y
            )
            fine_counts.append(count)
        except:
            fine_counts.append(0)

    if fine_counts:
        fine_min = min([c for c in fine_counts if c > 0])
        fine_max = max(fine_counts)
        fine_variation = (fine_max - fine_min) / np.mean(fine_counts) * 100

        print(f"Fine-scale variation: {fine_variation:.1f}%")
        print(f"Count range: {fine_min} to {fine_max}")

        if fine_variation > 15:
            print("  🎯 SIGNIFICANT FINE-SCALE SENSITIVITY")
            print("  This supports the grid alignment hypothesis!")
        else:
            print("  ❌ Limited fine-scale sensitivity")

    return {
        'baseline_error': baseline_error,
        'grid_variation': variation_range if offset_results else 0,
        'dragon_vs_koch': dragon_variation / koch_variation if koch_variation > 0 else float('inf'),
        'fine_scale_variation': fine_variation if fine_counts else 0
    }

def main():
    print("🐉 DRAGON CURVE GRID ALIGNMENT HYPOTHESIS TEST")
    print("=" * 60)
    print("Testing if Dragon curves create systematic grid alignment artifacts")
    print("that explain their poor box counting performance (~30% error)")
    print()

    # Test Dragon L4 (our known problematic case)
    level = 4
    results = test_dragon_grid_alignment(level)

    if results:
        print(f"\n🎯 HYPOTHESIS VALIDATION SUMMARY")
        print("=" * 45)
        print(f"Dragon L{level} baseline error: {results['baseline_error']:.1f}%")
        print(f"Grid offset sensitivity: {results['grid_variation']:.1f}%")
        print(f"Dragon vs Koch sensitivity: {results['dragon_vs_koch']:.1f}x")
        print(f"Fine-scale variation: {results['fine_scale_variation']:.1f}%")

        # Overall assessment
        evidence_count = 0
        if results['grid_variation'] > 10:
            evidence_count += 1
            print("\n✅ Evidence 1: High grid offset sensitivity")
        if results['dragon_vs_koch'] > 2:
            evidence_count += 1
            print("✅ Evidence 2: Dragon more sensitive than Koch")
        if results['fine_scale_variation'] > 15:
            evidence_count += 1
            print("✅ Evidence 3: Significant fine-scale grid effects")

        print(f"\n🏆 GRID ALIGNMENT HYPOTHESIS:")
        if evidence_count >= 2:
            print("  ✅ CONFIRMED - Grid alignment significantly affects Dragon curves")
            print("  Recommendation: Implement grid rotation or multi-offset averaging")
        elif evidence_count == 1:
            print("  🔄 PARTIALLY CONFIRMED - Some grid alignment effects detected")
            print("  Recommendation: Further investigation into alternative approaches")
        else:
            print("  ❌ NOT CONFIRMED - Grid alignment not the primary issue")
            print("  Recommendation: Look for other causes of Dragon curve inaccuracy")

if __name__ == "__main__":
    main()