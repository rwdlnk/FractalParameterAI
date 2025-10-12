#!/usr/bin/env python3
"""
Simple Box Counting Grid Sensitivity Test
Direct test of your hypothesis that Dragon curves create grid alignment artifacts.
This bypasses full fractal analysis and focuses on the core box counting behavior.
"""

import sys
import time
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.box_counting import VectorizedBoxCounter

def test_box_counting_grid_sensitivity():
    """Test Dragon vs Koch curves for grid alignment sensitivity."""
    print("🐉 DRAGON vs KOCH: BOX COUNTING GRID SENSITIVITY")
    print("=" * 60)
    print("Direct test of grid alignment hypothesis using pure box counting")
    print()

    # Generate test fractals
    analyzer = FastFractalAnalyzer()
    dragon_segments = analyzer._generate_dragon_curve(4)
    koch_segments = analyzer._generate_koch_curve(4)

    print(f"Dragon L4: {dragon_segments.n_segments:,} segments")
    print(f"Koch L4: {koch_segments.n_segments:,} segments")
    print()

    # Initialize box counter
    box_counter = VectorizedBoxCounter()

    # Test different box sizes
    dragon_bbox = dragon_segments.bbox
    koch_bbox = koch_segments.bbox

    # Test at multiple scales relative to each fractal's size
    dragon_scales = [dragon_bbox.width / s for s in [8, 12, 16, 20, 24, 32]]
    koch_scales = [koch_bbox.width / s for s in [8, 12, 16, 20, 24, 32]]

    print("🔍 GRID OFFSET SENSITIVITY AT MULTIPLE SCALES")
    print("-" * 55)

    results = {'dragon': [], 'koch': []}

    for i, (dragon_size, koch_size) in enumerate(zip(dragon_scales, koch_scales)):
        scale_name = [8, 12, 16, 20, 24, 32][i]

        print(f"\nScale 1/{scale_name} (Dragon: {dragon_size:.4f}, Koch: {koch_size:.4f})")
        print("Fractal | 0% offset | 50% offset | Variation (%)")
        print("-" * 50)

        # Test Dragon curve
        try:
            dragon_count_0 = box_counter.count_boxes(dragon_segments, dragon_size, 0, 0)
            dragon_count_50 = box_counter.count_boxes(dragon_segments, dragon_size,
                                                     dragon_size * 0.5, dragon_size * 0.5)
            dragon_variation = abs(dragon_count_50 - dragon_count_0) / dragon_count_0 * 100

            print(f"Dragon  | {dragon_count_0:9d} | {dragon_count_50:10d} | {dragon_variation:9.1f}")

            results['dragon'].append({
                'scale': scale_name,
                'box_size': dragon_size,
                'count_0': dragon_count_0,
                'count_50': dragon_count_50,
                'variation': dragon_variation
            })
        except Exception as e:
            print(f"Dragon  | {'ERROR':>9} | {'ERROR':>10} | {'---':>9}")

        # Test Koch curve
        try:
            koch_count_0 = box_counter.count_boxes(koch_segments, koch_size, 0, 0)
            koch_count_50 = box_counter.count_boxes(koch_segments, koch_size,
                                                   koch_size * 0.5, koch_size * 0.5)
            koch_variation = abs(koch_count_50 - koch_count_0) / koch_count_0 * 100

            print(f"Koch    | {koch_count_0:9d} | {koch_count_50:10d} | {koch_variation:9.1f}")

            results['koch'].append({
                'scale': scale_name,
                'box_size': koch_size,
                'count_0': koch_count_0,
                'count_50': koch_count_50,
                'variation': koch_variation
            })
        except Exception as e:
            print(f"Koch    | {'ERROR':>9} | {'ERROR':>10} | {'---':>9}")

    # Analysis
    print(f"\n📊 GRID SENSITIVITY ANALYSIS")
    print("-" * 35)

    dragon_variations = [r['variation'] for r in results['dragon'] if r['variation'] < 100]
    koch_variations = [r['variation'] for r in results['koch'] if r['variation'] < 100]

    if dragon_variations and koch_variations:
        dragon_mean = np.mean(dragon_variations)
        dragon_max = np.max(dragon_variations)
        koch_mean = np.mean(koch_variations)
        koch_max = np.max(koch_variations)

        print(f"Dragon curve grid sensitivity:")
        print(f"  Mean variation: {dragon_mean:.1f}%")
        print(f"  Max variation: {dragon_max:.1f}%")
        print(f"  Sensitivity level: {'HIGH' if dragon_mean > 10 else 'MODERATE' if dragon_mean > 5 else 'LOW'}")

        print(f"\nKoch curve grid sensitivity:")
        print(f"  Mean variation: {koch_mean:.1f}%")
        print(f"  Max variation: {koch_max:.1f}%")
        print(f"  Sensitivity level: {'HIGH' if koch_mean > 10 else 'MODERATE' if koch_mean > 5 else 'LOW'}")

        sensitivity_ratio = dragon_mean / koch_mean if koch_mean > 0 else float('inf')
        print(f"\nDragon/Koch sensitivity ratio: {sensitivity_ratio:.1f}x")

        # Hypothesis validation
        print(f"\n🎯 GRID ALIGNMENT HYPOTHESIS VALIDATION:")

        evidence_points = 0
        if dragon_mean > 10:
            print("✅ Dragon shows high grid sensitivity (>10%)")
            evidence_points += 1
        elif dragon_mean > 5:
            print("🔄 Dragon shows moderate grid sensitivity (5-10%)")
            evidence_points += 0.5

        if sensitivity_ratio > 2:
            print("✅ Dragon significantly more sensitive than Koch (>2x)")
            evidence_points += 1
        elif sensitivity_ratio > 1.5:
            print("🔄 Dragon somewhat more sensitive than Koch (1.5-2x)")
            evidence_points += 0.5

        if dragon_max > 20:
            print("✅ Dragon shows extreme sensitivity at some scales (>20%)")
            evidence_points += 1

        print(f"\nEvidence score: {evidence_points:.1f}/3.0")

        if evidence_points >= 2:
            print("🏆 HYPOTHESIS CONFIRMED: Grid alignment significantly affects Dragon curves")
            print("   Recommendation: Implement grid rotation or offset averaging")
        elif evidence_points >= 1:
            print("🔄 HYPOTHESIS PARTIALLY CONFIRMED: Some grid alignment effects")
            print("   Recommendation: Consider grid optimization techniques")
        else:
            print("❌ HYPOTHESIS NOT CONFIRMED: Grid alignment not a major factor")
            print("   Recommendation: Investigate other causes of Dragon curve inaccuracy")

        return {
            'dragon_mean_variation': dragon_mean,
            'koch_mean_variation': koch_mean,
            'sensitivity_ratio': sensitivity_ratio,
            'evidence_score': evidence_points
        }
    else:
        print("❌ Insufficient data for analysis")
        return None

def test_fine_grid_analysis():
    """Detailed analysis at a single problematic scale."""
    print(f"\n🔬 FINE-SCALE GRID ANALYSIS")
    print("-" * 35)

    analyzer = FastFractalAnalyzer()
    dragon_segments = analyzer._generate_dragon_curve(4)
    box_counter = VectorizedBoxCounter()

    # Use a scale where Dragon curves typically show problems
    test_size = dragon_segments.bbox.width / 16
    print(f"Testing box size: {test_size:.6f}")
    print(f"(≈ 16 boxes across Dragon width)")

    # Test many offset positions
    offset_fractions = np.linspace(0, 0.95, 20)
    counts = []

    print(f"\nOffset testing:")
    print("Offset (%) | Box Count | Variation from first (%)")
    print("-" * 50)

    for i, offset_frac in enumerate(offset_fractions):
        offset_x = offset_frac * test_size
        offset_y = offset_frac * test_size

        try:
            count = box_counter.count_boxes(dragon_segments, test_size, offset_x, offset_y)
            counts.append(count)

            variation = (count - counts[0]) / counts[0] * 100 if counts[0] > 0 else 0
            print(f"{offset_frac*100:8.1f}   | {count:9d} | {variation:18.1f}")

        except Exception as e:
            print(f"{offset_frac*100:8.1f}   | {'ERROR':>9} | {'---':>18}")

    if counts:
        min_count = min(counts)
        max_count = max(counts)
        mean_count = np.mean(counts)
        total_variation = (max_count - min_count) / mean_count * 100

        print(f"\nFine-scale analysis results:")
        print(f"  Count range: {min_count} to {max_count}")
        print(f"  Mean count: {mean_count:.1f}")
        print(f"  Total variation: {total_variation:.1f}%")

        if total_variation > 25:
            print("  🚨 EXTREME fine-scale sensitivity!")
        elif total_variation > 15:
            print("  ⚠️  HIGH fine-scale sensitivity")
        elif total_variation > 8:
            print("  🔄 MODERATE fine-scale sensitivity")
        else:
            print("  ✅ LOW fine-scale sensitivity")

        return total_variation
    else:
        print("❌ No valid counts obtained")
        return 0

def main():
    print("🐉 SIMPLE BOX COUNTING GRID SENSITIVITY TEST")
    print("=" * 60)
    print("Testing your hypothesis: Dragon curves create grid alignment artifacts")
    print("Focus: Direct box counting behavior without full fractal analysis")
    print()

    # Test 1: Multi-scale comparison
    comparison_results = test_box_counting_grid_sensitivity()

    # Test 2: Fine-scale analysis
    fine_scale_variation = test_fine_grid_analysis()

    # Overall conclusion
    print(f"\n🏆 OVERALL ASSESSMENT")
    print("=" * 30)

    if comparison_results:
        evidence_total = comparison_results['evidence_score']
        if fine_scale_variation > 15:
            evidence_total += 0.5

        print(f"Grid sensitivity evidence: {evidence_total:.1f}/3.5")
        print(f"Dragon mean variation: {comparison_results['dragon_mean_variation']:.1f}%")
        print(f"Koch mean variation: {comparison_results['koch_mean_variation']:.1f}%")
        print(f"Sensitivity ratio: {comparison_results['sensitivity_ratio']:.1f}x")
        print(f"Fine-scale variation: {fine_scale_variation:.1f}%")

        if evidence_total >= 2.5:
            print("\n✅ STRONG EVIDENCE for grid alignment hypothesis")
            print("Grid rotation or offset averaging recommended for Dragon curves")
        elif evidence_total >= 1.5:
            print("\n🔄 MODERATE EVIDENCE for grid alignment hypothesis")
            print("Grid optimization techniques may help Dragon curve accuracy")
        else:
            print("\n❌ WEAK EVIDENCE for grid alignment hypothesis")
            print("Grid alignment not the primary cause of Dragon curve issues")

if __name__ == "__main__":
    main()