#!/usr/bin/env python3
"""
Test Higher Koch Iteration Levels

Generate and test Koch curves L4, L5, L6 to see:
1. If accuracy continues to improve with higher iterations
2. How computation time scales
3. What optimal parameters emerge for complex fractals
"""

import sys
import os
import time
sys.path.append('.')

from benchmarks.generators.koch_generator import generate_koch_curve, save_koch_curve
from core.interface_features import (
    parse_segment_file, extract_interface_features, classify_interface_type,
    suggest_optimal_parameters_adaptive, record_parameter_feedback
)
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def generate_higher_koch_levels():
    """Generate Koch curves L4, L5, L6."""
    print("🔧 Generating higher iteration Koch curves...")

    # Koch curve parameters
    p1 = [0.0, 0.0]
    p2 = [3.0, 0.0]
    theoretical_dim = 1.2618595071  # log(4)/log(3)

    levels = [4, 5, 6]
    generated_files = []

    for level in levels:
        print(f"   Generating Koch L{level}...")

        start_time = time.time()
        segments = generate_koch_curve(p1, p2, level)
        gen_time = time.time() - start_time

        filename = f"benchmarks/data/koch_curves/koch_iteration_{level}.txt"

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save to file
        save_koch_curve(segments, filename)

        print(f"   ✅ Koch L{level}: {len(segments)} segments ({gen_time:.3f}s generation)")
        generated_files.append((level, filename, len(segments)))

    return generated_files


def test_koch_level(level, filename, expected_segments, expected_dimension):
    """Test a single Koch iteration level."""
    print(f"\n{'='*70}")
    print(f"🧪 TESTING KOCH LEVEL {level}")
    print(f"{'='*70}")
    print(f"File: {os.path.basename(filename)}")
    print(f"Expected segments: {expected_segments}")
    print(f"Expected dimension: {expected_dimension:.6f}")

    # Load and analyze
    segments = parse_segment_file(filename)
    actual_segments = len(segments)

    if actual_segments != expected_segments:
        print(f"⚠️  Segment count mismatch: got {actual_segments}, expected {expected_segments}")

    features = extract_interface_features(segments)
    interface_type = classify_interface_type(features)

    print(f"Interface type: {interface_type}")
    print(f"Actual segments: {actual_segments}")
    print(f"Feature analysis:")
    print(f"   Mean segment length: {features['mean_segment_length']:.6f}")
    print(f"   Tortuosity: {features['tortuosity']:.3f}")
    print(f"   Complexity score: {features['complexity_score']:.3f}")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Get adaptive parameters
    suggested_params = suggest_optimal_parameters_adaptive(features, "basic_box_counting")

    print(f"\n🤖 AI Suggested Parameters:")
    print(f"   initial_delta: {suggested_params['initial_delta']:.6f}")
    print(f"   delta_factor:  {suggested_params['delta_factor']:.3f}")
    print(f"   num_steps:     {suggested_params['num_steps']}")

    # Test the parameters
    print(f"\n⏱️  Running box counting analysis...")
    start_time = time.time()

    try:
        result = calculate_fractal_dimension(
            segments, domain_x, domain_y,
            suggested_params['initial_delta'],
            suggested_params['delta_factor'],
            suggested_params['num_steps']
        )
        calculation_time = time.time() - start_time

        # Analyze results
        dimension = result.get('dimension')
        r_squared = result.get('r_squared')
        valid = result.get('valid', False)

        success = False
        error_pct = None

        if dimension is not None and not (dimension != dimension):  # Not NaN
            error_pct = abs(dimension - expected_dimension) / expected_dimension * 100
            success = error_pct < 5.0  # 5% threshold

            print(f"✅ RESULTS:")
            print(f"   Dimension: {dimension:.6f}")
            print(f"   Expected:  {expected_dimension:.6f}")
            print(f"   Error:     {error_pct:.2f}%")
            print(f"   R²:        {r_squared:.6f}")
            print(f"   Time:      {calculation_time:.3f}s")
            print(f"   Valid:     {valid}")

            if error_pct < 3.0:
                print(f"   🎯 EXCELLENT ACCURACY!")
            elif error_pct < 5.0:
                print(f"   ✅ GOOD ACCURACY")
            elif error_pct < 10.0:
                print(f"   ⚠️  MODERATE ACCURACY")
            else:
                print(f"   ❌ POOR ACCURACY")

        else:
            print(f"❌ FAILED:")
            print(f"   Dimension: {dimension}")
            print(f"   R²:        {r_squared}")
            print(f"   Time:      {calculation_time:.3f}s")

    except Exception as e:
        print(f"💥 ERROR during calculation: {e}")
        dimension = None
        r_squared = None
        calculation_time = time.time() - start_time

    # Record feedback
    feedback_record = record_parameter_feedback(
        features, suggested_params, dimension, expected_dimension,
        r_squared, calculation_time, "basic_box_counting"
    )

    print(f"📝 Feedback recorded: {feedback_record.success}")

    return {
        'level': level,
        'segments': actual_segments,
        'dimension': dimension,
        'error_pct': error_pct,
        'r_squared': r_squared,
        'computation_time': calculation_time,
        'success': success
    }


def main():
    """Main testing function."""
    print("🚀 HIGHER KOCH ITERATION LEVEL TESTING")
    print("Testing L4, L5, L6 to see if accuracy trend continues...\n")

    # Generate higher iteration Koch curves
    generated_files = generate_higher_koch_levels()

    # Test each level
    results = []
    theoretical_dim = 1.2618595071

    for level, filename, expected_segments in generated_files:
        try:
            result = test_koch_level(level, filename, expected_segments, theoretical_dim)
            results.append(result)
        except Exception as e:
            print(f"💥 Error testing L{level}: {e}")
            continue

    # Summary analysis
    print(f"\n{'='*70}")
    print("📊 HIGHER ITERATION SUMMARY")
    print(f"{'='*70}")

    print("Level | Segments    | Dimension  | Error   | Time    | Status")
    print("------|-------------|------------|---------|---------|--------")

    for result in results:
        level = result['level']
        segments = result['segments']
        dimension = result['dimension']
        error_pct = result['error_pct']
        time_s = result['computation_time']

        if dimension is not None and error_pct is not None:
            status = "✅ Good" if error_pct < 5.0 else "⚠️  Moderate" if error_pct < 10.0 else "❌ Poor"
            print(f"L{level}    | {segments:11,} | {dimension:8.6f} | {error_pct:5.2f}% | {time_s:5.1f}s | {status}")
        else:
            print(f"L{level}    | {segments:11,} | {'Failed':>8} | {'N/A':>5} | {time_s:5.1f}s | ❌ Failed")

    # Trend analysis
    valid_results = [r for r in results if r['error_pct'] is not None]

    if len(valid_results) >= 2:
        print(f"\n📈 TREND ANALYSIS:")

        # Show accuracy trend
        levels = [r['level'] for r in valid_results]
        errors = [r['error_pct'] for r in valid_results]
        times = [r['computation_time'] for r in valid_results]

        print(f"   Accuracy trend: L{levels[0]} ({errors[0]:.1f}%) → L{levels[-1]} ({errors[-1]:.1f}%)")

        if errors[-1] < errors[0]:
            improvement = errors[0] - errors[-1]
            print(f"   ✅ Improvement: {improvement:.1f}% error reduction")
        else:
            degradation = errors[-1] - errors[0]
            print(f"   ❌ Degradation: {degradation:.1f}% error increase")

        print(f"   Time scaling: L{levels[0]} ({times[0]:.1f}s) → L{levels[-1]} ({times[-1]:.1f}s)")
        time_factor = times[-1] / times[0] if times[0] > 0 else float('inf')
        print(f"   Time factor: {time_factor:.1f}x")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")

    best_result = min(valid_results, key=lambda r: r['error_pct']) if valid_results else None

    if best_result:
        print(f"   🏆 Best accuracy: L{best_result['level']} with {best_result['error_pct']:.2f}% error")

        if best_result['error_pct'] < 3.0:
            print(f"   ✅ Excellent accuracy achieved at higher iterations!")
        elif best_result['error_pct'] < 5.0:
            print(f"   ✅ Good accuracy, but still room for improvement")
        else:
            print(f"   ⚠️  Consider switching to Wu methodology for better results")

    print(f"   📁 Results saved to feedback_data.jsonl for further analysis")


if __name__ == "__main__":
    main()