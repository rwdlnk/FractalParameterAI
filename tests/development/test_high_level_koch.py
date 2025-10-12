#!/usr/bin/env python3
"""
Test High-Level Koch Curves (L5, L6) for Convergence

Based on user experience that Koch requires L6-L7 for convergence,
let's test the generated L5 and L6 curves to see if accuracy improves.
"""

import sys
import os
import time
sys.path.append('.')

from core.interface_features import (
    parse_segment_file, extract_interface_features, classify_interface_type,
    suggest_optimal_parameters_adaptive, record_parameter_feedback
)
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_koch_convergence():
    """Test Koch L5 and L6 for convergence behavior."""
    print("🔬 KOCH CONVERGENCE TESTING")
    print("Testing L5 and L6 based on user experience of L6-L7 convergence")
    print("="*70)

    theoretical_dim = 1.2618595071
    results = []

    # Test levels L5 and L6
    test_levels = [
        (5, "benchmarks/data/koch_curves/koch_iteration_5.txt"),
        (6, "benchmarks/data/koch_curves/koch_iteration_6.txt")
    ]

    for level, filename in test_levels:
        print(f"\n🧪 TESTING KOCH LEVEL {level}")
        print("="*50)

        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue

        # Load and analyze
        segments = parse_segment_file(filename)
        n_segments = len(segments)

        print(f"Segments: {n_segments:,}")
        print(f"Expected segments: {4 * (4 ** (level - 1)):,}")

        # Extract features and get AI parameters
        features = extract_interface_features(segments)
        suggested_params = suggest_optimal_parameters_adaptive(features, "basic_box_counting")

        print(f"AI suggested parameters:")
        print(f"   initial_delta: {suggested_params['initial_delta']:.6f}")
        print(f"   delta_factor:  {suggested_params['delta_factor']:.3f}")
        print(f"   num_steps:     {suggested_params['num_steps']}")

        # Calculate domain
        all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
        all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
        domain_x = [min(all_x), max(all_x)]
        domain_y = [min(all_y), max(all_y)]

        # Run analysis with timeout
        print(f"⏱️  Running dimensional analysis (this may take a while)...")
        start_time = time.time()

        try:
            result = calculate_fractal_dimension(
                segments, domain_x, domain_y,
                suggested_params['initial_delta'],
                suggested_params['delta_factor'],
                suggested_params['num_steps']
            )
            calc_time = time.time() - start_time

            dimension = result.get('dimension')
            r_squared = result.get('r_squared')
            valid = result.get('valid', False)

            if dimension is not None and not (dimension != dimension):  # Not NaN
                error_pct = abs(dimension - theoretical_dim) / theoretical_dim * 100

                print(f"✅ RESULTS:")
                print(f"   Dimension: {dimension:.6f}")
                print(f"   Theoretical: {theoretical_dim:.6f}")
                print(f"   Error: {error_pct:.2f}%")
                print(f"   R²: {r_squared:.6f}")
                print(f"   Time: {calc_time:.1f}s")
                print(f"   Valid: {valid}")

                # Quality assessment
                if error_pct < 2.0:
                    print(f"   🎯 EXCELLENT CONVERGENCE!")
                elif error_pct < 5.0:
                    print(f"   ✅ GOOD ACCURACY")
                elif error_pct < 10.0:
                    print(f"   ⚠️  MODERATE ACCURACY")
                else:
                    print(f"   ❌ POOR ACCURACY")

                results.append({
                    'level': level,
                    'segments': n_segments,
                    'dimension': dimension,
                    'error_pct': error_pct,
                    'r_squared': r_squared,
                    'time': calc_time,
                    'success': True
                })

                # Record feedback
                feedback_record = record_parameter_feedback(
                    features, suggested_params, dimension, theoretical_dim,
                    r_squared, calc_time, "basic_box_counting"
                )
                print(f"   📝 Feedback recorded: {feedback_record.success}")

            else:
                print(f"❌ FAILED: Dimension = {dimension}")
                results.append({
                    'level': level,
                    'segments': n_segments,
                    'dimension': None,
                    'error_pct': None,
                    'r_squared': r_squared,
                    'time': calc_time,
                    'success': False
                })

        except Exception as e:
            calc_time = time.time() - start_time
            print(f"💥 ERROR: {e}")
            print(f"   Time before error: {calc_time:.1f}s")

            results.append({
                'level': level,
                'segments': n_segments,
                'dimension': None,
                'error_pct': None,
                'r_squared': None,
                'time': calc_time,
                'success': False
            })

    # Analysis summary
    print(f"\n{'='*70}")
    print("📊 CONVERGENCE ANALYSIS SUMMARY")
    print(f"{'='*70}")

    # Include previous results for comparison
    previous_results = [
        {'level': 1, 'segments': 4, 'error_pct': 17.18, 'time': 0.2},
        {'level': 2, 'segments': 16, 'error_pct': 16.78, 'time': 0.2},
        {'level': 3, 'segments': 64, 'error_pct': 9.12, 'time': 4.2},
        {'level': 4, 'segments': 256, 'error_pct': 10.02, 'time': 159.5}
    ]

    # Combine all results
    all_results = previous_results + [r for r in results if r['success']]

    print("Level | Segments    | Error   | Time     | Convergence")
    print("------|-------------|---------|----------|-------------")

    for result in all_results:
        level = result['level']
        segments = result['segments']
        error = result.get('error_pct', 'N/A')
        time_s = result['time']

        if error != 'N/A':
            if error < 3.0:
                convergence = "🎯 Excellent"
            elif error < 5.0:
                convergence = "✅ Good"
            elif error < 10.0:
                convergence = "⚠️  Moderate"
            else:
                convergence = "❌ Poor"

            print(f"L{level}    | {segments:11,} | {error:5.2f}% | {time_s:6.1f}s | {convergence}")
        else:
            print(f"L{level}    | {segments:11,} | {'Failed':>5} | {time_s:6.1f}s | ❌ Failed")

    # Convergence trend analysis
    valid_results = [r for r in all_results if 'error_pct' in r and r['error_pct'] is not None]

    if len(valid_results) >= 3:
        print(f"\n📈 CONVERGENCE TREND:")

        # Check if we're approaching theoretical dimension
        errors = [r['error_pct'] for r in valid_results]
        levels = [r['level'] for r in valid_results]

        best_error = min(errors)
        best_level = levels[errors.index(best_error)]

        print(f"   Best accuracy: L{best_level} with {best_error:.2f}% error")

        # Check for convergence pattern
        if len(errors) >= 4:
            recent_errors = errors[-3:]  # Last 3 results
            if max(recent_errors) - min(recent_errors) < 2.0:
                print(f"   ✅ Convergence detected: errors within 2% range")
            else:
                print(f"   ⚠️  Still converging: {recent_errors[-3]:.1f}% → {recent_errors[-1]:.1f}%")

        # Time scaling analysis
        times = [r['time'] for r in valid_results]
        if len(times) >= 2:
            time_ratio = times[-1] / times[-2] if times[-2] > 0 else float('inf')
            print(f"   ⏱️  Time scaling: {time_ratio:.1f}x per level")

    print(f"\n💡 OBSERVATIONS:")
    print(f"   • User experience suggests L6-L7 needed for convergence")

    if valid_results:
        if best_error < 5.0:
            print(f"   • We achieved {best_error:.2f}% error - approaching convergence")
        else:
            print(f"   • Best error {best_error:.2f}% - more iterations may be needed")

    print(f"   • Box counting computational cost grows exponentially")
    print(f"   • Consider Wu methodology for better accuracy at lower levels")


if __name__ == "__main__":
    test_koch_convergence()