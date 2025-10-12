#!/usr/bin/env python3
"""
Dragon Curve Parameter Exploration

The AI system correctly identified that Koch-optimized parameters (0.06% error)
are inappropriate for Dragon curves (30% error). This script systematically
explores Dragon-specific parameter strategies using existing Dragon test files.

Dragon characteristics that may require different parameters:
- More complex angular patterns than Koch
- Different scaling behavior
- Potentially needs larger initial_delta or different delta_factor
"""

import sys
import os
import time
sys.path.append('.')

from core.interface_features import parse_segment_file, extract_interface_features, record_parameter_feedback
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def generate_dragon_test_file():
    """Generate Dragon test file using built-in generator."""
    try:
        # Import the generator from test_multi_fractal_convergence
        sys.path.append('.')
        from test_multi_fractal_convergence import generate_dragon_curve

        print("🐉 Generating Dragon D=1.5236 level 3...")
        segments = generate_dragon_curve(3)

        # Create temporary file
        filename = "temp_dragon_l3.txt"
        with open(filename, 'w') as f:
            for segment in segments:
                f.write(f"{segment[0]} {segment[1]} {segment[2]} {segment[3]}\n")

        print(f"✅ Generated Dragon test file: {filename} ({len(segments)} segments)")
        return filename

    except Exception as e:
        print(f"❌ Failed to generate Dragon file: {e}")
        return None


def test_dragon_parameter_strategy(name, initial_delta, delta_factor, num_steps,
                                 expected_time="unknown"):
    """Test a specific parameter strategy on Dragon L3."""

    print(f"\n🧪 Testing {name}:")
    print(f"   initial_delta: {initial_delta}")
    print(f"   delta_factor:  {delta_factor}")
    print(f"   num_steps:     {num_steps}")
    print(f"   Expected time: {expected_time}")

    filename = generate_dragon_test_file()
    if filename is None:
        return {'success': False, 'error': 'Failed to generate Dragon test file', 'time': 0}

    segments = parse_segment_file(filename)
    theoretical_dim = 1.5236

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    start_time = time.time()

    try:
        result = calculate_fractal_dimension(
            segments, domain_x, domain_y,
            initial_delta, delta_factor, num_steps
        )
        calc_time = time.time() - start_time

        dimension = result.get('dimension')
        r_squared = result.get('r_squared')

        if dimension is not None and not (dimension != dimension):  # Not NaN
            error_pct = abs(dimension - theoretical_dim) / theoretical_dim * 100

            print(f"   ✅ SUCCESS:")
            print(f"      Dimension: {dimension:.6f}")
            print(f"      Theoretical: {theoretical_dim:.6f}")
            print(f"      Error: {error_pct:.2f}%")
            print(f"      R²: {r_squared:.6f}")
            print(f"      Time: {calc_time:.1f}s")

            # Record in AI system
            features = extract_interface_features(segments)
            ai_params = {
                'initial_delta': initial_delta,
                'delta_factor': delta_factor,
                'num_steps': num_steps
            }

            feedback_result = record_parameter_feedback(
                features, ai_params, dimension, theoretical_dim,
                r_squared, calc_time, "basic_box_counting"
            )

            return {
                'success': True,
                'dimension': dimension,
                'error_pct': error_pct,
                'r_squared': r_squared,
                'time': calc_time
            }
        else:
            print(f"   ❌ FAILED: Dimension = {dimension}")
            return {'success': False, 'time': calc_time}

    except Exception as e:
        calc_time = time.time() - start_time
        print(f"   💥 ERROR: {e}")
        return {'success': False, 'error': str(e), 'time': calc_time}


def explore_dragon_parameters():
    """Systematically explore Dragon parameter strategies."""

    print("🐉 DRAGON PARAMETER EXPLORATION")
    print("Searching for Dragon-optimized parameters...")
    print("=" * 60)

    # Test multiple strategies based on Dragon characteristics
    strategies = [
        # Strategy 1: Much larger initial_delta (Dragon may need coarser start)
        ("Large Initial Delta", 1.0, 1.8, 8, "~30-60 seconds"),

        # Strategy 2: Smaller delta steps (more conservative scaling)
        ("Conservative Scaling", 0.5, 1.3, 12, "~60-120 seconds"),

        # Strategy 3: Very coarse analysis (minimal steps)
        ("Ultra-Fast", 2.0, 2.0, 5, "~10-20 seconds"),

        # Strategy 4: Fine-grained (small steps, many points)
        ("Fine-Grained", 0.2, 1.2, 15, "~120-300 seconds"),

        # Strategy 5: Balanced approach
        ("Dragon-Balanced", 0.8, 1.5, 10, "~45-90 seconds"),

        # Strategy 6: Koch-like but scaled up
        ("Scaled Koch", 0.3, 1.8, 8, "~30-60 seconds"),
    ]

    results = []

    for strategy in strategies:
        result = test_dragon_parameter_strategy(*strategy)
        results.append((strategy[0], result))

        # Brief pause between tests
        time.sleep(2)

    # Summary
    print(f"\n🏆 DRAGON PARAMETER EXPLORATION SUMMARY:")
    print(f"=" * 60)
    print(f"Strategy              | Success | Error   | Time   | Status")
    print(f"---------------------|---------|---------|--------|--------")

    best_result = None
    best_error = float('inf')

    for strategy_name, result in results:
        if result['success']:
            error = result['error_pct']
            time_str = f"{result['time']:.1f}s"
            status = "✅ GOOD" if error < 10 else "⚠️ POOR" if error < 20 else "❌ FAIL"

            if error < best_error:
                best_error = error
                best_result = (strategy_name, result)

            print(f"{strategy_name:20s} | ✅ YES  | {error:6.2f}% | {time_str:6s} | {status}")
        else:
            time_str = f"{result['time']:.1f}s" if 'time' in result else "N/A"
            print(f"{strategy_name:20s} | ❌ NO   |    N/A  | {time_str:6s} | ❌ FAIL")

    if best_result:
        strategy_name, result = best_result
        print(f"\n🥇 BEST DRAGON STRATEGY: {strategy_name}")
        print(f"   Error: {result['error_pct']:.2f}% (vs Koch's 0.06%)")
        print(f"   Time: {result['time']:.1f}s")

        if result['error_pct'] < 5:
            print(f"   🎯 EXCELLENT: Dragon parameters found!")
        elif result['error_pct'] < 10:
            print(f"   ✅ GOOD: Significant improvement over 30% baseline")
        else:
            print(f"   ⚠️ MODERATE: Better than baseline but still challenging")
    else:
        print(f"\n❌ No successful Dragon strategies found")
        print(f"   Dragon curves may require specialized methods beyond basic box counting")

    return results


if __name__ == "__main__":
    print("🚀 DRAGON CURVE PARAMETER OPTIMIZATION")
    print("Systematic exploration of Dragon-specific parameters...\n")

    results = explore_dragon_parameters()

    print(f"\n🧠 AI LEARNING IMPACT:")
    print(f"Dragon parameter exploration results recorded in AI system")
    print(f"Future Dragon curve analysis will benefit from this learning")