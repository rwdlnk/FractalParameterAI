#!/usr/bin/env python3
"""
Systematic Parameter Search for True AI Learning

Instead of formula-based parameter selection, this script systematically explores
parameter space to build a proper learning database. We'll focus on Dragon curves
where we know manual parameters can achieve good results.

Based on your successful parameters:
- Straight lines: {0.5, 1.8, 15} → 0.1% error
- Sierpinski: {1.0, 1.2, 15} → 2.2% error

We'll explore parameter ranges around these successful values.
"""

import sys
import os
import time
import itertools
import numpy as np
sys.path.append('.')

from core.interface_features import parse_segment_file, extract_interface_features, record_parameter_feedback
from test_multi_fractal_convergence import generate_dragon_curve
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_parameter_combination(fractal_name, generator_func, level, theoretical_dim,
                             initial_delta, delta_factor, num_steps):
    """Test a specific parameter combination and record results."""

    # Generate fractal
    segments = generator_func(level)
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)

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
            success = error_pct < 10.0  # Success threshold: <10% error

            # Record in AI learning system
            features = extract_interface_features(segments)
            ai_params = {
                'initial_delta': initial_delta,
                'delta_factor': delta_factor,
                'num_steps': num_steps
            }

            record_parameter_feedback(
                features, ai_params, dimension, theoretical_dim,
                r_squared, calc_time, "basic_box_counting"
            )

            return {
                'success': True,
                'dimension': dimension,
                'error_pct': error_pct,
                'r_squared': r_squared,
                'time': calc_time,
                'ai_success': success
            }
        else:
            return {'success': False, 'error': 'NaN result', 'time': calc_time}

    except Exception as e:
        calc_time = time.time() - start_time
        return {'success': False, 'error': str(e), 'time': calc_time}


def systematic_dragon_search():
    """Systematically search parameter space for Dragon curves."""

    print("🐉 SYSTEMATIC DRAGON PARAMETER SEARCH")
    print("Building proper learning database with parameter exploration")
    print("=" * 70)

    # Parameter ranges based on your successful manual values
    initial_deltas = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]  # Around your 0.5 and 1.0 successes
    delta_factors = [1.1, 1.2, 1.3, 1.5, 1.8, 2.0]   # Around your 1.2 and 1.8 successes
    num_steps_list = [8, 10, 12, 15, 18, 20]          # Around your 15 success

    total_combinations = len(initial_deltas) * len(delta_factors) * len(num_steps_list)
    print(f"Testing {total_combinations} parameter combinations on Dragon L3")
    print(f"Estimated time: {total_combinations * 45 / 60:.1f} minutes")
    print()

    results = []
    combination_count = 0
    successful_count = 0

    for initial_delta in initial_deltas:
        for delta_factor in delta_factors:
            for num_steps in num_steps_list:
                combination_count += 1

                print(f"[{combination_count:3d}/{total_combinations}] Testing δ={initial_delta}, factor={delta_factor}, steps={num_steps}")

                result = test_parameter_combination(
                    "Dragon", generate_dragon_curve, 3, 1.5236,
                    initial_delta, delta_factor, num_steps
                )

                if result['success']:
                    error = result['error_pct']
                    time_str = f"{result['time']:.1f}s"

                    if result['ai_success']:  # Error < 10%
                        successful_count += 1
                        status = f"✅ SUCCESS ({error:.1f}%)"
                    else:
                        status = f"⚠️ POOR ({error:.1f}%)"

                    print(f"    → {status} in {time_str}")
                else:
                    print(f"    → ❌ FAILED")

                results.append({
                    'initial_delta': initial_delta,
                    'delta_factor': delta_factor,
                    'num_steps': num_steps,
                    'result': result
                })

                # Brief pause to avoid overheating
                time.sleep(0.5)

    # Analysis
    print(f"\n🏆 SYSTEMATIC SEARCH RESULTS")
    print("=" * 70)
    print(f"Total combinations tested: {total_combinations}")
    print(f"Successful combinations (<10% error): {successful_count}")
    print(f"Success rate: {successful_count/total_combinations*100:.1f}%")

    # Find best results
    successful_results = [r for r in results if r['result']['success'] and r['result']['ai_success']]

    if successful_results:
        # Sort by error
        successful_results.sort(key=lambda x: x['result']['error_pct'])

        print(f"\n🥇 TOP 5 DRAGON PARAMETER COMBINATIONS:")
        print("Rank | δ    | Factor | Steps | Error  | Time  | R²")
        print("-----|------|--------|-------|--------|-------|--------")

        for i, r in enumerate(successful_results[:5]):
            params = r
            result = r['result']
            print(f"{i+1:4d} | {params['initial_delta']:4.1f} | {params['delta_factor']:6.1f} | {params['num_steps']:5d} | {result['error_pct']:5.1f}% | {result['time']:4.1f}s | {result['r_squared']:.4f}")

        # Best parameters
        best = successful_results[0]
        print(f"\n🎯 BEST DRAGON PARAMETERS:")
        print(f"   initial_delta: {best['initial_delta']}")
        print(f"   delta_factor: {best['delta_factor']}")
        print(f"   num_steps: {best['num_steps']}")
        print(f"   Error: {best['result']['error_pct']:.2f}%")
        print(f"   Time: {best['result']['time']:.1f}s")

        print(f"\n🧠 AI LEARNING STATUS:")
        print(f"   Dragon parameter database now contains {len(results)} examples")
        print(f"   Success examples: {successful_count}")
        print(f"   AI can now learn Dragon-specific parameter patterns!")

    else:
        print(f"\n❌ No successful parameter combinations found")
        print(f"   May need to expand parameter search ranges")
        print(f"   Or Dragon L3 may need different approach")

    return results


if __name__ == "__main__":
    print("🚀 SYSTEMATIC PARAMETER EXPLORATION")
    print("Building data-driven AI parameter learning database")
    print()

    results = systematic_dragon_search()

    print(f"\n✅ Parameter exploration complete!")
    print(f"AI learning database updated with Dragon parameter knowledge")