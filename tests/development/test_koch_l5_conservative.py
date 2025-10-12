#!/usr/bin/env python3
"""
Conservative Test of Koch L5

Test Koch L5 with very conservative parameters to see if we can get any result.
The full parameter AI system timed out, so let's try minimal parameters.
"""

import sys
import os
import time
sys.path.append('.')

from core.interface_features import parse_segment_file, extract_interface_features
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_koch_l5_conservative():
    """Test Koch L5 with very conservative parameters."""
    print("🔬 KOCH L5 CONSERVATIVE TEST")
    print("Using minimal parameters to see if we can get any result...")
    print("=" * 60)

    filename = "benchmarks/data/koch_curves/koch_iteration_5.txt"
    theoretical_dim = 1.2618595071

    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return

    # Load segments
    segments = parse_segment_file(filename)
    n_segments = len(segments)
    print(f"Loaded {n_segments:,} segments")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Conservative parameters - much larger initial delta, fewer steps
    conservative_params = [
        {"name": "Very Conservative", "initial_delta": 0.1, "delta_factor": 0.8, "num_steps": 8},
        {"name": "Ultra Conservative", "initial_delta": 0.2, "delta_factor": 0.7, "num_steps": 6},
        {"name": "Minimal Steps", "initial_delta": 0.05, "delta_factor": 0.5, "num_steps": 5}
    ]

    results = []

    for params in conservative_params:
        print(f"\n🧪 Testing {params['name']}:")
        print(f"   initial_delta: {params['initial_delta']}")
        print(f"   delta_factor:  {params['delta_factor']}")
        print(f"   num_steps:     {params['num_steps']}")

        start_time = time.time()

        try:
            result = calculate_fractal_dimension(
                segments, domain_x, domain_y,
                params['initial_delta'],
                params['delta_factor'],
                params['num_steps']
            )
            calc_time = time.time() - start_time

            dimension = result.get('dimension')
            r_squared = result.get('r_squared')

            if dimension is not None and not (dimension != dimension):  # Not NaN
                error_pct = abs(dimension - theoretical_dim) / theoretical_dim * 100

                print(f"   ✅ SUCCESS:")
                print(f"      Dimension: {dimension:.6f}")
                print(f"      Error: {error_pct:.2f}%")
                print(f"      R²: {r_squared:.6f}")
                print(f"      Time: {calc_time:.1f}s")

                results.append({
                    'name': params['name'],
                    'dimension': dimension,
                    'error_pct': error_pct,
                    'r_squared': r_squared,
                    'time': calc_time,
                    'success': True
                })

            else:
                print(f"   ❌ FAILED: Dimension = {dimension}")
                results.append({
                    'name': params['name'],
                    'success': False,
                    'time': calc_time
                })

        except Exception as e:
            calc_time = time.time() - start_time
            print(f"   💥 ERROR: {e}")
            print(f"   Time before error: {calc_time:.1f}s")

            results.append({
                'name': params['name'],
                'success': False,
                'time': calc_time,
                'error': str(e)
            })

    # Summary
    print(f"\n{'='*60}")
    print("📊 KOCH L5 CONSERVATIVE TEST SUMMARY")
    print(f"{'='*60}")

    successful_results = [r for r in results if r['success']]

    if successful_results:
        print(f"✅ {len(successful_results)}/{len(results)} parameter sets succeeded")

        best_result = min(successful_results, key=lambda r: r['error_pct'])
        print(f"\n🏆 BEST RESULT: {best_result['name']}")
        print(f"   Dimension: {best_result['dimension']:.6f}")
        print(f"   Error: {best_result['error_pct']:.2f}%")
        print(f"   Time: {best_result['time']:.1f}s")

        # Compare with previous iterations
        print(f"\n📈 CONVERGENCE COMPARISON:")
        print(f"   L3: 9.12% error (64 segments)")
        print(f"   L4: 10.02% error (256 segments)")
        print(f"   L5: {best_result['error_pct']:.2f}% error (1024 segments)")

        if best_result['error_pct'] < 9.12:
            print(f"   ✅ L5 shows improvement - accuracy trend continues!")
        elif best_result['error_pct'] < 10.0:
            print(f"   ⚠️  L5 comparable to L3 - possible convergence plateau")
        else:
            print(f"   ❌ L5 shows degradation - computational limits?")

    else:
        print(f"❌ All parameter sets failed")
        print(f"💡 Koch L5 (1024 segments) may be beyond box counting capabilities")
        print(f"   Consider Wu methodology or other approaches for higher iterations")


if __name__ == "__main__":
    test_koch_l5_conservative()