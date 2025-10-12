#!/usr/bin/env python3
"""
Quick Test: Apply Manual Success Parameters to "Failed" Fractals

Test your successful Sierpinski parameters on Dragon and Hilbert curves
to validate that the AI disappointment is fixable with proper parameter selection.

Your successful parameters:
- Sierpinski: initial_delta=1.0, delta_factor=1.2, num_steps=15 → 2.2% error
- Straight line: initial_delta=0.5, delta_factor=1.8, num_steps=15 → 0.1% error

Hypothesis: These parameters will dramatically improve Dragon (30%→<10%) and Hilbert (23%→<10%).
"""

import sys
import os
import time
sys.path.append('.')

from core.interface_features import parse_segment_file
from test_multi_fractal_convergence import generate_dragon_curve, generate_hilbert_curve
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_manual_parameters_on_fractal(fractal_name, generator_func, level, theoretical_dim,
                                     initial_delta, delta_factor, num_steps):
    """Test manual parameters on a specific fractal."""

    print(f"\n🧪 TESTING {fractal_name.upper()} L{level}")
    print(f"Parameters: δ={initial_delta}, factor={delta_factor}, steps={num_steps}")
    print("-" * 60)

    # Generate fractal
    print(f"🔧 Generating {fractal_name} L{level}...")
    start_time = time.time()
    segments = generator_func(level)
    gen_time = time.time() - start_time
    print(f"✅ Generated: {len(segments)} segments ({gen_time:.3f}s)")

    # Convert to numpy array if needed
    import numpy as np
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Run analysis
    print(f"⏱️  Running box counting analysis...")
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

            print(f"📊 RESULTS:")
            print(f"   Dimension: {dimension:.6f}")
            print(f"   Theoretical: {theoretical_dim:.6f}")
            print(f"   Error: {error_pct:.2f}%")
            print(f"   R²: {r_squared:.6f}")
            print(f"   Time: {calc_time:.1f}s")

            # Success assessment
            if error_pct < 5.0:
                print(f"   🎯 EXCELLENT: Error < 5%!")
                success_level = "EXCELLENT"
            elif error_pct < 10.0:
                print(f"   ✅ GOOD: Error < 10%!")
                success_level = "GOOD"
            elif error_pct < 20.0:
                print(f"   🟡 MODERATE: Error < 20%")
                success_level = "MODERATE"
            else:
                print(f"   ❌ POOR: Error > 20%")
                success_level = "POOR"

            return {
                'success': True,
                'dimension': dimension,
                'error_pct': error_pct,
                'r_squared': r_squared,
                'time': calc_time,
                'success_level': success_level
            }
        else:
            print(f"   ❌ FAILED: Dimension = {dimension}")
            return {'success': False, 'error': 'NaN result', 'time': calc_time}

    except Exception as e:
        calc_time = time.time() - start_time
        print(f"   💥 ERROR: {e}")
        return {'success': False, 'error': str(e), 'time': calc_time}


def main():
    print("🚀 MANUAL PARAMETER VALIDATION TEST")
    print("Testing your successful parameters on 'failed' fractals")
    print("=" * 80)

    # Test parameters based on your manual successes
    test_cases = [
        {
            'name': 'Dragon',
            'generator': generate_dragon_curve,
            'level': 3,
            'theoretical': 1.5236,
            'params': {'initial_delta': 1.0, 'delta_factor': 1.2, 'num_steps': 15},
            'baseline_error': '30.53%',
            'note': 'Using your successful Sierpinski parameters'
        },
        {
            'name': 'Dragon',
            'generator': generate_dragon_curve,
            'level': 3,
            'theoretical': 1.5236,
            'params': {'initial_delta': 0.5, 'delta_factor': 1.8, 'num_steps': 15},
            'baseline_error': '30.53%',
            'note': 'Using your successful straight-line parameters'
        },
        {
            'name': 'Hilbert',
            'generator': generate_hilbert_curve,
            'level': 3,
            'theoretical': 2.0,
            'params': {'initial_delta': 1.0, 'delta_factor': 1.2, 'num_steps': 15},
            'baseline_error': '23%',
            'note': 'Using your successful Sierpinski parameters'
        },
        {
            'name': 'Hilbert',
            'generator': generate_hilbert_curve,
            'level': 3,
            'theoretical': 2.0,
            'params': {'initial_delta': 0.5, 'delta_factor': 1.8, 'num_steps': 15},
            'baseline_error': '23%',
            'note': 'Using your successful straight-line parameters'
        }
    ]

    results = []

    for test in test_cases:
        print(f"\n{'='*80}")
        print(f"TEST: {test['name']} with {test['note']}")
        print(f"AI Baseline Error: {test['baseline_error']}")

        result = test_manual_parameters_on_fractal(
            test['name'], test['generator'], test['level'], test['theoretical'],
            test['params']['initial_delta'], test['params']['delta_factor'], test['params']['num_steps']
        )

        result['test_info'] = test
        results.append(result)

        # Brief pause between tests
        time.sleep(1)

    # Summary
    print(f"\n🏆 VALIDATION TEST SUMMARY")
    print("=" * 80)
    print(f"Testing if manual parameters can fix AI 'disappointments'")
    print()

    for i, result in enumerate(results):
        test = result['test_info']
        if result['success']:
            improvement = f"vs {test['baseline_error']} baseline"
            print(f"{test['name']} ({test['note'][:20]}...): {result['error_pct']:.2f}% error {improvement} → {result['success_level']}")
        else:
            print(f"{test['name']} ({test['note'][:20]}...): FAILED")

    print()
    successful_tests = [r for r in results if r['success'] and r['error_pct'] < 10]
    if successful_tests:
        print(f"🎯 BREAKTHROUGH: {len(successful_tests)}/{len(results)} tests achieved <10% error!")
        print(f"✅ PROOF: AI disappointment is FIXABLE with proper parameters!")
        print(f"🔧 Next step: Implement data-driven parameter learning system")
    else:
        print(f"🤔 Results mixed - may need fractal-specific parameter exploration")
        print(f"💡 But proves the approach: manual tuning → systematic improvement")


if __name__ == "__main__":
    main()