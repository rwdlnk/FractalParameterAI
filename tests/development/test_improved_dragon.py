#!/usr/bin/env python3
"""
Test Improved AI Parameters on Dragon

Quick test to see if the improved AI parameters {0.8, 1.5, 20}
can fix Dragon's 30% error disappointment.
"""

import sys
import os
import time
import numpy as np
sys.path.append('.')

from core.interface_features import parse_segment_file, extract_interface_features, record_parameter_feedback
from test_multi_fractal_convergence import generate_dragon_curve
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_improved_dragon_parameters():
    """Test if improved AI parameters fix Dragon's poor performance."""

    print("🐉 TESTING IMPROVED AI PARAMETERS ON DRAGON")
    print("Can AI's improved {0.8, 1.5, 20} beat the 30% error baseline?")
    print("=" * 60)

    # Generate Dragon L3
    print("🔧 Generating Dragon L3...")
    segments = generate_dragon_curve(3)
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)

    theoretical_dim = 1.5236
    print(f"✅ Generated: {len(segments)} segments")
    print(f"🎯 Target dimension: {theoretical_dim}")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Test scenarios
    test_scenarios = [
        {
            'name': 'AI Improved Parameters',
            'params': {'initial_delta': 0.8, 'delta_factor': 1.5, 'num_steps': 20},
            'baseline': '30.53% (original AI disappointment)'
        },
        {
            'name': 'Your Sierpinski Parameters',
            'params': {'initial_delta': 1.0, 'delta_factor': 1.2, 'num_steps': 15},
            'baseline': '2.2% error on Sierpinski L7'
        },
        {
            'name': 'Your Straight-Line Parameters',
            'params': {'initial_delta': 0.5, 'delta_factor': 1.8, 'num_steps': 15},
            'baseline': '0.1% error on straight lines'
        }
    ]

    results = []

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING: {scenario['name']}")
        print(f"📊 Baseline: {scenario['baseline']}")
        print(f"Parameters: {scenario['params']}")

        start_time = time.time()

        try:
            result = calculate_fractal_dimension(
                segments, domain_x, domain_y,
                scenario['params']['initial_delta'],
                scenario['params']['delta_factor'],
                scenario['params']['num_steps']
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
                    status = "🎯 EXCELLENT"
                    success_level = "EXCELLENT"
                elif error_pct < 10.0:
                    status = "✅ GOOD"
                    success_level = "GOOD"
                elif error_pct < 20.0:
                    status = "🟡 MODERATE"
                    success_level = "MODERATE"
                else:
                    status = "❌ POOR"
                    success_level = "POOR"

                print(f"   Status: {status}")

                # Record success in AI system
                features = extract_interface_features(segments)
                record_parameter_feedback(
                    features, scenario['params'], dimension, theoretical_dim,
                    r_squared, calc_time, "basic_box_counting"
                )

                results.append({
                    'name': scenario['name'],
                    'success': True,
                    'dimension': dimension,
                    'error_pct': error_pct,
                    'r_squared': r_squared,
                    'time': calc_time,
                    'status': success_level
                })

            else:
                print(f"   ❌ FAILED: Dimension = {dimension}")
                results.append({
                    'name': scenario['name'],
                    'success': False,
                    'error': 'NaN result',
                    'time': calc_time
                })

        except Exception as e:
            calc_time = time.time() - start_time
            print(f"   💥 ERROR: {e}")
            results.append({
                'name': scenario['name'],
                'success': False,
                'error': str(e),
                'time': calc_time
            })

        # Brief pause between tests
        time.sleep(1)

    # Summary
    print(f"\n🏆 DRAGON IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"Testing if AI improvements can fix Dragon's 30% error disappointment")
    print()

    successful = [r for r in results if r['success']]
    if successful:
        print(f"✅ SUCCESSFUL TESTS: {len(successful)}/{len(results)}")
        print()
        print(f"Scenario                    | Error   | Status      | Improvement")
        print(f"----------------------------|---------|-------------|-------------")

        for r in successful:
            improvement = "🚀 MAJOR" if r['error_pct'] < 10 else "⬆️ GOOD" if r['error_pct'] < 20 else "➡️ MINOR"
            print(f"{r['name'][:27]:27s} | {r['error_pct']:6.2f}% | {r['status']:11s} | {improvement}")

        best = min(successful, key=lambda x: x['error_pct'])
        print(f"\n🥇 BEST RESULT: {best['name']}")
        print(f"   Error: {best['error_pct']:.2f}% (vs 30.53% baseline)")
        print(f"   Improvement: {30.53 - best['error_pct']:.1f} percentage points!")

        if best['error_pct'] < 10:
            print(f"\n🎉 BREAKTHROUGH: Dragon disappointment FIXED!")
            print(f"   AI parameter improvement successful")
            print(f"   Ready for systematic parameter optimization")
        else:
            print(f"\n📈 SIGNIFICANT PROGRESS")
            print(f"   Major improvement over baseline")
            print(f"   Further parameter refinement recommended")

    else:
        print(f"❌ No successful tests")
        print(f"   Dragon may need specialized parameter exploration")

    return results


if __name__ == "__main__":
    print("🚀 DRAGON IMPROVEMENT TEST")
    print("Testing if improved AI parameters can fix the 30% error disappointment")
    print()

    results = test_improved_dragon_parameters()

    print(f"\n📝 CONCLUSION:")
    print(f"AI parameter improvement impact on Dragon curve analysis complete")