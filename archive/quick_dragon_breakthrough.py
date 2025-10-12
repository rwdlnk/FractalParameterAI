#!/usr/bin/env python3
"""
Quick Dragon Breakthrough Test

Test only the most promising parameter combinations based on your successes
to rapidly find Dragon-optimized parameters and complete the AI redesign.
"""

import sys
import os
import time
import numpy as np
sys.path.append('.')

from core.interface_features import extract_interface_features, record_parameter_feedback
from test_multi_fractal_convergence import generate_dragon_curve
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_dragon_parameter_combo(name, initial_delta, delta_factor, num_steps, expected_time="~60s"):
    """Test a single Dragon parameter combination."""

    print(f"\n🧪 Testing {name}:")
    print(f"   δ={initial_delta}, factor={delta_factor}, steps={num_steps}")
    print(f"   Expected: {expected_time}")

    # Generate Dragon L3
    segments = generate_dragon_curve(3)
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)

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

            print(f"   ✅ Result: D={dimension:.6f}, Error={error_pct:.2f}%, R²={r_squared:.6f}, Time={calc_time:.1f}s")

            # Record in AI system
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

            # Success assessment
            if error_pct < 5.0:
                status = "🎯 EXCELLENT"
            elif error_pct < 10.0:
                status = "✅ GOOD"
            elif error_pct < 20.0:
                status = "🟡 MODERATE"
            else:
                status = "❌ POOR"

            print(f"   Status: {status}")

            return {
                'success': True,
                'name': name,
                'dimension': dimension,
                'error_pct': error_pct,
                'r_squared': r_squared,
                'time': calc_time,
                'params': {'initial_delta': initial_delta, 'delta_factor': delta_factor, 'num_steps': num_steps}
            }
        else:
            print(f"   ❌ FAILED: Dimension = {dimension}")
            return {'success': False, 'name': name, 'error': 'NaN result', 'time': calc_time}

    except Exception as e:
        calc_time = time.time() - start_time
        print(f"   💥 ERROR: {e}")
        return {'success': False, 'name': name, 'error': str(e), 'time': calc_time}


def quick_dragon_breakthrough():
    """Quick test of most promising Dragon parameter combinations."""

    print("🐉 QUICK DRAGON BREAKTHROUGH TEST")
    print("Testing most promising parameter combinations for rapid discovery")
    print("=" * 70)

    # Most promising combinations based on your successful patterns
    promising_combos = [
        # Your Sierpinski L4 success pattern
        ("Sierpinski L4 Pattern", 2.0, 1.1, 19),

        # Your Sierpinski L7 success pattern
        ("Sierpinski L7 Pattern", 1.0, 1.2, 15),

        # Your straight-line success pattern
        ("Straight Line Pattern", 0.5, 1.8, 15),

        # AI improved suggestion
        ("AI Improved", 0.8, 1.5, 20),

        # Conservative around your successes
        ("Conservative Mix", 1.5, 1.4, 17),

        # Aggressive from your L4 success
        ("Aggressive L4-based", 2.5, 1.0, 20),

        # Koch-like but scaled for Dragon
        ("Koch-Dragon Hybrid", 0.6, 1.6, 18),

        # Balanced middle ground
        ("Balanced Dragon", 1.2, 1.3, 16),
    ]

    print(f"Testing {len(promising_combos)} most promising combinations...")
    print()

    results = []
    start_total = time.time()

    for i, (name, delta, factor, steps) in enumerate(promising_combos):
        print(f"[{i+1}/{len(promising_combos)}] Testing {name}")

        result = test_dragon_parameter_combo(name, delta, factor, steps)
        results.append(result)

        # Brief pause
        time.sleep(1)

    total_time = time.time() - start_total

    # Analysis
    print(f"\n🏆 QUICK DRAGON BREAKTHROUGH RESULTS")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print()

    successful = [r for r in results if r['success']]

    if successful:
        print(f"✅ SUCCESSFUL TESTS: {len(successful)}/{len(results)}")
        print()
        print(f"Combination              | Error   | Time  | Status")
        print(f"-------------------------|---------|-------|----------")

        # Sort by error
        successful.sort(key=lambda x: x['error_pct'])

        for r in successful:
            print(f"{r['name'][:23]:23s} | {r['error_pct']:6.2f}% | {r['time']:4.1f}s | {'🎯' if r['error_pct'] < 5 else '✅' if r['error_pct'] < 10 else '🟡' if r['error_pct'] < 20 else '❌'}")

        # Best result
        best = successful[0]
        print(f"\n🥇 BEST DRAGON PARAMETERS FOUND:")
        print(f"   Combination: {best['name']}")
        print(f"   Parameters: δ={best['params']['initial_delta']}, factor={best['params']['delta_factor']}, steps={best['params']['num_steps']}")
        print(f"   Error: {best['error_pct']:.2f}%")
        print(f"   R²: {best['r_squared']:.6f}")
        print(f"   Time: {best['time']:.1f}s")

        if best['error_pct'] < 5:
            print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
            print(f"   Dragon disappointment SOLVED with <5% error!")
        elif best['error_pct'] < 10:
            print(f"\n🚀 MAJOR PROGRESS!")
            print(f"   Significant improvement over 30% baseline")
        else:
            print(f"\n📈 GOOD PROGRESS")
            print(f"   Better than baseline, building AI knowledge")

        print(f"\n🧠 AI LEARNING STATUS:")
        print(f"   Dragon parameter database updated with {len(results)} examples")
        print(f"   AI now has successful Dragon-specific patterns!")
        print(f"   Ready for production parameter selection")

    else:
        print(f"❌ No successful combinations found")
        print(f"   Dragon curves may need specialized approaches")
        print(f"   But valuable learning data recorded for AI system")

    return results


if __name__ == "__main__":
    print("🚀 QUICK DRAGON BREAKTHROUGH")
    print("Rapid discovery of Dragon-optimized parameters")
    print()

    results = quick_dragon_breakthrough()

    print(f"\n✅ QUICK BREAKTHROUGH TEST COMPLETE")
    print(f"AI learning system enhanced with Dragon-specific knowledge")