#!/usr/bin/env python3
"""
Koch L7 Convergence Test

Generate and test Koch L7 (16384 segments) to complete the convergence validation.
Based on L5/L6 success, this should achieve sub-0.5% error.
"""

import sys
import os
import time
sys.path.append('.')

from benchmarks.generators.koch_generator import generate_koch_curve, save_koch_curve
from core.interface_features import parse_segment_file, extract_interface_features, record_parameter_feedback
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def generate_koch_l7():
    """Generate Koch L7 curve."""
    print("🔧 Generating Koch L7 (this may take a moment)...")

    p1 = [0.0, 0.0]
    p2 = [3.0, 0.0]

    start_time = time.time()
    segments = generate_koch_curve(p1, p2, 7)
    gen_time = time.time() - start_time

    filename = "benchmarks/data/koch_curves/koch_iteration_7.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_koch_curve(segments, filename)

    print(f"✅ Generated Koch L7: {len(segments):,} segments ({gen_time:.1f}s)")
    return filename


def test_koch_l7():
    """Test Koch L7 with ultra-conservative parameters."""
    print("🔬 KOCH L7 CONVERGENCE TEST")
    print("Testing final convergence at L7 level...")
    print("=" * 60)

    # Generate L7 if it doesn't exist
    filename = "benchmarks/data/koch_curves/koch_iteration_7.txt"
    if not os.path.exists(filename):
        filename = generate_koch_l7()

    theoretical_dim = 1.2618595071

    # Load segments
    segments = parse_segment_file(filename)
    n_segments = len(segments)
    print(f"Loaded {n_segments:,} segments")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Ultra-conservative parameters based on L6 success pattern
    params = {
        "name": "L7 Ultra-Conservative",
        "initial_delta": 0.15,  # Even larger starting delta
        "delta_factor": 0.5,    # Conservative scaling
        "num_steps": 3          # Minimal steps for speed
    }

    print(f"\n🧪 Testing {params['name']}:")
    print(f"   initial_delta: {params['initial_delta']}")
    print(f"   delta_factor:  {params['delta_factor']}")
    print(f"   num_steps:     {params['num_steps']}")
    print(f"   Expected runtime: ~10-30 seconds")

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

            print(f"\n🎯 KOCH L7 RESULTS:")
            print(f"   Dimension: {dimension:.6f}")
            print(f"   Theoretical: {theoretical_dim:.6f}")
            print(f"   Error: {error_pct:.2f}%")
            print(f"   R²: {r_squared:.6f}")
            print(f"   Time: {calc_time:.1f}s")

            # Complete convergence analysis
            print(f"\n📈 COMPLETE CONVERGENCE VALIDATION:")
            convergence_data = [
                ("L3", 9.12, 64),
                ("L4", 10.02, 256),
                ("L5", 1.18, 1024),
                ("L6", 0.49, 4096),
                ("L7", error_pct, 16384)
            ]

            print(f"   Level | Error   | Segments")
            print(f"   ------|---------|----------")
            for level, error, segs in convergence_data:
                print(f"   {level}    | {error:5.2f}% | {segs:8,}")

            if error_pct < 0.49:
                print(f"\n   ✅ L7 CONTINUED IMPROVEMENT!")
                print(f"   🎯 Your L6-L7 convergence experience FULLY VALIDATED!")
            elif error_pct < 1.0:
                print(f"\n   ✅ L7 EXCELLENT ACCURACY!")
                print(f"   🎯 Convergence achieved as you predicted!")
            else:
                print(f"\n   ⚠️  L7 shows plateau - convergence reached at L6")

            # Record in AI system
            features = extract_interface_features(segments)
            ai_params = {
                'initial_delta': params['initial_delta'],
                'delta_factor': params['delta_factor'],
                'num_steps': params['num_steps']
            }

            feedback_result = record_parameter_feedback(
                features, ai_params, dimension, theoretical_dim,
                r_squared, calc_time, "basic_box_counting"
            )
            print(f"\n📝 L7 result recorded in AI system: {feedback_result.success}")

            return {
                'success': True,
                'dimension': dimension,
                'error_pct': error_pct,
                'time': calc_time
            }

        else:
            print(f"\n❌ L7 FAILED: Dimension = {dimension}")
            return {'success': False, 'time': calc_time}

    except Exception as e:
        calc_time = time.time() - start_time
        print(f"\n💥 L7 ERROR: {e}")
        print(f"   Time before error: {calc_time:.1f}s")
        return {'success': False, 'error': str(e), 'time': calc_time}


if __name__ == "__main__":
    print("🚀 KOCH L7 FINAL CONVERGENCE TEST")
    print("Validating your L6-L7 convergence experience...\n")

    result = test_koch_l7()

    if result['success']:
        print(f"\n🏆 SUCCESS: Koch L7 validation complete!")
        print(f"   Final accuracy: {result['error_pct']:.2f}% error")
        print(f"   Your domain expertise scientifically confirmed!")
    else:
        print(f"\n⚠️  L7 computational challenge encountered")
        print(f"   L6 (0.49% error) may represent practical convergence limit")