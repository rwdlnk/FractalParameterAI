#!/usr/bin/env python3
"""
Test Adaptive Parameter Learning System

Tests the feedback loop system:
1. Uses original heuristic parameters
2. Records results as feedback
3. Uses learned parameters on subsequent tests
4. Shows improvement over time
"""

import sys
import os
import time
sys.path.append('.')

from core.interface_features import (
    parse_segment_file, extract_interface_features, classify_interface_type,
    suggest_optimal_parameters_adaptive, record_parameter_feedback,
    get_learning_statistics
)
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def run_adaptive_test(filename, expected_dimension, test_name, implementation="basic_box_counting"):
    """Run a single adaptive learning test."""
    print(f"\n{'='*70}")
    print(f"🧠 ADAPTIVE LEARNING TEST: {test_name}")
    print(f"{'='*70}")
    print(f"File: {os.path.basename(filename)}")
    print(f"Expected dimension: {expected_dimension:.6f}")
    print(f"Implementation: {implementation}")

    # Load segments and extract features
    segments = parse_segment_file(filename)
    features = extract_interface_features(segments)
    interface_type = classify_interface_type(features)

    print(f"Interface type: {interface_type}")
    print(f"Segments: {len(segments)}")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Add padding for analysis
    y_range = domain_y[1] - domain_y[0]
    if y_range < 0.01:  # Handle horizontal lines
        domain_y = [domain_y[0] - 0.1, domain_y[1] + 0.1]

    # Get adaptive parameters
    start_time = time.time()
    suggested_params = suggest_optimal_parameters_adaptive(features, implementation)
    suggestion_time = time.time() - start_time

    print(f"\n🤖 Adaptive parameters:")
    print(f"   initial_delta: {suggested_params['initial_delta']:.6f}")
    print(f"   delta_factor:  {suggested_params['delta_factor']:.3f}")
    print(f"   num_steps:     {suggested_params['num_steps']}")
    print(f"   source:        {suggested_params.get('source', 'unknown')}")

    # Test the suggested parameters
    start_time = time.time()
    result = calculate_fractal_dimension(
        segments, domain_x, domain_y,
        suggested_params['initial_delta'],
        suggested_params['delta_factor'],
        suggested_params['num_steps']
    )
    calculation_time = time.time() - start_time

    # Calculate accuracy metrics
    dimension = result.get('dimension')
    r_squared = result.get('r_squared')
    valid = result.get('valid', False)

    success = False
    accuracy_score = None
    error_pct = None

    if dimension is not None and not (dimension != dimension):  # Check for NaN
        error_pct = abs(dimension - expected_dimension) / expected_dimension * 100
        accuracy_score = 1.0 - (error_pct / 100.0)
        success = error_pct < 5.0  # 5% accuracy threshold

    print(f"\n📊 Results:")
    if success:
        print(f"   ✅ Dimension: {dimension:.6f} (error: {error_pct:.2f}%)")
        print(f"   ✅ R²: {r_squared:.6f}")
        print(f"   ✅ Valid: {valid}")
    else:
        print(f"   ❌ Dimension: {dimension}")
        print(f"   ❌ R²: {r_squared}")
        print(f"   ❌ Valid: {valid}")
        if error_pct:
            print(f"   ❌ Error: {error_pct:.2f}%")

    print(f"   ⏱️  Calculation time: {calculation_time:.3f}s")

    # Record feedback for learning
    feedback_record = record_parameter_feedback(
        features, suggested_params, dimension, expected_dimension,
        r_squared, calculation_time, implementation
    )

    print(f"\n📝 Feedback recorded: {feedback_record.success}")

    return {
        'success': success,
        'dimension': dimension,
        'error_pct': error_pct,
        'r_squared': r_squared,
        'calculation_time': calculation_time,
        'feedback_record': feedback_record
    }


def main():
    print("🧠 ADAPTIVE PARAMETER LEARNING TEST SYSTEM")
    print("Testing feedback-based parameter improvement")
    print("\nThis system will:")
    print("1. Start with heuristic parameters")
    print("2. Record results as feedback")
    print("3. Learn from successes and failures")
    print("4. Improve parameter selection over time")

    # Test files and expected dimensions
    test_cases = [
        ("benchmarks/data/test_straight_line.txt", 1.0, "Straight Line"),
        ("benchmarks/data/koch_curves/koch_iteration_1.txt", 1.2618595071, "Koch L1"),
        ("benchmarks/data/koch_curves/koch_iteration_2.txt", 1.2618595071, "Koch L2"),
        ("benchmarks/data/koch_curves/koch_iteration_3.txt", 1.2618595071, "Koch L3"),
    ]

    # Run multiple learning cycles
    for cycle in range(3):
        print(f"\n\n{'='*80}")
        print(f"🔄 LEARNING CYCLE {cycle + 1}")
        print(f"{'='*80}")

        cycle_results = []

        # Test each case in this cycle
        for filename, expected_dim, test_name in test_cases:
            if not os.path.exists(filename):
                print(f"⚠️  Skipping {test_name}: file not found")
                continue

            result = run_adaptive_test(filename, expected_dim, test_name)
            cycle_results.append((test_name, result))

        # Show cycle summary
        print(f"\n📈 CYCLE {cycle + 1} SUMMARY:")
        successes = sum(1 for _, r in cycle_results if r['success'])
        total = len(cycle_results)
        print(f"   Success rate: {successes}/{total} ({successes/total*100:.1f}%)")

        if cycle_results:
            avg_error = sum(r['error_pct'] for _, r in cycle_results if r['error_pct'] is not None) / len([r for _, r in cycle_results if r['error_pct'] is not None])
            print(f"   Average error: {avg_error:.2f}%")

        # Show learning statistics
        stats = get_learning_statistics()
        print(f"\n📊 LEARNING STATUS:")
        print(f"   Total records: {stats['total_records']}")
        print(f"   Recent success rate: {stats['recent_success_rate']:.1%}")

        for impl, impl_stats in stats['by_implementation'].items():
            print(f"   {impl}: {impl_stats['successes']}/{impl_stats['total_records']} success rate")

        # Wait between cycles to show progression
        if cycle < 2:
            print(f"\n⏳ Preparing for cycle {cycle + 2}...")
            time.sleep(2)

    print(f"\n{'='*80}")
    print("🎓 LEARNING COMPLETE")
    print("The system has now collected feedback and should show improved parameter selection!")
    print("Check the feedback_data.jsonl file to see the learning history.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()