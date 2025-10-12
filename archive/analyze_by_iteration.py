#!/usr/bin/env python3
"""
Analyze Results by Koch Iteration Level

This script breaks down the results by Koch curve iteration level to see:
1. Which parameters work best for each iteration level
2. How accuracy changes with iteration complexity
3. Whether optimal parameters are iteration-specific
"""

import json
import numpy as np
from collections import defaultdict


def load_feedback_data(filename="feedback_data.jsonl"):
    """Load and parse feedback data."""
    records = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"❌ File {filename} not found!")
        return []
    return records


def identify_iteration_level(record):
    """Identify Koch iteration level from segment count."""
    if record['interface_type'] != 'koch_curve':
        return None

    n_segments = record['features']['n_segments']

    # Koch curve segment counts: L1=4, L2=16, L3=64, L4=256, etc.
    if n_segments == 4:
        return 1
    elif n_segments == 16:
        return 2
    elif n_segments == 64:
        return 3
    elif n_segments == 256:
        return 4
    else:
        return f"Unknown({n_segments})"


def analyze_by_iteration(records):
    """Analyze results grouped by Koch iteration level."""
    print("="*80)
    print("📊 ANALYSIS BY KOCH ITERATION LEVEL")
    print("="*80)

    # Group by iteration level
    by_iteration = defaultdict(list)

    for record in records:
        if record['interface_type'] == 'koch_curve':
            iteration = identify_iteration_level(record)
            dimension = record['dimension_result']
            theoretical = record['theoretical_dimension']

            if dimension is not None and not (dimension != dimension):  # Not NaN
                error_pct = abs(dimension - theoretical) / theoretical * 100

                by_iteration[iteration].append({
                    'dimension': dimension,
                    'theoretical': theoretical,
                    'error_pct': error_pct,
                    'r_squared': record['r_squared'],
                    'initial_delta': record['suggested_parameters']['initial_delta'],
                    'delta_factor': record['suggested_parameters']['delta_factor'],
                    'num_steps': record['suggested_parameters']['num_steps'],
                    'computation_time': record['computation_time'],
                    'n_segments': record['features']['n_segments']
                })

    # Analyze each iteration level
    for iteration in sorted(by_iteration.keys()):
        results = by_iteration[iteration]
        print(f"\n🔍 KOCH ITERATION LEVEL {iteration}:")
        print(f"   Segments: {results[0]['n_segments']}")
        print(f"   Total tests: {len(results)}")

        if results:
            errors = [r['error_pct'] for r in results]
            r_squareds = [r['r_squared'] for r in results]
            times = [r['computation_time'] for r in results]

            print(f"   📈 Accuracy:")
            print(f"      Best error: {min(errors):.2f}%")
            print(f"      Average error: {np.mean(errors):.2f}% ± {np.std(errors):.2f}%")
            print(f"      Worst error: {max(errors):.2f}%")
            print(f"      Average R²: {np.mean(r_squareds):.6f}")

            print(f"   ⏱️  Performance:")
            print(f"      Average time: {np.mean(times):.3f}s")
            print(f"      Time range: {min(times):.3f}s - {max(times):.3f}s")

            # Find best performing parameters for this iteration
            best_idx = np.argmin(errors)
            best = results[best_idx]
            print(f"   🏆 BEST PARAMETERS for L{iteration}:")
            print(f"      initial_delta: {best['initial_delta']:.6f}")
            print(f"      delta_factor:  {best['delta_factor']:.3f}")
            print(f"      num_steps:     {best['num_steps']}")
            print(f"      → Result: D = {best['dimension']:.6f} (error: {best['error_pct']:.2f}%)")

            # Show parameter distribution for this iteration
            deltas = [r['initial_delta'] for r in results]
            factors = [r['delta_factor'] for r in results]
            steps = [r['num_steps'] for r in results]

            print(f"   📊 Parameter Distribution:")
            print(f"      initial_delta: {np.mean(deltas):.6f} ± {np.std(deltas):.6f}")
            print(f"      delta_factor:  {np.mean(factors):.3f} ± {np.std(factors):.3f}")
            print(f"      num_steps:     {np.mean(steps):.1f} ± {np.std(steps):.1f}")


def compare_across_iterations(records):
    """Compare how the same parameters perform across different iterations."""
    print("\n" + "="*80)
    print("📊 PARAMETER PERFORMANCE ACROSS ITERATIONS")
    print("="*80)

    # Group by parameter combination
    param_performance = defaultdict(lambda: defaultdict(list))

    for record in records:
        if record['interface_type'] == 'koch_curve':
            iteration = identify_iteration_level(record)
            dimension = record['dimension_result']
            theoretical = record['theoretical_dimension']

            if dimension is not None and not (dimension != dimension):  # Not NaN
                error_pct = abs(dimension - theoretical) / theoretical * 100

                # Create parameter signature
                params = record['suggested_parameters']
                param_key = f"δ={params['initial_delta']:.3f}, f={params['delta_factor']:.1f}, s={params['num_steps']}"

                param_performance[param_key][iteration].append(error_pct)

    # Show how each parameter set performs across iterations
    for param_key, iterations in param_performance.items():
        print(f"\n🔧 {param_key}:")

        total_tests = sum(len(errors) for errors in iterations.values())
        print(f"   Total tests: {total_tests}")

        for iteration in sorted(iterations.keys()):
            errors = iterations[iteration]
            avg_error = np.mean(errors)
            print(f"   L{iteration}: {avg_error:.2f}% ± {np.std(errors):.2f}% ({len(errors)} tests)")


def iteration_complexity_analysis(records):
    """Analyze how error scales with iteration complexity."""
    print("\n" + "="*80)
    print("📈 COMPLEXITY vs ACCURACY ANALYSIS")
    print("="*80)

    # Collect data by iteration
    iteration_data = {}

    for record in records:
        if record['interface_type'] == 'koch_curve':
            iteration = identify_iteration_level(record)
            dimension = record['dimension_result']
            theoretical = record['theoretical_dimension']

            if dimension is not None and not (dimension != dimension):  # Not NaN
                error_pct = abs(dimension - theoretical) / theoretical * 100

                if iteration not in iteration_data:
                    iteration_data[iteration] = []
                iteration_data[iteration].append(error_pct)

    print("\n📊 Error Summary by Iteration:")
    print("   Level | Segments | Best Error | Avg Error | Complexity")
    print("   ------|----------|------------|-----------|------------")

    for iteration in sorted(iteration_data.keys()):
        errors = iteration_data[iteration]
        segments = 4 * (4 ** (iteration - 1))  # Koch curve formula

        best_error = min(errors)
        avg_error = np.mean(errors)

        # Complexity indicator
        if avg_error < 10:
            complexity = "✅ Good"
        elif avg_error < 15:
            complexity = "⚠️  Moderate"
        else:
            complexity = "❌ Poor"

        print(f"   L{iteration}    | {segments:8d} | {best_error:8.2f}% | {avg_error:7.2f}% | {complexity}")

    print(f"\n💡 Observations:")
    print(f"   • Lower iterations (L1, L2) tend to have higher errors")
    print(f"   • Higher iterations (L3+) may have better accuracy due to more segments")
    print(f"   • Box counting works better with more geometric detail")


def main():
    """Main analysis function."""
    print("🧠 KOCH ITERATION LEVEL ANALYSIS")
    print("Analyzing how parameters perform across different Koch curve complexities...\n")

    # Load data
    records = load_feedback_data()
    if not records:
        return

    koch_records = [r for r in records if r['interface_type'] == 'koch_curve']
    print(f"📂 Found {len(koch_records)} Koch curve tests")

    # Run analyses
    analyze_by_iteration(records)
    compare_across_iterations(records)
    iteration_complexity_analysis(records)

    print("\n" + "="*80)
    print("💡 KEY FINDINGS")
    print("="*80)
    print("1. 🎯 Optimal parameters are likely ITERATION-SPECIFIC")
    print("2. 📈 Lower iterations may need different parameters than higher ones")
    print("3. ⚡ Box counting accuracy improves with more geometric detail")
    print("4. 🔧 Consider adaptive parameters based on segment count")


if __name__ == "__main__":
    main()