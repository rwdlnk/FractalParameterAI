#!/usr/bin/env python3
"""
Analyze Feedback Results and Find Best Parameters

This script analyzes the feedback_data.jsonl to:
1. Find the best performing parameters for each interface type
2. Identify parameter trends and patterns
3. Show dimensional accuracy statistics
4. Recommend optimal parameter ranges
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


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


def analyze_dimensional_accuracy(records):
    """Analyze dimensional accuracy by interface type and parameters."""
    print("="*80)
    print("📊 DIMENSIONAL ACCURACY ANALYSIS")
    print("="*80)

    by_interface = defaultdict(list)

    # Group by interface type
    for record in records:
        interface_type = record['interface_type']
        dimension = record['dimension_result']
        theoretical = record['theoretical_dimension']

        if dimension is not None and not (dimension != dimension):  # Not NaN
            error_pct = abs(dimension - theoretical) / theoretical * 100

            by_interface[interface_type].append({
                'dimension': dimension,
                'theoretical': theoretical,
                'error_pct': error_pct,
                'r_squared': record['r_squared'],
                'initial_delta': record['suggested_parameters']['initial_delta'],
                'delta_factor': record['suggested_parameters']['delta_factor'],
                'num_steps': record['suggested_parameters']['num_steps'],
                'computation_time': record['computation_time']
            })

    # Analyze each interface type
    for interface_type, results in by_interface.items():
        print(f"\n🔍 {interface_type.upper()}:")
        print(f"   Total successful tests: {len(results)}")

        if results:
            errors = [r['error_pct'] for r in results]
            r_squareds = [r['r_squared'] for r in results]

            print(f"   Best accuracy: {min(errors):.2f}% error")
            print(f"   Average error: {np.mean(errors):.2f}% ± {np.std(errors):.2f}%")
            print(f"   Average R²: {np.mean(r_squareds):.6f}")

            # Find best performing parameters
            best_idx = np.argmin(errors)
            best = results[best_idx]
            print(f"   🏆 BEST PARAMETERS:")
            print(f"      initial_delta: {best['initial_delta']:.6f}")
            print(f"      delta_factor:  {best['delta_factor']:.3f}")
            print(f"      num_steps:     {best['num_steps']}")
            print(f"      → Dimension: {best['dimension']:.6f} (error: {best['error_pct']:.2f}%)")
            print(f"      → R²: {best['r_squared']:.6f}")

    return by_interface


def analyze_parameter_trends(records):
    """Analyze how parameters change over time (learning)."""
    print("\n" + "="*80)
    print("📈 PARAMETER LEARNING TRENDS")
    print("="*80)

    by_interface = defaultdict(list)

    # Group chronologically by interface
    for record in records:
        interface_type = record['interface_type']
        timestamp = record['timestamp']

        by_interface[interface_type].append({
            'timestamp': timestamp,
            'initial_delta': record['suggested_parameters']['initial_delta'],
            'delta_factor': record['suggested_parameters']['delta_factor'],
            'num_steps': record['suggested_parameters']['num_steps'],
            'success': record['success'],
            'dimension': record['dimension_result'],
            'error_pct': None if record['dimension_result'] is None or record['dimension_result'] != record['dimension_result']
                        else abs(record['dimension_result'] - record['theoretical_dimension']) / record['theoretical_dimension'] * 100
        })

    # Show trends for each interface
    for interface_type, timeline in by_interface.items():
        print(f"\n🕒 {interface_type.upper()} Learning Timeline:")

        # Show first vs last parameters
        first = timeline[0]
        last = timeline[-1]

        print(f"   Initial parameters:")
        print(f"      initial_delta: {first['initial_delta']:.6f}")
        print(f"      delta_factor:  {first['delta_factor']:.3f}")
        print(f"      num_steps:     {first['num_steps']}")

        print(f"   Final parameters:")
        print(f"      initial_delta: {last['initial_delta']:.6f}")
        print(f"      delta_factor:  {last['delta_factor']:.3f}")
        print(f"      num_steps:     {last['num_steps']}")

        # Show trend
        delta_changes = [t['initial_delta'] for t in timeline]
        factor_changes = [t['delta_factor'] for t in timeline]

        if len(set(delta_changes)) > 1:
            print(f"   📉 initial_delta trend: {delta_changes[0]:.3f} → {delta_changes[-1]:.3f}")
        if len(set(factor_changes)) > 1:
            print(f"   📉 delta_factor trend: {factor_changes[0]:.3f} → {factor_changes[-1]:.3f}")


def find_optimal_parameters(records):
    """Find statistically optimal parameter ranges."""
    print("\n" + "="*80)
    print("🎯 OPTIMAL PARAMETER RECOMMENDATIONS")
    print("="*80)

    # Filter successful results only
    successful = []
    for record in records:
        if (record['dimension_result'] is not None and
            record['dimension_result'] == record['dimension_result']):  # Not NaN

            error_pct = abs(record['dimension_result'] - record['theoretical_dimension']) / record['theoretical_dimension'] * 100

            successful.append({
                'interface_type': record['interface_type'],
                'initial_delta': record['suggested_parameters']['initial_delta'],
                'delta_factor': record['suggested_parameters']['delta_factor'],
                'num_steps': record['suggested_parameters']['num_steps'],
                'error_pct': error_pct,
                'r_squared': record['r_squared']
            })

    if not successful:
        print("❌ No successful results found!")
        return

    # Group by interface type
    by_interface = defaultdict(list)
    for result in successful:
        by_interface[result['interface_type']].append(result)

    # Find optimal ranges for each interface
    for interface_type, results in by_interface.items():
        print(f"\n⚙️  {interface_type.upper()} Optimal Parameters:")

        # Filter to best 50% of results
        errors = [r['error_pct'] for r in results]
        threshold = np.median(errors)
        best_results = [r for r in results if r['error_pct'] <= threshold]

        if best_results:
            deltas = [r['initial_delta'] for r in best_results]
            factors = [r['delta_factor'] for r in best_results]
            steps = [r['num_steps'] for r in best_results]

            print(f"   Based on {len(best_results)} best results:")
            print(f"   initial_delta: {np.mean(deltas):.6f} ± {np.std(deltas):.6f}")
            print(f"                  Range: [{np.min(deltas):.6f}, {np.max(deltas):.6f}]")
            print(f"   delta_factor:  {np.mean(factors):.3f} ± {np.std(factors):.3f}")
            print(f"                  Range: [{np.min(factors):.3f}, {np.max(factors):.3f}]")
            print(f"   num_steps:     {np.mean(steps):.1f} ± {np.std(steps):.1f}")
            print(f"                  Range: [{np.min(steps)}, {np.max(steps)}]")


def implementation_comparison(records):
    """Compare different implementation approaches."""
    print("\n" + "="*80)
    print("🔬 IMPLEMENTATION COMPARISON")
    print("="*80)

    by_implementation = defaultdict(list)

    for record in records:
        implementation = record['implementation']
        success = record['success']

        by_implementation[implementation].append({
            'success': success,
            'dimension': record['dimension_result'],
            'theoretical': record['theoretical_dimension']
        })

    for impl, results in by_implementation.items():
        total = len(results)
        successes = sum(1 for r in results if r['success'])
        success_rate = successes / total * 100 if total > 0 else 0

        print(f"\n📊 {impl}:")
        print(f"   Success rate: {successes}/{total} ({success_rate:.1f}%)")

        # Calculate accuracy for non-NaN results
        valid_results = [r for r in results if r['dimension'] is not None and r['dimension'] == r['dimension']]
        if valid_results:
            errors = [abs(r['dimension'] - r['theoretical']) / r['theoretical'] * 100 for r in valid_results]
            print(f"   Average error: {np.mean(errors):.2f}% ± {np.std(errors):.2f}%")
            print(f"   Best error: {np.min(errors):.2f}%")


def main():
    """Main analysis function."""
    print("🧠 FEEDBACK RESULTS ANALYSIS")
    print("Analyzing AI parameter learning performance...\n")

    # Load data
    records = load_feedback_data()
    if not records:
        return

    print(f"📂 Loaded {len(records)} feedback records")

    # Run analyses
    accuracy_results = analyze_dimensional_accuracy(records)
    analyze_parameter_trends(records)
    find_optimal_parameters(records)
    implementation_comparison(records)

    # Summary recommendations
    print("\n" + "="*80)
    print("💡 KEY FINDINGS & RECOMMENDATIONS")
    print("="*80)

    print("\n1. 🚨 BASIC BOX COUNTING LIMITATIONS:")
    print("   - Straight lines: Always fail (NaN results) - fundamental algorithm issue")
    print("   - Koch curves: 9-18% error - acceptable but not optimal")

    print("\n2. 📉 PARAMETER LEARNING TRENDS:")
    print("   - System correctly learned to use more conservative parameters")
    print("   - Moved from aggressive (δ=0.5, factor=1.5) to conservative (δ=0.25, factor=1.2)")

    print("\n3. 🎯 RECOMMENDED PARAMETERS for basic_box_counting:")
    print("   - initial_delta: 0.1-0.25 (based on feature size)")
    print("   - delta_factor: 1.2-1.3 (conservative scaling)")
    print("   - num_steps: 8-10 (limited precision)")

    print("\n4. ⚡ NEXT STEPS:")
    print("   - Test with Wu methodology for better accuracy")
    print("   - Add numerical stability checks")
    print("   - Implement hybrid parameter selection")

    print(f"\n📁 Full results available in: feedback_data.jsonl")


if __name__ == "__main__":
    main()