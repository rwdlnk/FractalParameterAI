#!/usr/bin/env python3
"""
Test AI-suggested parameters for dimensional accuracy
"""

import sys
import os
sys.path.append('.')

from core.interface_features import parse_segment_file, extract_interface_features, suggest_optimal_parameters
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def test_ai_parameters(filename, expected_dimension):
    """Test AI-suggested parameters against expected dimension."""
    print("="*70)
    print(f"TESTING AI PARAMETERS: {os.path.basename(filename)}")
    print("="*70)

    # Load segments
    segments = parse_segment_file(filename)
    print(f"Loaded {len(segments)} segments")

    # Get AI feature extraction and suggestions
    features = extract_interface_features(segments)
    ai_suggestions = suggest_optimal_parameters(features)

    print(f"\nAI Classification: {features}")
    print(f"AI Suggestions: {ai_suggestions}")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]

    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    # Add padding for analysis
    x_range = domain_x[1] - domain_x[0]
    y_range = domain_y[1] - domain_y[0]

    if y_range < 0.01:  # Handle horizontal lines
        domain_y = [domain_y[0] - 0.1, domain_y[1] + 0.1]

    print(f"\nDomain: X={domain_x}, Y={domain_y}")

    # Test AI-suggested parameters
    result = calculate_fractal_dimension(
        segments,
        domain_x,
        domain_y,
        ai_suggestions['initial_delta'],
        ai_suggestions['delta_factor'],
        ai_suggestions['num_steps']
    )

    print(f"\n🤖 AI PARAMETER RESULTS:")
    print(f"   Dimension: {result['dimension']:.8f}")
    print(f"   Expected:  {expected_dimension:.8f}")
    print(f"   Error:     {abs(result['dimension'] - expected_dimension):.8f}")
    print(f"   Error %:   {abs(result['dimension'] - expected_dimension)/expected_dimension*100:.4f}%")
    print(f"   R²:        {result['r_squared']:.10f}")
    print(f"   Valid:     {result['valid']}")

    # Test default parameters for comparison
    default_result = calculate_fractal_dimension(
        segments, domain_x, domain_y,
        initial_delta=0.5,  # Default
        delta_factor=1.5,   # Default
        num_steps=10        # Default
    )

    print(f"\n📊 DEFAULT PARAMETER RESULTS:")
    print(f"   Dimension: {default_result['dimension']:.8f}")
    print(f"   Error:     {abs(default_result['dimension'] - expected_dimension):.8f}")
    print(f"   Error %:   {abs(default_result['dimension'] - expected_dimension)/expected_dimension*100:.4f}%")
    print(f"   R²:        {default_result['r_squared']:.10f}")
    print(f"   Valid:     {default_result['valid']}")

    # Performance comparison
    ai_error = abs(result['dimension'] - expected_dimension)
    default_error = abs(default_result['dimension'] - expected_dimension)

    improvement = (default_error - ai_error) / default_error * 100 if default_error > 0 else 0

    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"   AI Error:      {ai_error:.8f}")
    print(f"   Default Error: {default_error:.8f}")
    print(f"   Improvement:   {improvement:.2f}%")

    if ai_error < default_error:
        print("   ✅ AI parameters are MORE ACCURATE!")
    else:
        print("   ❌ AI parameters are less accurate")

    return result, default_result


def main():
    print("🧠 AI PARAMETER VALIDATION TEST")
    print("Testing AI-suggested parameters vs theoretical dimensions\n")

    # Test straight line
    print("Testing straight line...")
    result1, default1 = test_ai_parameters(
        "benchmarks/data/test_straight_line.txt",
        1.0
    )

    # Test Koch curve
    print("\n" + "="*70)
    print("Testing Koch curve...")
    result2, default2 = test_ai_parameters(
        "benchmarks/data/koch_curves/koch_iteration_3.txt",
        1.2618595071  # log(4)/log(3)
    )

    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)

    if result1['valid'] and result2['valid']:
        avg_ai_accuracy = (result1['r_squared'] + result2['r_squared']) / 2
        print(f"✅ AI System Successfully Analyzed Both Cases")
        print(f"   Average R²: {avg_ai_accuracy:.8f}")
        print(f"   AI Parameter Selection: WORKING")
    else:
        print(f"❌ Some AI analyses failed")


if __name__ == "__main__":
    main()