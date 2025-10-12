#!/usr/bin/env python3
"""
Test Dragon Curve AI Enhancement
Verifies that enhanced classification and specialized parameters improve Dragon curve accuracy.
"""

import sys
import time
import numpy as np
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from core.interface_features import (
    extract_interface_features,
    classify_interface_type,
    suggest_optimal_parameters
)

def test_dragon_classification_enhancement():
    """Test that Dragon curves are now properly classified as extreme_turning_fractal."""
    print("🔬 Testing Enhanced Dragon Curve Classification")
    print("=" * 55)

    analyzer = FastFractalAnalyzer()

    test_levels = [3, 4, 5]
    results = []

    for level in test_levels:
        print(f"\n🐉 Dragon L{level} Classification Test:")

        # Generate Dragon curve
        segments = analyzer._generate_dragon_curve(level)

        # Convert to interface features format
        segments_array = np.column_stack([
            segments.segments[:, 0, 0],  # x1
            segments.segments[:, 0, 1],  # y1
            segments.segments[:, 1, 0],  # x2
            segments.segments[:, 1, 1]   # y2
        ])

        # Extract features and classify
        features = extract_interface_features(segments_array)
        interface_type = classify_interface_type(features)
        suggested_params = suggest_optimal_parameters(features)

        print(f"  Segments: {segments.n_segments:,}")
        print(f"  Direction change rate: {features['direction_change_rate']:.3f}")
        print(f"  Tortuosity: {features['tortuosity']:.3f}")
        print(f"  Complexity score: {features['complexity_score']:.3f}")
        print(f"  Classification: {interface_type}")
        print(f"  Initial δ: {suggested_params['initial_delta']:.6f}")
        print(f"  Reason: {suggested_params['reason']}")

        # Check if properly classified
        is_correctly_classified = interface_type == 'extreme_turning_fractal'
        status = "✅ CORRECT" if is_correctly_classified else "❌ INCORRECT"
        print(f"  Status: {status}")

        results.append({
            'level': level,
            'n_segments': segments.n_segments,
            'direction_change_rate': features['direction_change_rate'],
            'tortuosity': features['tortuosity'],
            'complexity_score': features['complexity_score'],
            'interface_type': interface_type,
            'correctly_classified': is_correctly_classified,
            'initial_delta': suggested_params['initial_delta'],
            'delta_factor': suggested_params['delta_factor'],
            'num_steps': suggested_params['num_steps']
        })

    return results

def test_dragon_accuracy_improvement():
    """Test Dragon curve fractal dimension accuracy with enhanced parameters."""
    print("\n\n🎯 Testing Dragon Curve Accuracy Improvement")
    print("=" * 55)

    analyzer = FastFractalAnalyzer()
    theoretical_dim = 2.0  # Dragon curve theoretical dimension

    test_level = 4  # Focus on L4 where we found 96.8% change rate

    print(f"\n🐉 Dragon L{test_level} Accuracy Test:")

    # Generate Dragon curve
    segments = analyzer._generate_dragon_curve(test_level)

    # Convert to interface features format
    segments_array = np.column_stack([
        segments.segments[:, 0, 0],
        segments.segments[:, 0, 1],
        segments.segments[:, 1, 0],
        segments.segments[:, 1, 1]
    ])

    # Get AI-enhanced parameters
    features = extract_interface_features(segments_array)
    interface_type = classify_interface_type(features)
    ai_params = suggest_optimal_parameters(features)

    print(f"  Classification: {interface_type}")
    print(f"  AI Parameters: δ₀={ai_params['initial_delta']:.6f}, "
          f"factor={ai_params['delta_factor']}, steps={ai_params['num_steps']}")

    # Test with AI-enhanced parameters
    try:
        start_time = time.time()
        result = analyzer.analyze_mathematical_fractal(
            segments,
            initial_delta=ai_params['initial_delta'],
            delta_factor=ai_params['delta_factor'],
            num_steps=ai_params['num_steps']
        )
        analysis_time = time.time() - start_time

        if result and hasattr(result, 'dimension'):
            error_pct = abs(result.dimension - theoretical_dim) / theoretical_dim * 100

            print(f"\n📊 Enhanced Results:")
            print(f"  Dimension: {result.dimension:.6f}")
            print(f"  Theoretical: {theoretical_dim:.6f}")
            print(f"  Error: {error_pct:.2f}%")
            print(f"  R²: {result.r_squared:.6f}")
            print(f"  Time: {analysis_time:.1f}s")

            # Success criteria
            if error_pct < 15:  # Target: reduce from ~30% to <15%
                print(f"  Status: ✅ IMPROVED (error < 15%)")
            elif error_pct < 25:
                print(f"  Status: 🔄 PARTIAL IMPROVEMENT (error < 25%)")
            else:
                print(f"  Status: ❌ NEEDS FURTHER WORK (error ≥ 25%)")

            return {
                'dimension': result.dimension,
                'error_pct': error_pct,
                'r_squared': result.r_squared,
                'analysis_time': analysis_time,
                'success': error_pct < 15
            }
        else:
            print(f"  Status: ❌ ANALYSIS FAILED")
            return None

    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    print("🚀 DRAGON CURVE AI ENHANCEMENT VALIDATION")
    print("=" * 60)
    print("Testing enhanced classification and specialized parameters")
    print()

    # Test 1: Classification enhancement
    classification_results = test_dragon_classification_enhancement()

    # Test 2: Accuracy improvement
    accuracy_result = test_dragon_accuracy_improvement()

    # Summary
    print("\n\n📋 ENHANCEMENT VALIDATION SUMMARY")
    print("=" * 60)

    # Classification summary
    correct_classifications = sum(1 for r in classification_results if r['correctly_classified'])
    total_tests = len(classification_results)

    print(f"\n🔬 Classification Enhancement:")
    print(f"  Correct classifications: {correct_classifications}/{total_tests}")

    for result in classification_results:
        status = "✅" if result['correctly_classified'] else "❌"
        print(f"  Dragon L{result['level']}: {result['interface_type']} {status}")

    # Accuracy summary
    print(f"\n🎯 Accuracy Enhancement:")
    if accuracy_result:
        print(f"  Dragon L4 error: {accuracy_result['error_pct']:.2f}%")
        print(f"  Target achieved: {'✅' if accuracy_result['success'] else '❌'}")
        print(f"  Quality (R²): {accuracy_result['r_squared']:.6f}")
    else:
        print(f"  Dragon L4: ❌ Analysis failed")

    # Overall assessment
    classification_success = correct_classifications == total_tests
    accuracy_success = accuracy_result and accuracy_result['success']

    print(f"\n🏆 Overall Enhancement Status:")
    if classification_success and accuracy_success:
        print("  ✅ DRAGON CURVE AI ENHANCEMENT SUCCESSFUL")
        print("  Dragon curves now properly classified and analyzed with high accuracy!")
    elif classification_success:
        print("  🔄 PARTIAL SUCCESS - Classification enhanced, accuracy needs refinement")
    else:
        print("  ❌ ENHANCEMENT NEEDS FURTHER WORK")
        print("  Consider adjusting classification criteria or parameters")

if __name__ == "__main__":
    main()