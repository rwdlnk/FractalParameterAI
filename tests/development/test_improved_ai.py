#!/usr/bin/env python3
"""
Test Improved AI Parameter Selection

Quick test to verify that the AI now uses your successful manual parameters
instead of formula-based generation.
"""

import sys
sys.path.append('.')

from core.interface_features import suggest_optimal_parameters_adaptive, extract_interface_features
from test_multi_fractal_convergence import generate_dragon_curve
import numpy as np


def test_ai_parameter_improvement():
    """Test that AI now suggests better parameters."""

    print("🧠 TESTING IMPROVED AI PARAMETER SELECTION")
    print("Verifying that AI now uses your successful manual parameters")
    print("=" * 60)

    # Generate Dragon L3 to test parameter suggestion
    print("🐉 Generating Dragon L3 for parameter testing...")
    segments = generate_dragon_curve(3)
    if not isinstance(segments, np.ndarray):
        segments = np.array(segments)

    features = extract_interface_features(segments)

    # Debug features
    print(f"\n📊 Dragon L3 Features:")
    print(f"   tortuosity: {features.get('tortuosity', 'N/A'):.3f}")
    print(f"   complexity_score: {features.get('complexity_score', 'N/A'):.3f}")
    print(f"   direction_change_rate: {features.get('direction_change_rate', 'N/A'):.3f}")

    from core.interface_features import classify_interface_type
    interface_type = classify_interface_type(features)
    print(f"   Interface classification: {interface_type}")

    # Get AI-suggested parameters
    suggested_params = suggest_optimal_parameters_adaptive(features, "basic_box_counting")

    print(f"\n🎯 AI-SUGGESTED PARAMETERS:")
    print(f"   initial_delta: {suggested_params.get('initial_delta')}")
    print(f"   delta_factor: {suggested_params.get('delta_factor')}")
    print(f"   num_steps: {suggested_params.get('num_steps')}")
    print(f"   Source: {suggested_params.get('source', 'unknown')}")

    # Check if parameters match your successful ranges
    delta = suggested_params.get('initial_delta', 0)
    factor = suggested_params.get('delta_factor', 0)
    steps = suggested_params.get('num_steps', 0)

    print(f"\n✅ PARAMETER ASSESSMENT:")

    # Check if parameters are in your successful ranges
    delta_good = 0.5 <= delta <= 1.2  # Your range: 0.5 (straight) to 1.0 (Sierpinski)
    factor_good = 1.2 <= factor <= 1.8  # Your range: 1.2 (Sierpinski) to 1.8 (straight)
    steps_good = 12 <= steps <= 18  # Around your 15

    if delta_good:
        print(f"   ✅ initial_delta {delta} is in your successful range [0.5-1.2]")
    else:
        print(f"   ❌ initial_delta {delta} is outside your successful range [0.5-1.2]")

    if factor_good:
        print(f"   ✅ delta_factor {factor} is in your successful range [1.2-1.8]")
    else:
        print(f"   ❌ delta_factor {factor} is outside your successful range [1.2-1.8]")

    if steps_good:
        print(f"   ✅ num_steps {steps} is near your successful value (15)")
    else:
        print(f"   ❌ num_steps {steps} is far from your successful value (15)")

    overall_good = delta_good and factor_good and steps_good

    if overall_good:
        print(f"\n🎯 EXCELLENT: AI now suggests parameters in your successful ranges!")
        print(f"   This should dramatically improve Dragon performance")
    else:
        print(f"\n⚠️  IMPROVEMENT NEEDED: Some parameters still outside optimal ranges")
        print(f"   May need more learning data or parameter refinement")

    return suggested_params, overall_good


if __name__ == "__main__":
    params, success = test_ai_parameter_improvement()

    if success:
        print(f"\n🚀 READY FOR SYSTEMATIC SEARCH")
        print(f"AI parameter selection has been improved with your empirical knowledge")
        print(f"Run systematic_parameter_search.py to build comprehensive learning database")
    else:
        print(f"\n🔧 NEEDS MORE WORK")
        print(f"AI parameter selection may need additional refinement")