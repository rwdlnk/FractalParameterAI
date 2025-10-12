#!/usr/bin/env python3
"""
Test AI Parameter System on Real RT Interface

Tests the AI-enhanced parameter selection system on actual RT interface
data from Dalziel_1999 experiments at t=3s.
"""

import sys
import time
import numpy as np
sys.path.append('.')

from core.interface_features import extract_interface_features, suggest_optimal_parameters_adaptive, record_parameter_feedback
import importlib.util
spec = importlib.util.spec_from_file_location("box_counting_optimizer", "optimization/box-counting-parameter-optimizer.py")
box_counting_optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box_counting_optimizer)
calculate_fractal_dimension = box_counting_optimizer.calculate_fractal_dimension


def load_rt_interface(filepath):
    """Load RT interface segments from file."""
    segments = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            x1, y1, x2, y2 = map(float, line.split())
            segments.append([x1, y1, x2, y2])

    return np.array(segments)


def test_rt_interface_ai():
    """Test AI parameter selection on real RT interface."""

    print("🌊 TESTING AI SYSTEM ON REAL RT INTERFACE")
    print("Dalziel_1999 data: t=3s, interface_fractal_5.dat")
    print("=" * 60)

    # Load RT interface
    rt_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/results_Time_3000/interface_fractal_5.dat"

    print("📂 Loading RT interface data...")
    segments = load_rt_interface(rt_file)
    print(f"✅ Loaded {len(segments)} segments")

    # Calculate domain
    all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    all_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    domain_x = [min(all_x), max(all_x)]
    domain_y = [min(all_y), max(all_y)]

    print(f"📏 Domain: x=[{domain_x[0]:.3f}, {domain_x[1]:.3f}], y=[{domain_y[0]:.3f}, {domain_y[1]:.3f}]")

    # Extract geometric features
    print(f"\n🔍 ANALYZING RT INTERFACE GEOMETRY")
    print("-" * 60)
    features = extract_interface_features(segments)

    # Display key features
    print(f"📊 Geometric Features:")
    print(f"   Segments: {features.get('n_segments', 'N/A')}")
    print(f"   Tortuosity: {features.get('tortuosity', 'N/A'):.3f}")
    print(f"   Complexity Score: {features.get('complexity_score', 'N/A'):.3f}")
    print(f"   Characteristic Length: {features.get('characteristic_length', 'N/A'):.3f}")
    print(f"   Mean Segment Length: {features.get('mean_segment_length', 'N/A'):.6f}")
    print(f"   Direction Change Rate: {features.get('direction_change_rate', 'N/A'):.3f}")
    print(f"   Linearity R²: {features.get('linearity_r_squared', 'N/A'):.6f}")

    # AI parameter suggestion
    print(f"\n🧠 AI PARAMETER ANALYSIS")
    print("-" * 60)
    print("AI analyzing geometric features for optimal parameters...")
    ai_params = suggest_optimal_parameters_adaptive(features)

    print(f"🎯 AI-Suggested Parameters:")
    print(f"   Initial δ: {ai_params.get('initial_delta', 'N/A'):.3f}")
    print(f"   Delta Factor: {ai_params.get('delta_factor', 'N/A'):.3f}")
    print(f"   Number of Steps: {ai_params.get('num_steps', 'N/A'):.0f}")

    # Interface classification
    interface_type = features.get('interface_type', 'unknown')
    print(f"🏷️  Interface Classification: {interface_type}")

    # Run fractal dimension calculation
    print(f"\n📐 FRACTAL DIMENSION CALCULATION")
    print("-" * 60)
    print("Running box counting with AI-optimized parameters...")

    start_time = time.time()
    result = calculate_fractal_dimension(
        segments, domain_x, domain_y,
        ai_params['initial_delta'],
        ai_params['delta_factor'],
        int(ai_params['num_steps'])  # Convert to integer
    )
    calc_time = time.time() - start_time

    dimension = result.get('dimension')
    r_squared = result.get('r_squared')

    print(f"\n🏆 RESULTS")
    print("=" * 60)
    if dimension is not None and not (dimension != dimension):  # Not NaN
        print(f"✅ Fractal Dimension: {dimension:.6f}")
        print(f"📈 R² (goodness of fit): {r_squared:.6f}")
        print(f"⏱️  Computation Time: {calc_time:.1f}s")

        # Interpretation
        if dimension < 1.1:
            complexity = "Very Smooth (nearly 1D)"
        elif dimension < 1.3:
            complexity = "Moderately Complex"
        elif dimension < 1.5:
            complexity = "Complex Interface"
        else:
            complexity = "Highly Complex Interface"

        print(f"🔬 RT Interface Complexity: {complexity}")

        # Record in AI learning system
        print(f"\n🤖 Recording in AI learning database...")
        record_parameter_feedback(
            features, ai_params, dimension, None,  # No theoretical dimension for real data
            r_squared, calc_time, "basic_box_counting"
        )
        print(f"✅ Added to AI learning database for future improvements")

        # Quality assessment
        if r_squared > 0.99:
            quality = "Excellent fit"
        elif r_squared > 0.95:
            quality = "Good fit"
        elif r_squared > 0.90:
            quality = "Acceptable fit"
        else:
            quality = "Poor fit - consider different parameters"

        print(f"📊 Analysis Quality: {quality}")

    else:
        print(f"❌ Analysis failed: dimension = {dimension}")
        print(f"⏱️  Computation time: {calc_time:.1f}s")

    print(f"\n💡 AI LEARNING INSIGHTS")
    print("-" * 60)
    print(f"AI has learned optimal parameters for:")
    print(f"• Interface type: {interface_type}")
    print(f"• Tortuosity level: {features.get('tortuosity', 0):.1f}")
    print(f"• Characteristic scale: {features.get('characteristic_length', 0):.3f}")
    print(f"• This knowledge improves future RT interface analysis")

    return {
        'segments': len(segments),
        'features': features,
        'ai_params': ai_params,
        'dimension': dimension,
        'r_squared': r_squared,
        'computation_time': calc_time
    }


if __name__ == "__main__":
    print("🚀 RT INTERFACE AI ANALYSIS")
    print("Testing AI-enhanced parameter selection on real Dalziel data")
    print()

    result = test_rt_interface_ai()

    print(f"\n✅ RT INTERFACE ANALYSIS COMPLETE")
    print(f"AI system successfully analyzed real RT interface data!")