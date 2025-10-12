#!/usr/bin/env python3
"""
Test AI Parameter System on Real RT Interface - Scale Corrected

Tests the AI system with proper computational domain context:
Domain: 0 <= x <= 0.4 m, 0 <= y <= 0.5 m
Initial interface: y = 0.25 m (straight line)
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


def test_rt_interface_with_proper_domain():
    """Test AI parameter selection on RT interface with proper domain context."""

    print("🌊 RT INTERFACE ANALYSIS - PROPER DOMAIN CONTEXT")
    print("Dalziel_1999: Computational domain 0≤x≤0.4m, 0≤y≤0.5m, t₀: y=0.25m")
    print("=" * 70)

    # Load RT interface
    rt_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/results_Time_3000/interface_fractal_5.dat"

    print("📂 Loading RT interface data...")
    segments = load_rt_interface(rt_file)
    print(f"✅ Loaded {len(segments)} segments")

    # Use proper computational domain
    domain_x = [0.0, 0.4]  # Computational domain
    domain_y = [0.0, 0.5]  # Computational domain

    # Actual interface bounds
    actual_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
    actual_y = [segments[:, 1].min(), segments[:, 3].min(), segments[:, 1].max(), segments[:, 3].max()]
    interface_x = [min(actual_x), max(actual_x)]
    interface_y = [min(actual_y), max(actual_y)]

    print(f"📏 Computational Domain: x=[0.0, 0.4]m, y=[0.0, 0.5]m")
    print(f"📏 Interface Bounds: x=[{interface_x[0]:.3f}, {interface_x[1]:.3f}]m, y=[{interface_y[0]:.3f}, {interface_y[1]:.3f}]m")
    print(f"📐 Interface deviation from y=0.25m: {interface_y[0]-0.25:.3f}m to {interface_y[1]-0.25:.3f}m")

    # Extract geometric features
    print(f"\n🔍 RT INTERFACE GEOMETRIC ANALYSIS")
    print("-" * 70)
    features = extract_interface_features(segments)

    print(f"📊 RT Interface Properties:")
    print(f"   Segments: {features.get('n_segments', 'N/A')}")
    print(f"   Tortuosity: {features.get('tortuosity', 'N/A'):.3f}")
    print(f"   Complexity Score: {features.get('complexity_score', 'N/A'):.3f}")
    print(f"   Characteristic Length: {features.get('characteristic_length', 'N/A'):.6f}m")
    print(f"   Mean Segment Length: {features.get('mean_segment_length', 'N/A'):.6f}m")
    print(f"   Direction Change Rate: {features.get('direction_change_rate', 'N/A'):.3f}")
    print(f"   Linearity R²: {features.get('linearity_r_squared', 'N/A'):.6f}")

    # AI parameter suggestion
    print(f"\n🧠 AI PARAMETER ANALYSIS")
    print("-" * 70)
    ai_params = suggest_optimal_parameters_adaptive(features)
    print(f"🎯 AI-Suggested Parameters:")
    print(f"   Initial δ: {ai_params.get('initial_delta', 'N/A'):.6f}m")
    print(f"   Delta Factor: {ai_params.get('delta_factor', 'N/A'):.3f}")
    print(f"   Number of Steps: {ai_params.get('num_steps', 'N/A'):.0f}")

    # Scale-corrected parameters based on domain
    domain_width = domain_x[1] - domain_x[0]  # 0.4m
    interface_width = interface_x[1] - interface_x[0]  # actual interface span

    # Suggest scale corrections
    scale_corrected_params = [
        {
            'name': 'AI Original',
            'initial_delta': ai_params['initial_delta'],
            'delta_factor': ai_params['delta_factor'],
            'num_steps': int(ai_params['num_steps'])
        },
        {
            'name': 'Domain-Scaled (5% of domain)',
            'initial_delta': domain_width * 0.05,  # 5% of computational domain
            'delta_factor': ai_params['delta_factor'],
            'num_steps': int(ai_params['num_steps'])
        },
        {
            'name': 'Interface-Scaled (20% of interface)',
            'initial_delta': interface_width * 0.2,  # 20% of interface width
            'delta_factor': ai_params['delta_factor'],
            'num_steps': int(ai_params['num_steps'])
        },
        {
            'name': 'Conservative Small',
            'initial_delta': 0.01,  # 1cm boxes
            'delta_factor': 1.5,    # Moderate scaling
            'num_steps': 20         # More steps for precision
        }
    ]

    print(f"\n📐 FRACTAL DIMENSION CALCULATIONS")
    print("=" * 70)

    best_result = None
    best_params = None

    for params in scale_corrected_params:
        print(f"\n🧪 Testing: {params['name']}")
        print(f"   δ₀={params['initial_delta']:.6f}m, factor={params['delta_factor']:.3f}, steps={params['num_steps']}")

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
                print(f"   ✅ Dimension: {dimension:.6f}")
                print(f"   📈 R²: {r_squared:.6f}")
                print(f"   ⏱️  Time: {calc_time:.1f}s")

                # Save best result
                if best_result is None or (r_squared > best_result.get('r_squared', 0)):
                    best_result = {
                        'dimension': dimension,
                        'r_squared': r_squared,
                        'time': calc_time
                    }
                    best_params = params

            else:
                print(f"   ❌ Failed: NaN result")

        except Exception as e:
            calc_time = time.time() - start_time
            print(f"   💥 Error: {e}")

    # Summary
    print(f"\n🏆 RT INTERFACE ANALYSIS SUMMARY")
    print("=" * 70)

    if best_result:
        print(f"✅ Best Result: {best_params['name']}")
        print(f"   Fractal Dimension: {best_result['dimension']:.6f}")
        print(f"   R² (goodness of fit): {best_result['r_squared']:.6f}")
        print(f"   Computation Time: {best_result['time']:.1f}s")

        # Physical interpretation
        dimension = best_result['dimension']
        if dimension < 1.05:
            complexity = "Very Smooth (nearly straight line)"
        elif dimension < 1.15:
            complexity = "Slightly Rough Interface"
        elif dimension < 1.3:
            complexity = "Moderately Complex RT Interface"
        elif dimension < 1.5:
            complexity = "Complex RT Interface"
        else:
            complexity = "Highly Complex RT Interface"

        print(f"🔬 RT Interface Classification: {complexity}")
        print(f"🌊 Physical Context:")
        print(f"   • t=3s after RT instability onset")
        print(f"   • Interface evolved from straight line y=0.25m")
        print(f"   • Computational domain: 0.4m × 0.5m")
        print(f"   • Interface spans: {interface_x[1]-interface_x[0]:.3f}m horizontally")

        # Record successful result in AI learning
        print(f"\n🤖 Recording successful result in AI learning database...")
        record_parameter_feedback(
            features, best_params, best_result['dimension'], None,
            best_result['r_squared'], best_result['time'], "basic_box_counting"
        )
        print(f"✅ AI database updated with RT interface knowledge")

    else:
        print(f"❌ All parameter combinations failed")
        print(f"   RT interface may need specialized approaches")

    return {
        'segments': len(segments),
        'features': features,
        'best_params': best_params,
        'best_result': best_result,
        'domain_context': {
            'computational': [domain_x, domain_y],
            'interface_bounds': [interface_x, interface_y],
            'initial_interface': 0.25
        }
    }


if __name__ == "__main__":
    print("🚀 RT INTERFACE ANALYSIS - DOMAIN CORRECTED")
    print("AI system with proper Dalziel computational domain context")
    print()

    result = test_rt_interface_with_proper_domain()

    print(f"\n🎉 RT INTERFACE ANALYSIS COMPLETE")
    print(f"AI system tested on real RT instability data with proper scale context!")