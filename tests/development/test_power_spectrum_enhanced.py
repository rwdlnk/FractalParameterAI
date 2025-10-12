#!/usr/bin/env python3
"""
Test Power Spectrum Enhanced AI Parameter Selection

Demonstrates the new spectral analysis approach for predicting optimal
box counting parameters by analyzing frequency content of interfaces.

This advanced method should predict the successful RT parameters directly!
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('.')

from core.power_spectrum_features import extract_power_spectrum_features, suggest_parameters_from_spectrum
from core.interface_features import extract_interface_features, suggest_optimal_parameters_adaptive
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


def test_spectrum_vs_traditional():
    """Compare spectral analysis with traditional geometric features for RT interface."""

    print("🌊 POWER SPECTRUM vs TRADITIONAL ANALYSIS")
    print("Testing enhanced AI parameter selection on RT interface")
    print("=" * 70)

    # Load RT interface data
    rt_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/results_Time_3000/interface_fractal_5.dat"

    print("📂 Loading RT interface data...")
    segments = load_rt_interface(rt_file)
    print(f"✅ Loaded {len(segments)} segments")

    # Domain context
    domain_x = [0.0, 0.4]  # Computational domain
    domain_y = [0.0, 0.5]
    domain_scale = domain_x[1] - domain_x[0]  # 0.4m

    print(f"📏 Domain: {domain_x[0]:.1f}m ≤ x ≤ {domain_x[1]:.1f}m")
    print(f"📐 Domain scale: {domain_scale:.1f}m")

    # Extract traditional geometric features
    print(f"\n🔍 TRADITIONAL GEOMETRIC ANALYSIS")
    print("-" * 50)
    start_time = time.time()
    geometric_features = extract_interface_features(segments)
    geometric_time = time.time() - start_time

    print(f"📊 Traditional Features ({geometric_time:.3f}s):")
    print(f"   Tortuosity: {geometric_features.get('tortuosity', 'N/A'):.3f}")
    print(f"   Complexity: {geometric_features.get('complexity_score', 'N/A'):.3f}")
    print(f"   Characteristic Length: {geometric_features.get('characteristic_length', 'N/A'):.6f}m")
    print(f"   Mean Segment Length: {geometric_features.get('mean_segment_length', 'N/A'):.6f}m")

    traditional_params = suggest_optimal_parameters_adaptive(geometric_features)
    print(f"\n🎯 Traditional AI Suggestions:")
    print(f"   δ₀: {traditional_params.get('initial_delta', 'N/A'):.6f}m")
    print(f"   Factor: {traditional_params.get('delta_factor', 'N/A'):.3f}")
    print(f"   Steps: {traditional_params.get('num_steps', 'N/A'):.0f}")

    # Extract power spectrum features
    print(f"\n🌈 POWER SPECTRUM ANALYSIS")
    print("-" * 50)
    start_time = time.time()
    spectral_features = extract_power_spectrum_features(segments, method='welch', num_points=512)
    spectral_time = time.time() - start_time

    print(f"📈 Spectral Features ({spectral_time:.3f}s):")
    print(f"   Dominant Frequency: {spectral_features.get('dominant_frequency', 'N/A'):.3f} Hz")
    print(f"   Spectral Slope: {spectral_features.get('spectral_slope', 'N/A'):.3f}")
    print(f"   High Freq Content: {spectral_features.get('high_frequency_content', 'N/A'):.3f}")
    print(f"   Spectral Bandwidth: {spectral_features.get('spectral_bandwidth', 'N/A'):.3f} Hz")
    print(f"   Spectral Centroid: {spectral_features.get('spectral_centroid', 'N/A'):.3f} Hz")

    spectral_params = suggest_parameters_from_spectrum(spectral_features, domain_scale)
    print(f"\n🎯 Spectral AI Suggestions:")
    print(f"   δ₀: {spectral_params.get('initial_delta', 'N/A'):.6f}m")
    print(f"   Factor: {spectral_params.get('delta_factor', 'N/A'):.3f}")
    print(f"   Steps: {spectral_params.get('num_steps', 'N/A'):.0f}")
    print(f"   Confidence: {spectral_params.get('confidence', 'N/A'):.3f}")

    # Compare with known successful parameters
    print(f"\n✅ KNOWN SUCCESSFUL PARAMETERS (from previous analysis):")
    successful_params = {
        'initial_delta': 0.020000,  # Domain-scaled (5% of domain)
        'delta_factor': 1.263,
        'num_steps': 15
    }
    print(f"   δ₀: {successful_params['initial_delta']:.6f}m")
    print(f"   Factor: {successful_params['delta_factor']:.3f}")
    print(f"   Steps: {successful_params['num_steps']:.0f}")
    print(f"   → Achieved D = 1.215545, R² = 0.987739")

    # Calculate prediction errors
    print(f"\n📊 PREDICTION ACCURACY COMPARISON")
    print("=" * 70)

    def calc_param_error(predicted, actual):
        """Calculate percentage error in parameter prediction."""
        if actual == 0:
            return float('inf') if predicted != 0 else 0
        return abs(predicted - actual) / actual * 100

    print(f"{'Parameter':<15} | {'Traditional':<12} | {'Spectral':<12} | {'Actual':<12}")
    print("-" * 60)

    # Delta errors
    trad_delta_err = calc_param_error(traditional_params['initial_delta'], successful_params['initial_delta'])
    spec_delta_err = calc_param_error(spectral_params['initial_delta'], successful_params['initial_delta'])
    print(f"{'δ₀ error %':<15} | {trad_delta_err:>11.1f} | {spec_delta_err:>11.1f} | {'0.0':>12}")

    # Factor errors
    trad_factor_err = calc_param_error(traditional_params['delta_factor'], successful_params['delta_factor'])
    spec_factor_err = calc_param_error(spectral_params['delta_factor'], successful_params['delta_factor'])
    print(f"{'Factor error %':<15} | {trad_factor_err:>11.1f} | {spec_factor_err:>11.1f} | {'0.0':>12}")

    # Steps errors
    trad_steps_err = calc_param_error(traditional_params['num_steps'], successful_params['num_steps'])
    spec_steps_err = calc_param_error(spectral_params['num_steps'], successful_params['num_steps'])
    print(f"{'Steps error %':<15} | {trad_steps_err:>11.1f} | {spec_steps_err:>11.1f} | {'0.0':>12}")

    # Overall accuracy score
    trad_avg_err = (trad_delta_err + trad_factor_err + trad_steps_err) / 3
    spec_avg_err = (spec_delta_err + spec_factor_err + spec_steps_err) / 3

    print(f"\n📈 OVERALL ACCURACY:")
    print(f"   Traditional Method: {100 - trad_avg_err:.1f}% accurate")
    print(f"   Spectral Method: {100 - spec_avg_err:.1f}% accurate")

    if spec_avg_err < trad_avg_err:
        improvement = trad_avg_err - spec_avg_err
        print(f"   🎉 Spectral method is {improvement:.1f}% more accurate!")
    else:
        print(f"   📊 Traditional method performs better by {spec_avg_err - trad_avg_err:.1f}%")

    # Test both parameter sets on actual RT interface
    print(f"\n🔬 VALIDATION: TESTING PARAMETER PREDICTIONS")
    print("=" * 70)

    test_params = [
        ('Traditional AI', traditional_params),
        ('Spectral AI', spectral_params),
        ('Known Success', successful_params)
    ]

    for name, params in test_params:
        print(f"\n🧪 Testing: {name}")
        print(f"   δ₀={params['initial_delta']:.6f}m, factor={params['delta_factor']:.3f}, steps={params['num_steps']:.0f}")

        try:
            start_time = time.time()
            result = calculate_fractal_dimension(
                segments, domain_x, domain_y,
                params['initial_delta'],
                params['delta_factor'],
                int(params['num_steps'])
            )
            calc_time = time.time() - start_time

            dimension = result.get('dimension')
            r_squared = result.get('r_squared')

            if dimension is not None and not (dimension != dimension):  # Not NaN
                print(f"   ✅ Dimension: {dimension:.6f}")
                print(f"   📈 R²: {r_squared:.6f}")
                print(f"   ⏱️  Time: {calc_time:.1f}s")

                # Compare to known good result
                known_dimension = 1.215545
                error_pct = abs(dimension - known_dimension) / known_dimension * 100
                print(f"   📊 Error vs known: {error_pct:.2f}%")
            else:
                print(f"   ❌ Failed: NaN result")

        except Exception as e:
            print(f"   💥 Error: {e}")

    return {
        'traditional_features': geometric_features,
        'spectral_features': spectral_features,
        'traditional_params': traditional_params,
        'spectral_params': spectral_params,
        'successful_params': successful_params
    }


def analyze_spectral_insights():
    """Analyze what the spectral features reveal about RT interface."""

    print(f"\n🔬 SPECTRAL INSIGHTS FOR RT INTERFACE")
    print("=" * 70)

    # Load RT interface
    rt_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/results_Time_3000/interface_fractal_5.dat"
    segments = load_rt_interface(rt_file)

    # Extract spectral features
    spectral_features = extract_power_spectrum_features(segments, method='welch', num_points=512)

    print(f"💡 Physical Interpretation of Spectral Features:")
    print(f"-" * 50)

    dom_freq = spectral_features.get('dominant_frequency', 0)
    if dom_freq > 0:
        characteristic_length = 1.0 / dom_freq
        print(f"   Dominant Frequency: {dom_freq:.3f} Hz")
        print(f"   → Characteristic Length Scale: {characteristic_length:.6f}m")
        print(f"   → Suggests initial box size: ~{characteristic_length/10:.6f}m")

    slope = spectral_features.get('spectral_slope', 0)
    print(f"   Spectral Slope: {slope:.3f}")
    if slope > 2:
        print(f"   → Strong fractal scaling (steep slope)")
        print(f"   → Suggests larger delta factor for scale separation")
    elif slope > 1:
        print(f"   → Moderate fractal scaling")
        print(f"   → Suggests moderate delta factor")
    else:
        print(f"   → Weak fractal scaling")
        print(f"   → Suggests smaller delta factor")

    high_freq = spectral_features.get('high_frequency_content', 0)
    print(f"   High Frequency Content: {high_freq:.3f}")
    if high_freq > 0.3:
        print(f"   → Lots of fine detail → need more steps")
    else:
        print(f"   → Mostly smooth → fewer steps sufficient")

    bandwidth = spectral_features.get('spectral_bandwidth', 0)
    print(f"   Spectral Bandwidth: {bandwidth:.3f} Hz")
    print(f"   → Range of important scales in interface")


if __name__ == "__main__":
    print("🚀 POWER SPECTRUM ENHANCED AI TESTING")
    print("Testing spectral analysis for fractal parameter prediction")
    print()

    # Main comparison test
    results = test_spectrum_vs_traditional()

    # Spectral insights
    analyze_spectral_insights()

    print(f"\n🎉 POWER SPECTRUM ANALYSIS COMPLETE")
    print(f"Advanced characterization methods tested on real RT interface!")