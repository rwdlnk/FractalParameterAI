#!/usr/bin/env python3
"""
Analyze Correlations Between Geometric Features and AI Parameters

Investigates relationships between curve-derived quantities (tortuosity, complexity, etc.)
and the AI-suggested parameters to understand what the system has learned.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
sys.path.append('.')

from core.interface_features import extract_interface_features, suggest_optimal_parameters_adaptive
from test_multi_fractal_convergence import generate_koch_curve, generate_dragon_curve, generate_sierpinski_curve


def load_feedback_data():
    """Load all feedback records from the learning database."""
    records = []
    try:
        with open('feedback_data.jsonl', 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    records.append(data)
    except FileNotFoundError:
        print("No feedback data file found")
        return []

    print(f"Loaded {len(records)} feedback records")
    return records


def extract_feature_parameter_data(records):
    """Extract feature and parameter data for correlation analysis."""

    data = {
        'tortuosity': [],
        'complexity_score': [],
        'direction_change_rate': [],
        'mean_segment_length': [],
        'characteristic_length': [],
        'linearity_r_squared': [],
        'initial_delta': [],
        'delta_factor': [],
        'num_steps': [],
        'success': [],
        'error_pct': []
    }

    for record in records:
        features = record.get('features', {})
        params = record.get('suggested_parameters', {})

        # Extract features
        for feature_name in ['tortuosity', 'complexity_score', 'direction_change_rate',
                           'mean_segment_length', 'characteristic_length', 'linearity_r_squared']:
            if feature_name in features:
                data[feature_name].append(features[feature_name])
            else:
                data[feature_name].append(np.nan)

        # Extract parameters
        for param_name in ['initial_delta', 'delta_factor', 'num_steps']:
            if param_name in params:
                data[param_name].append(params[param_name])
            else:
                data[param_name].append(np.nan)

        # Extract success metrics
        data['success'].append(record.get('success', False))

        # Calculate error percentage if possible
        if record.get('dimension_result') and record.get('theoretical_dimension'):
            error_pct = abs(record['dimension_result'] - record['theoretical_dimension']) / record['theoretical_dimension'] * 100
            data['error_pct'].append(error_pct)
        else:
            data['error_pct'].append(np.nan)

    # Convert to numpy arrays and remove NaN entries
    for key in data:
        data[key] = np.array(data[key])

    return data


def calculate_correlations(data):
    """Calculate correlation coefficients between features and parameters."""

    feature_names = ['tortuosity', 'complexity_score', 'direction_change_rate',
                    'mean_segment_length', 'characteristic_length', 'linearity_r_squared']
    param_names = ['initial_delta', 'delta_factor', 'num_steps']

    print("\n📊 FEATURE-PARAMETER CORRELATIONS")
    print("=" * 70)
    print(f"{'Feature':25s} | {'δ₀':>8s} | {'factor':>8s} | {'steps':>8s}")
    print("-" * 70)

    correlations = {}

    for feature in feature_names:
        correlations[feature] = {}
        feature_data = data[feature]

        # Remove NaN values for correlation calculation
        valid_indices = ~np.isnan(feature_data)

        if np.sum(valid_indices) < 3:  # Need at least 3 points for correlation
            print(f"{feature:25s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s}")
            continue

        row_str = f"{feature:25s} |"

        for param in param_names:
            param_data = data[param]

            # Find common valid indices
            common_valid = valid_indices & ~np.isnan(param_data)

            if np.sum(common_valid) >= 3:
                correlation, p_value = stats.pearsonr(feature_data[common_valid], param_data[common_valid])
                correlations[feature][param] = {'r': correlation, 'p': p_value}

                # Format correlation with significance indicator
                if p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = ""

                row_str += f" {correlation:+7.3f}{sig:1s} |"
            else:
                correlations[feature][param] = {'r': np.nan, 'p': np.nan}
                row_str += f" {'N/A':>8s} |"

        print(row_str)

    print("\n* p < 0.05, ** p < 0.01")
    return correlations


def analyze_ai_learning_patterns(data):
    """Analyze what patterns the AI has learned."""

    print(f"\n🧠 AI LEARNING PATTERN ANALYSIS")
    print("=" * 70)

    # Successful vs failed parameter patterns
    success_mask = np.array(data['success'])

    if np.sum(success_mask) > 0 and np.sum(~success_mask) > 0:
        print(f"\n✅ SUCCESSFUL CASES ({np.sum(success_mask)} cases):")
        for param in ['initial_delta', 'delta_factor', 'num_steps']:
            successful_values = data[param][success_mask]
            successful_values = successful_values[~np.isnan(successful_values)]
            if len(successful_values) > 0:
                print(f"   {param:15s}: {np.mean(successful_values):6.2f} ± {np.std(successful_values):5.2f} (range: {np.min(successful_values):.2f}-{np.max(successful_values):.2f})")

        print(f"\n❌ FAILED CASES ({np.sum(~success_mask)} cases):")
        for param in ['initial_delta', 'delta_factor', 'num_steps']:
            failed_values = data[param][~success_mask]
            failed_values = failed_values[~np.isnan(failed_values)]
            if len(failed_values) > 0:
                print(f"   {param:15s}: {np.mean(failed_values):6.2f} ± {np.std(failed_values):5.2f} (range: {np.min(failed_values):.2f}-{np.max(failed_values):.2f})")

    # Error patterns
    valid_errors = data['error_pct'][~np.isnan(data['error_pct'])]
    if len(valid_errors) > 0:
        print(f"\n📈 ERROR ANALYSIS:")
        print(f"   Mean error: {np.mean(valid_errors):6.2f}%")
        print(f"   Best error: {np.min(valid_errors):6.2f}%")
        print(f"   Worst error: {np.max(valid_errors):6.2f}%")
        print(f"   Cases < 10%: {np.sum(valid_errors < 10)}/{len(valid_errors)}")


def test_current_ai_understanding():
    """Test AI parameter suggestions on different fractal types."""

    print(f"\n🔬 CURRENT AI PARAMETER UNDERSTANDING")
    print("=" * 70)

    test_fractals = [
        ("Koch L3", lambda: generate_koch_curve(3)),
        ("Dragon L3", lambda: generate_dragon_curve(3)),
        ("Sierpinski L3", lambda: generate_sierpinski_curve(3)),
    ]

    for name, generator in test_fractals:
        print(f"\n{name}:")
        segments = np.array(generator())
        features = extract_interface_features(segments)
        ai_params = suggest_optimal_parameters_adaptive(features)

        # Key features
        print(f"   Features:")
        print(f"     Tortuosity: {features.get('tortuosity', 'N/A'):.3f}")
        print(f"     Complexity: {features.get('complexity_score', 'N/A'):.3f}")
        print(f"     Mean length: {features.get('mean_segment_length', 'N/A'):.3f}")

        # AI suggestions
        print(f"   AI Parameters:")
        print(f"     δ₀: {ai_params.get('initial_delta', 'N/A'):.3f}")
        print(f"     factor: {ai_params.get('delta_factor', 'N/A'):.3f}")
        print(f"     steps: {ai_params.get('num_steps', 'N/A'):.1f}")


def main():
    print("🔍 FEATURE-PARAMETER CORRELATION ANALYSIS")
    print("Investigating relationships between geometric features and AI parameters")
    print("=" * 70)

    # Load feedback data
    records = load_feedback_data()
    if not records:
        print("No data available for analysis")
        return

    # Extract data for analysis
    data = extract_feature_parameter_data(records)

    # Calculate correlations
    correlations = calculate_correlations(data)

    # Analyze AI learning patterns
    analyze_ai_learning_patterns(data)

    # Test current AI understanding
    test_current_ai_understanding()

    # Summary insights
    print(f"\n💡 KEY INSIGHTS:")
    print("=" * 70)

    # Find strongest correlations
    strong_correlations = []
    for feature, params in correlations.items():
        for param, stats_dict in params.items():
            r = stats_dict.get('r', 0)
            p = stats_dict.get('p', 1)
            if not np.isnan(r) and abs(r) > 0.3 and p < 0.05:
                strong_correlations.append((feature, param, r, p))

    if strong_correlations:
        print("\nStrongest feature-parameter relationships:")
        for feature, param, r, p in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            direction = "increases" if r > 0 else "decreases"
            print(f"   • Higher {feature} → AI {direction} {param} (r={r:+.3f}, p={p:.3f})")
    else:
        print("\nNo strong correlations found - AI may be using complex non-linear patterns")

    print(f"\nDatabase contains {len(records)} learning examples")
    print(f"AI system is continuously learning from each analysis")


if __name__ == "__main__":
    main()