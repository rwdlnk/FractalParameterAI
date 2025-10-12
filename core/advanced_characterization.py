#!/usr/bin/env python3
"""
Integrated Advanced Characterization System

Combines power spectrum, wavelet, curvature, and self-similarity analysis
to provide comprehensive interface characterization and optimal parameter
prediction for fractal dimension calculation.

This implements the advanced workflow:
Interface segments → Multi-scale analysis → AI parameter prediction → Box counting
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from .power_spectrum_features import extract_power_spectrum_features, suggest_parameters_from_spectrum
from .wavelet_features import extract_wavelet_features, suggest_parameters_from_wavelets
from .curvature_features import extract_curvature_features, suggest_parameters_from_curvature
from .self_similarity_features import extract_self_similarity_features, suggest_parameters_from_self_similarity
from .interface_features import extract_interface_features, suggest_optimal_parameters_adaptive


class AdvancedCharacterizationEngine:
    """
    Advanced multi-scale interface characterization engine.

    Combines multiple advanced analysis methods to provide optimal
    box counting parameter prediction with high accuracy.
    """

    def __init__(self, methods=None, weights=None):
        """
        Initialize advanced characterization engine.

        Args:
            methods: List of methods to use ['spectrum', 'wavelet', 'curvature', 'self_similarity', 'traditional']
            weights: Dictionary of method weights for consensus building
        """
        if methods is None:
            self.methods = ['spectrum', 'wavelet', 'curvature', 'self_similarity', 'traditional']
        else:
            self.methods = methods

        if weights is None:
            self.weights = {
                'spectrum': 0.25,
                'wavelet': 0.25,
                'curvature': 0.20,
                'self_similarity': 0.25,
                'traditional': 0.05
            }
        else:
            self.weights = weights

    def analyze_comprehensive(self, segments, domain_scale=None):
        """
        Perform comprehensive advanced characterization analysis.

        Args:
            segments: Nx4 array of [x1, y1, x2, y2] segments
            domain_scale: Characteristic domain scale (auto-detected if None)

        Returns:
            analysis_results: Comprehensive analysis results
        """
        if len(segments) == 0:
            return self._empty_results()

        # Auto-detect domain scale if not provided
        if domain_scale is None:
            all_x = [segments[:, 0].min(), segments[:, 2].min(), segments[:, 0].max(), segments[:, 2].max()]
            domain_scale = max(all_x) - min(all_x)
            if domain_scale <= 0:
                domain_scale = 1.0

        print(f"🔬 ADVANCED CHARACTERIZATION ANALYSIS")
        print(f"   Methods: {', '.join(self.methods)}")
        print(f"   Domain scale: {domain_scale:.6f}")
        print("-" * 50)

        # Storage for all results
        feature_results = {}
        parameter_suggestions = {}
        analysis_times = {}
        confidences = {}

        # Method 1: Power Spectrum Analysis
        if 'spectrum' in self.methods:
            start_time = time.time()
            try:
                print("📈 Power Spectrum Analysis...")
                spectral_features = extract_power_spectrum_features(segments, method='welch', num_points=512)
                spectral_params = suggest_parameters_from_spectrum(spectral_features, domain_scale)

                feature_results['spectrum'] = spectral_features
                parameter_suggestions['spectrum'] = spectral_params
                confidences['spectrum'] = spectral_params.get('confidence', 0.5)
                analysis_times['spectrum'] = time.time() - start_time
                print(f"   ✅ Completed ({analysis_times['spectrum']:.2f}s)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                feature_results['spectrum'] = {}
                parameter_suggestions['spectrum'] = self._default_params('spectrum')
                confidences['spectrum'] = 0.0
                analysis_times['spectrum'] = time.time() - start_time

        # Method 2: Wavelet Analysis
        if 'wavelet' in self.methods:
            start_time = time.time()
            try:
                print("🌊 Wavelet Analysis...")
                wavelet_features = extract_wavelet_features(segments, method='continuous')
                wavelet_params = suggest_parameters_from_wavelets(wavelet_features, domain_scale)

                feature_results['wavelet'] = wavelet_features
                parameter_suggestions['wavelet'] = wavelet_params
                confidences['wavelet'] = wavelet_params.get('confidence', 0.5)
                analysis_times['wavelet'] = time.time() - start_time
                print(f"   ✅ Completed ({analysis_times['wavelet']:.2f}s)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                feature_results['wavelet'] = {}
                parameter_suggestions['wavelet'] = self._default_params('wavelet')
                confidences['wavelet'] = 0.0
                analysis_times['wavelet'] = time.time() - start_time

        # Method 3: Curvature Analysis
        if 'curvature' in self.methods:
            start_time = time.time()
            try:
                print("📐 Curvature Analysis...")
                curvature_features = extract_curvature_features(segments, method='finite_difference')
                curvature_params = suggest_parameters_from_curvature(curvature_features, domain_scale)

                feature_results['curvature'] = curvature_features
                parameter_suggestions['curvature'] = curvature_params
                confidences['curvature'] = curvature_params.get('confidence', 0.5)
                analysis_times['curvature'] = time.time() - start_time
                print(f"   ✅ Completed ({analysis_times['curvature']:.2f}s)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                feature_results['curvature'] = {}
                parameter_suggestions['curvature'] = self._default_params('curvature')
                confidences['curvature'] = 0.0
                analysis_times['curvature'] = time.time() - start_time

        # Method 4: Self-Similarity Analysis
        if 'self_similarity' in self.methods:
            start_time = time.time()
            try:
                print("🔄 Self-Similarity Analysis...")
                similarity_features = extract_self_similarity_features(segments)
                similarity_params = suggest_parameters_from_self_similarity(similarity_features, domain_scale)

                feature_results['self_similarity'] = similarity_features
                parameter_suggestions['self_similarity'] = similarity_params
                confidences['self_similarity'] = similarity_params.get('confidence', 0.5)
                analysis_times['self_similarity'] = time.time() - start_time
                print(f"   ✅ Completed ({analysis_times['self_similarity']:.2f}s)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                feature_results['self_similarity'] = {}
                parameter_suggestions['self_similarity'] = self._default_params('self_similarity')
                confidences['self_similarity'] = 0.0
                analysis_times['self_similarity'] = time.time() - start_time

        # Method 5: Traditional Geometric Analysis (for comparison)
        if 'traditional' in self.methods:
            start_time = time.time()
            try:
                print("📊 Traditional Geometric Analysis...")
                traditional_features = extract_interface_features(segments)
                traditional_params = suggest_optimal_parameters_adaptive(traditional_features)

                feature_results['traditional'] = traditional_features
                parameter_suggestions['traditional'] = {
                    'initial_delta': traditional_params.get('initial_delta', domain_scale * 0.05),
                    'delta_factor': traditional_params.get('delta_factor', 1.5),
                    'num_steps': traditional_params.get('num_steps', 12),
                    'method': 'traditional_geometric',
                    'confidence': 0.7  # Moderate confidence for traditional methods
                }
                confidences['traditional'] = 0.7
                analysis_times['traditional'] = time.time() - start_time
                print(f"   ✅ Completed ({analysis_times['traditional']:.2f}s)")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                feature_results['traditional'] = {}
                parameter_suggestions['traditional'] = self._default_params('traditional')
                confidences['traditional'] = 0.0
                analysis_times['traditional'] = time.time() - start_time

        # Build consensus parameters
        consensus_params = self._build_consensus(parameter_suggestions, confidences)

        # Create comprehensive results
        total_time = sum(analysis_times.values())

        results = {
            'consensus_parameters': consensus_params,
            'individual_parameters': parameter_suggestions,
            'feature_results': feature_results,
            'confidences': confidences,
            'analysis_times': analysis_times,
            'total_analysis_time': total_time,
            'methods_used': self.methods,
            'domain_scale': domain_scale,
            'segments_analyzed': len(segments)
        }

        self._print_summary(results)

        return results

    def _build_consensus(self, parameter_suggestions, confidences):
        """
        Build consensus parameters from multiple methods using weighted voting.

        Args:
            parameter_suggestions: Dictionary of parameter suggestions by method
            confidences: Dictionary of confidence scores by method

        Returns:
            consensus_params: Consensus parameter set
        """
        print(f"\n🏛️  BUILDING CONSENSUS PARAMETERS")
        print("-" * 50)

        # Collect valid suggestions
        valid_methods = []
        valid_params = []
        valid_confidences = []

        for method in self.methods:
            if method in parameter_suggestions and method in confidences:
                params = parameter_suggestions[method]
                confidence = confidences[method]

                # Check if parameters are valid
                if (np.isfinite(params.get('initial_delta', np.nan)) and
                    np.isfinite(params.get('delta_factor', np.nan)) and
                    np.isfinite(params.get('num_steps', np.nan)) and
                    confidence > 0):

                    valid_methods.append(method)
                    valid_params.append(params)
                    valid_confidences.append(confidence)

        print(f"📊 Valid methods: {len(valid_methods)}/{len(self.methods)}")
        for method, conf in zip(valid_methods, valid_confidences):
            print(f"   {method}: confidence = {conf:.3f}")

        if not valid_methods:
            print("❌ No valid parameter suggestions - using defaults")
            return self._default_params('consensus')

        # Calculate weighted consensus
        total_weight = 0
        weighted_delta = 0
        weighted_factor = 0
        weighted_steps = 0

        for method, params, confidence in zip(valid_methods, valid_params, valid_confidences):
            method_weight = self.weights.get(method, 0.2)
            effective_weight = method_weight * confidence
            total_weight += effective_weight

            weighted_delta += params['initial_delta'] * effective_weight
            weighted_factor += params['delta_factor'] * effective_weight
            weighted_steps += params['num_steps'] * effective_weight

            print(f"   {method}: weight = {method_weight:.2f} × conf = {confidence:.2f} = {effective_weight:.3f}")

        if total_weight == 0:
            print("❌ Total weight is zero - using defaults")
            return self._default_params('consensus')

        # Normalize by total weight
        consensus_delta = weighted_delta / total_weight
        consensus_factor = weighted_factor / total_weight
        consensus_steps = int(round(weighted_steps / total_weight))

        # Calculate consensus confidence
        consensus_confidence = np.mean(valid_confidences)

        # Apply sanity checks
        consensus_delta = np.clip(consensus_delta, 1e-6, 1e6)
        consensus_factor = np.clip(consensus_factor, 1.1, 3.0)
        consensus_steps = np.clip(consensus_steps, 5, 50)

        consensus_params = {
            'initial_delta': consensus_delta,
            'delta_factor': consensus_factor,
            'num_steps': consensus_steps,
            'method': 'advanced_consensus',
            'confidence': consensus_confidence,
            'total_weight': total_weight,
            'contributing_methods': valid_methods,
            'method_weights': {method: self.weights.get(method, 0.2) for method in valid_methods}
        }

        print(f"\n🎯 CONSENSUS PARAMETERS:")
        print(f"   δ₀: {consensus_delta:.6f}")
        print(f"   Factor: {consensus_factor:.3f}")
        print(f"   Steps: {consensus_steps}")
        print(f"   Confidence: {consensus_confidence:.3f}")
        print(f"   Methods: {len(valid_methods)}")

        return consensus_params

    def _default_params(self, method_name):
        """Generate default parameters for failed methods."""
        return {
            'initial_delta': 0.01,
            'delta_factor': 1.5,
            'num_steps': 12,
            'method': f'{method_name}_default',
            'confidence': 0.1
        }

    def _empty_results(self):
        """Return empty results structure."""
        return {
            'consensus_parameters': self._default_params('empty'),
            'individual_parameters': {},
            'feature_results': {},
            'confidences': {},
            'analysis_times': {},
            'total_analysis_time': 0.0,
            'methods_used': [],
            'domain_scale': 1.0,
            'segments_analyzed': 0
        }

    def _print_summary(self, results):
        """Print comprehensive analysis summary."""
        print(f"\n🏆 ADVANCED CHARACTERIZATION SUMMARY")
        print("=" * 70)

        consensus = results['consensus_parameters']
        individual = results['individual_parameters']
        confidences = results['confidences']

        # Summary table
        print(f"{'Method':<15} | {'δ₀':<10} | {'Factor':<8} | {'Steps':<6} | {'Conf':<6} | {'Time':<6}")
        print("-" * 70)

        for method in self.methods:
            if method in individual:
                params = individual[method]
                conf = confidences.get(method, 0.0)
                time_val = results['analysis_times'].get(method, 0.0)

                print(f"{method:<15} | {params.get('initial_delta', 0):<10.6f} | "
                      f"{params.get('delta_factor', 0):<8.3f} | {params.get('num_steps', 0):<6.0f} | "
                      f"{conf:<6.3f} | {time_val:<6.2f}")

        print("-" * 70)
        print(f"{'CONSENSUS':<15} | {consensus['initial_delta']:<10.6f} | "
              f"{consensus['delta_factor']:<8.3f} | {consensus['num_steps']:<6.0f} | "
              f"{consensus['confidence']:<6.3f} | {results['total_analysis_time']:<6.2f}")

        # Performance insights
        best_confidence = max(confidences.values()) if confidences else 0.0
        worst_confidence = min(confidences.values()) if confidences else 0.0

        print(f"\n📈 ANALYSIS INSIGHTS:")
        print(f"   Best method confidence: {best_confidence:.3f}")
        print(f"   Worst method confidence: {worst_confidence:.3f}")
        print(f"   Consensus confidence: {consensus['confidence']:.3f}")
        print(f"   Total analysis time: {results['total_analysis_time']:.2f}s")
        print(f"   Methods successful: {len([c for c in confidences.values() if c > 0])}/{len(self.methods)}")


def test_advanced_characterization_on_rt_interface():
    """
    Test the integrated advanced characterization system on RT interface.
    """
    print("🚀 TESTING ADVANCED CHARACTERIZATION ON RT INTERFACE")
    print("=" * 70)

    # Load RT interface
    def load_rt_interface(filepath):
        segments = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                x1, y1, x2, y2 = map(float, line.split())
                segments.append([x1, y1, x2, y2])
        return np.array(segments)

    rt_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/results_Time_3000/interface_fractal_5.dat"
    segments = load_rt_interface(rt_file)
    domain_scale = 0.4  # 0.4m domain width

    print(f"📂 Loaded {len(segments)} RT interface segments")
    print(f"📏 Domain scale: {domain_scale}m")

    # Initialize advanced characterization engine
    engine = AdvancedCharacterizationEngine()

    # Perform comprehensive analysis
    start_time = time.time()
    results = engine.analyze_comprehensive(segments, domain_scale)
    total_time = time.time() - start_time

    # Compare with known successful parameters
    print(f"\n✅ COMPARISON WITH KNOWN SUCCESSFUL PARAMETERS")
    print("-" * 70)

    known_successful = {
        'initial_delta': 0.020000,
        'delta_factor': 1.263,
        'num_steps': 15
    }

    consensus = results['consensus_parameters']

    def calc_error(predicted, actual):
        return abs(predicted - actual) / actual * 100 if actual != 0 else float('inf')

    delta_error = calc_error(consensus['initial_delta'], known_successful['initial_delta'])
    factor_error = calc_error(consensus['delta_factor'], known_successful['delta_factor'])
    steps_error = calc_error(consensus['num_steps'], known_successful['num_steps'])

    print(f"Parameter      | Predicted    | Actual       | Error")
    print("-" * 60)
    print(f"δ₀             | {consensus['initial_delta']:<12.6f} | {known_successful['initial_delta']:<12.6f} | {delta_error:<.1f}%")
    print(f"Factor         | {consensus['delta_factor']:<12.3f} | {known_successful['delta_factor']:<12.3f} | {factor_error:<.1f}%")
    print(f"Steps          | {consensus['num_steps']:<12.0f} | {known_successful['num_steps']:<12.0f} | {steps_error:<.1f}%")

    avg_error = (delta_error + factor_error + steps_error) / 3
    print(f"\nOverall accuracy: {100 - avg_error:.1f}%")

    if avg_error < 50:
        print("🎉 Advanced characterization achieved good accuracy!")
    elif avg_error < 100:
        print("📊 Advanced characterization achieved moderate accuracy")
    else:
        print("⚠️  Advanced characterization needs calibration")

    return results


if __name__ == "__main__":
    # Test the integrated system
    test_advanced_characterization_on_rt_interface()