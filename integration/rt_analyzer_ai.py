#!/usr/bin/env python3
"""
AI-Enhanced RT Analyzer Integration

This module integrates the FractalParameterAI framework with RT interface analysis.
It can work as either:
1. A wrapper around the existing rt_analyzer.py (initial approach)
2. A full reimplementation using AI Framework directly (future evolution)

Key enhancements:
- Automatic parameter selection based on interface characteristics
- Learning from analysis results to improve future RT analyses
- Scale-aware parameter normalization
- Interface type classification (RT interfaces → complex_fractal or moderate_fractal)
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

# Add paths for dependencies
FAST_FRACTAL_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer'
FRACTAL_ANALYZER_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalAnalyzer'

# Add both to path BEFORE any imports
if FAST_FRACTAL_PATH not in sys.path:
    sys.path.insert(0, FAST_FRACTAL_PATH)
if FRACTAL_ANALYZER_PATH not in sys.path:
    sys.path.insert(0, FRACTAL_ANALYZER_PATH)

# Import AI Framework components
from core.interface_features import (
    extract_interface_features,
    classify_interface_type,
    suggest_optimal_parameters_adaptive,
    record_parameter_feedback
)

# Import FastFractalAnalyzer
from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode

# Import original RT analyzer (for wrapper mode)
ORIGINAL_RT_AVAILABLE = False
OriginalRTAnalyzer = None

try:
    # Direct import using importlib since package structure might be complex
    import importlib.util
    rt_analyzer_path = os.path.join(FRACTAL_ANALYZER_PATH, 'fractal_analyzer', 'core', 'rt_analyzer.py')

    if os.path.exists(rt_analyzer_path):
        spec = importlib.util.spec_from_file_location("rt_analyzer_module", rt_analyzer_path)
        rt_analyzer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt_analyzer_module)
        OriginalRTAnalyzer = rt_analyzer_module.RTAnalyzer
        ORIGINAL_RT_AVAILABLE = True
        print("✅ Original rt_analyzer loaded successfully (direct file import)")
    else:
        raise ImportError(f"rt_analyzer.py not found at {rt_analyzer_path}")

except Exception as e:
    print(f"⚠️  Original rt_analyzer not available: {e}")
    print("   Operating in standalone mode")
    ORIGINAL_RT_AVAILABLE = False


class AIEnhancedRTAnalyzer:
    """
    AI-enhanced RT analyzer that uses the FractalParameterAI framework.

    Can operate in two modes:
    1. Wrapper mode: Enhances existing rt_analyzer with AI parameter selection
    2. Standalone mode: Direct AI-based analysis without original rt_analyzer
    """

    def __init__(self, output_dir="./rt_analysis_ai",
                 performance_mode=PerformanceMode.FAST,
                 use_wrapper=True,
                 enable_learning=True,
                 debug=False):
        """
        Initialize AI-enhanced RT analyzer.

        Args:
            output_dir: Directory for analysis outputs
            performance_mode: FastFractalAnalyzer performance mode
            use_wrapper: If True, use original rt_analyzer for interface extraction
            enable_learning: If True, record feedback for AI learning
            debug: Enable debug output
        """
        self.output_dir = output_dir
        self.performance_mode = performance_mode
        self.use_wrapper = use_wrapper and ORIGINAL_RT_AVAILABLE
        self.enable_learning = enable_learning
        self.debug = debug

        os.makedirs(output_dir, exist_ok=True)

        # Initialize FastFractalAnalyzer
        self.fractal_analyzer = FastFractalAnalyzer(performance_mode=performance_mode)

        # Initialize original RT analyzer if in wrapper mode
        self.rt_analyzer = None
        if self.use_wrapper:
            self.rt_analyzer = OriginalRTAnalyzer(
                output_dir=output_dir,
                use_grid_optimization=False,  # We handle this in FastFractalAnalyzer
                no_titles=True
            )
            print("✅ Operating in WRAPPER mode - using original rt_analyzer for interface extraction")
        else:
            print("✅ Operating in STANDALONE mode - direct AI analysis")

        print(f"✨ AI Framework active with {performance_mode.name} performance mode")
        if enable_learning:
            print("📚 Learning enabled - results will improve over time")

    def _convert_rt_interface_to_segments(self, interface_points: List[Tuple]) -> np.ndarray:
        """
        Convert RT interface points to segment array format.

        Args:
            interface_points: List of (x, y) tuples from RT interface extraction

        Returns:
            numpy array of shape (n_segments, 4) with [x1, y1, x2, y2] format
        """
        segments = []
        for i in range(len(interface_points) - 1):
            x1, y1 = interface_points[i]
            x2, y2 = interface_points[i + 1]
            segments.append([x1, y1, x2, y2])

        return np.array(segments)

    def analyze_rt_interface(self, interface_points: List[Tuple],
                            theoretical_dimension: Optional[float] = None,
                            metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze RT interface with AI-enhanced parameter selection.

        Args:
            interface_points: List of (x, y) interface coordinates
            theoretical_dimension: Known dimension for validation (optional)
            metadata: Additional metadata (time, resolution, etc.)

        Returns:
            Dictionary with analysis results including dimension, parameters, and metadata
        """
        if len(interface_points) < 2:
            return {
                'success': False,
                'error': 'Insufficient interface points',
                'dimension': np.nan
            }

        print(f"\n{'='*60}")
        print(f"🔬 AI-ENHANCED RT INTERFACE ANALYSIS")
        print(f"{'='*60}")

        # Convert to segments
        segments_array = self._convert_rt_interface_to_segments(interface_points)
        print(f"📊 Interface: {len(interface_points)} points → {len(segments_array)} segments")

        # Extract features
        print("🔍 Extracting interface features...")
        features = extract_interface_features(segments_array)
        interface_type = classify_interface_type(features)

        print(f"   🏷️  Interface type: {interface_type}")
        print(f"   📏 Characteristic length: {features['characteristic_length']:.6f}")
        print(f"   🌀 Tortuosity: {features['tortuosity']:.3f}")
        print(f"   📊 Complexity score: {features['complexity_score']:.3f}")
        print(f"   🔗 Connectivity: {features['connectivity_ratio']:.1%}")

        # Get AI-suggested parameters
        print("\n🎯 AI Parameter Selection...")
        suggested_params = suggest_optimal_parameters_adaptive(
            features,
            implementation="basic_box_counting"
        )

        print(f"   🔧 Suggested parameters:")
        print(f"      initial_delta: {suggested_params['initial_delta']:.6f}")
        print(f"      delta_factor: {suggested_params['delta_factor']:.3f}")
        print(f"      num_steps: {int(suggested_params['num_steps'])}")

        # Convert to SegmentArray for FastFractalAnalyzer
        # SegmentArray expects shape (n_segments, 2, 2) where each segment is [[x1,y1], [x2,y2]]
        segments_reshaped = segments_array.reshape(-1, 2, 2)
        segment_array_obj = SegmentArray(segments_reshaped)

        # Run fractal analysis
        print(f"\n🏁 Running fractal dimension analysis...")
        start_time = time.time()

        result = self.fractal_analyzer.analyze_segments(
            segment_array_obj,
            initial_delta=suggested_params['initial_delta'],
            delta_factor=suggested_params['delta_factor'],
            num_steps=int(suggested_params['num_steps'])
        )

        analysis_time = time.time() - start_time

        if result and hasattr(result, 'dimension'):
            dimension = result.dimension
            r_squared = result.r_squared

            print(f"\n{'='*60}")
            print(f"🎉 RESULTS:")
            print(f"{'='*60}")
            print(f"📐 Fractal Dimension: {dimension:.6f}")
            print(f"📈 R² Quality: {r_squared:.6f}")
            print(f"⏱️  Analysis Time: {analysis_time:.1f}s")

            if theoretical_dimension is not None:
                error = abs(dimension - theoretical_dimension) / theoretical_dimension * 100
                print(f"🎯 Error vs. theoretical ({theoretical_dimension:.3f}): {error:.2f}%")

            # Quality assessment
            if r_squared > 0.99:
                print("✅ Excellent quality - very high confidence")
            elif r_squared > 0.95:
                print("✅ Good quality - high confidence")
            elif r_squared > 0.90:
                print("⚠️  Fair quality - moderate confidence")
            else:
                print("❌ Poor quality - low confidence")

            # Record feedback for learning
            if self.enable_learning:
                print("\n📝 Recording feedback for AI learning...")
                record_parameter_feedback(
                    features=features,
                    suggested_parameters=suggested_params,
                    dimension_result=dimension,
                    theoretical_dimension=theoretical_dimension,
                    r_squared=r_squared,
                    computation_time=analysis_time
                )
                print("   ✅ Feedback recorded - AI will improve from this analysis")

            # Compile results
            results = {
                'success': True,
                'dimension': dimension,
                'r_squared': r_squared,
                'analysis_time': analysis_time,
                'interface_type': interface_type,
                'suggested_parameters': suggested_params,
                'features': features,
                'n_points': len(interface_points),
                'n_segments': len(segments_array)
            }

            if metadata:
                results['metadata'] = metadata

            if theoretical_dimension is not None:
                results['theoretical_dimension'] = theoretical_dimension
                results['error_percent'] = error

            return results
        else:
            print("❌ Analysis failed - no valid result")
            return {
                'success': False,
                'error': 'Fractal analysis failed',
                'dimension': np.nan
            }

    def analyze_vtk_file_ai(self, vtk_file_path: str,
                           theoretical_dimension: Optional[float] = None) -> Dict:
        """
        Analyze VTK file with AI-enhanced fractal dimension calculation.

        This is the main entry point for RT VTK file analysis.

        Args:
            vtk_file_path: Path to RT simulation VTK file
            theoretical_dimension: Known dimension for validation (optional)

        Returns:
            Dictionary with complete analysis results
        """
        if not self.use_wrapper:
            raise NotImplementedError(
                "Standalone VTK parsing not yet implemented. "
                "Use use_wrapper=True to leverage original rt_analyzer."
            )

        print(f"\n🚀 AI-Enhanced VTK Analysis")
        print(f"📂 File: {os.path.basename(vtk_file_path)}")

        # Use original rt_analyzer to extract interface
        print("🔄 Extracting interface (using original rt_analyzer)...")
        extraction_start = time.time()

        # Call original analyzer to get interface points
        original_results = self.rt_analyzer.analyze_vtk_file(
            vtk_file_path,
            analysis_types=['fractal_dim'],  # We'll replace the dimension calc
            min_box_size=None
        )

        if original_results is None or 'fractal_dimension' not in original_results:
            print("❌ Interface extraction failed")
            return {'success': False, 'error': 'Interface extraction failed'}

        # Get interface points from cache
        # Note: We need to access the interface data from rt_analyzer's cache
        # This is a bit of a hack - in standalone mode, we'd parse VTK directly
        cache_key = self.rt_analyzer._generate_cache_key(vtk_file_path)
        cached_data = self.rt_analyzer.interface_cache.get(cache_key)

        if cached_data and 'base_interface' in cached_data:
            interface_points = cached_data['base_interface']
            extraction_time = time.time() - extraction_start
            print(f"✅ Interface extracted: {len(interface_points)} points ({extraction_time:.1f}s)")

            # Extract metadata
            metadata = {
                'vtk_file': vtk_file_path,
                'time': original_results.get('time', 0.0),
                'extraction_time': extraction_time,
                'original_dimension': original_results.get('fractal_dimension', np.nan)
            }

            # Run AI-enhanced analysis
            ai_results = self.analyze_rt_interface(
                interface_points,
                theoretical_dimension=theoretical_dimension,
                metadata=metadata
            )

            # Add comparison with original method
            if 'original_dimension' in metadata and ai_results['success']:
                original_dim = metadata['original_dimension']
                if not np.isnan(original_dim):
                    diff = abs(ai_results['dimension'] - original_dim)
                    print(f"\n📊 Comparison with original rt_analyzer:")
                    print(f"   Original dimension: {original_dim:.6f}")
                    print(f"   AI dimension: {ai_results['dimension']:.6f}")
                    print(f"   Difference: {diff:.6f}")

            return ai_results
        else:
            print("❌ Could not retrieve interface from cache")
            return {'success': False, 'error': 'Interface cache miss'}


def main():
    """Example usage of AI-enhanced RT analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description='AI-Enhanced RT Interface Analysis')
    parser.add_argument('vtk_file', help='Path to RT VTK file')
    parser.add_argument('--theoretical-dim', type=float, help='Theoretical dimension for validation')
    parser.add_argument('--output-dir', default='./rt_analysis_ai', help='Output directory')
    parser.add_argument('--performance', choices=['fast', 'balanced', 'accurate'],
                       default='fast', help='Performance mode')
    parser.add_argument('--no-learning', action='store_true', help='Disable AI learning')
    parser.add_argument('--standalone', action='store_true',
                       help='Use standalone mode (no original rt_analyzer)')

    args = parser.parse_args()

    # Map performance string to enum
    perf_map = {
        'fast': PerformanceMode.FAST,
        'balanced': PerformanceMode.BALANCED,
        'accurate': PerformanceMode.ACCURATE
    }

    # Create analyzer
    analyzer = AIEnhancedRTAnalyzer(
        output_dir=args.output_dir,
        performance_mode=perf_map[args.performance],
        use_wrapper=not args.standalone,
        enable_learning=not args.no_learning
    )

    # Run analysis
    results = analyzer.analyze_vtk_file_ai(
        args.vtk_file,
        theoretical_dimension=args.theoretical_dim
    )

    if results['success']:
        print(f"\n✅ Analysis complete!")
        print(f"   Dimension: {results['dimension']:.6f}")
        print(f"   R²: {results['r_squared']:.6f}")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
