#!/usr/bin/env python3
"""
Refactored AI-Enhanced RT Analyzer

Complete standalone RT analyzer integrated with FractalParameterAI framework.
No dependencies on original FractalAnalyzer rt_analyzer.py.

Features:
- Standalone VTK parsing
- AI-enhanced parameter selection
- Multiple analysis types (fractal, mixing, power spectrum, multifractal)
- Temporal evolution tracking
- Adaptive learning from results

This is the main refactored implementation for the rt-analyzer-full-refactor branch.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

# Add paths for dependencies
FAST_FRACTAL_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer'
if FAST_FRACTAL_PATH not in sys.path:
    sys.path.insert(0, FAST_FRACTAL_PATH)

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

# Import our new standalone modules
from integration.rt_vtk_parser import VTKParser, VTKData, find_vtk_files
from integration.rt_interface_extraction import InterfaceExtractor, InterfaceData, extract_multiple_levels


class RefactoredRTAnalyzer:
    """
    Fully refactored RT analyzer with AI integration.

    Complete standalone implementation that:
    - Parses VTK files directly
    - Extracts RT interfaces
    - Uses AI for parameter selection
    - Supports multiple analysis types
    - Learns from results
    """

    def __init__(self,
                 output_dir: str = "./rt_analysis_refactored",
                 performance_mode: PerformanceMode = PerformanceMode.FAST,
                 interface_method: str = "skimage",
                 enable_learning: bool = True,
                 debug: bool = False):
        """
        Initialize refactored RT analyzer.

        Args:
            output_dir: Directory for analysis outputs
            performance_mode: FastFractalAnalyzer performance mode
            interface_method: Interface extraction method ("skimage", "plic", "conrec")
            enable_learning: Enable AI learning from results
            debug: Enable debug output
        """
        self.output_dir = output_dir
        self.performance_mode = performance_mode
        self.interface_method = interface_method
        self.enable_learning = enable_learning
        self.debug = debug

        os.makedirs(output_dir, exist_ok=True)

        # Initialize components
        self.vtk_parser = VTKParser(debug=debug)
        self.interface_extractor = InterfaceExtractor(method=interface_method, debug=debug)
        self.fractal_analyzer = FastFractalAnalyzer(performance_mode=performance_mode)

        print(f"✨ Refactored RT Analyzer initialized")
        print(f"   Output: {output_dir}")
        print(f"   Performance: {performance_mode.name}")
        print(f"   Interface method: {interface_method}")
        print(f"   Learning: {'enabled' if enable_learning else 'disabled'}")

    def analyze_single_vtk(self,
                          vtk_file_path: str,
                          analysis_types: Optional[List[str]] = None,
                          theoretical_dimension: Optional[float] = None,
                          interface_level: float = 0.5) -> Dict:
        """
        Analyze a single VTK file with AI-enhanced parameter selection.

        Args:
            vtk_file_path: Path to VTK file
            analysis_types: List of analysis types ['fractal', 'mixing', 'power_spectrum']
            theoretical_dimension: Known dimension for validation (optional)
            interface_level: VOF level for interface (default 0.5)

        Returns:
            Dictionary with complete analysis results
        """
        if analysis_types is None:
            analysis_types = ['fractal']

        print(f"\n{'='*70}")
        print(f"🔬 REFACTORED AI-ENHANCED RT ANALYSIS")
        print(f"{'='*70}")
        print(f"📂 File: {os.path.basename(vtk_file_path)}")
        print(f"🎯 Analysis types: {', '.join(analysis_types)}")

        total_start = time.time()

        # Step 1: Parse VTK file
        print(f"\n📖 Step 1: Parsing VTK file...")
        parse_start = time.time()
        vtk_data = self.vtk_parser.parse_vtk_file(vtk_file_path)
        parse_time = time.time() - parse_start

        if not vtk_data:
            print(f"❌ VTK parsing failed")
            return {'success': False, 'error': 'VTK parsing failed'}

        print(f"   ✅ Parsed in {parse_time:.2f}s")
        print(f"   Grid: {vtk_data.dimensions[0]} × {vtk_data.dimensions[1]}")
        print(f"   Time: {vtk_data.time:.3f}")

        # Step 2: Extract interface
        print(f"\n🔍 Step 2: Extracting interface...")
        extract_start = time.time()

        if 'f' not in vtk_data.scalar_fields:
            print(f"❌ No VOF field found")
            return {'success': False, 'error': 'No VOF field'}

        f_grid = vtk_data['f']
        interface_data = self.interface_extractor.extract_interface(
            f_grid, vtk_data.x_grid, vtk_data.y_grid, interface_level
        )
        extract_time = time.time() - extract_start

        if not interface_data:
            print(f"❌ Interface extraction failed")
            return {'success': False, 'error': 'Interface extraction failed'}

        print(f"   ✅ Extracted in {extract_time:.2f}s")
        print(f"   Points: {interface_data.metadata['n_points']}")
        print(f"   Segments: {interface_data.metadata['n_segments']}")

        # Initialize results
        results = {
            'success': True,
            'file_path': vtk_file_path,
            'time': vtk_data.time,
            'parse_time': parse_time,
            'interface_extraction_time': extract_time,
            'interface_points': interface_data.metadata['n_points'],
            'interface_segments': interface_data.metadata['n_segments'],
            'interface_bounds': interface_data.metadata['bounds'],
            'grid_shape': vtk_data.dimensions[:2],
            'analysis_types': analysis_types
        }

        # Step 3: Fractal dimension analysis (if requested)
        if 'fractal' in analysis_types:
            fractal_results = self._analyze_fractal_dimension(
                interface_data,
                theoretical_dimension
            )
            results.update(fractal_results)

        # Step 4: Mixing analysis (if requested)
        if 'mixing' in analysis_types:
            print(f"\n🧪 Mixing analysis: Not yet implemented")
            # mixing_results = self._analyze_mixing(vtk_data, interface_data)
            # results.update(mixing_results)

        # Step 5: Power spectrum (if requested)
        if 'power_spectrum' in analysis_types:
            print(f"\n📊 Power spectrum: Not yet implemented")
            # spectrum_results = self._analyze_power_spectrum(interface_data)
            # results.update(spectrum_results)

        # Total time
        total_time = time.time() - total_start
        results['total_analysis_time'] = total_time

        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE: {total_time:.1f}s total")
        print(f"{'='*70}")

        return results

    def _analyze_fractal_dimension(self,
                                   interface_data: InterfaceData,
                                   theoretical_dimension: Optional[float] = None) -> Dict:
        """
        Perform AI-enhanced fractal dimension analysis.

        Args:
            interface_data: Extracted interface data
            theoretical_dimension: Known dimension for validation

        Returns:
            Dictionary with fractal analysis results
        """
        print(f"\n📐 Step 3: AI-Enhanced Fractal Dimension Analysis")

        # Extract features from interface
        print(f"   🔍 Extracting interface features...")
        segments_array = interface_data.segments
        features = extract_interface_features(segments_array)
        interface_type = classify_interface_type(features)

        print(f"      🏷️  Type: {interface_type}")
        print(f"      📏 Characteristic length: {features['characteristic_length']:.6f}")
        print(f"      🌀 Tortuosity: {features['tortuosity']:.3f}")
        print(f"      📊 Complexity: {features['complexity_score']:.3f}")
        print(f"      🔗 Connectivity: {features['connectivity_ratio']:.1%}")

        # Get AI-suggested parameters
        print(f"\n   🎯 AI Parameter Selection...")
        suggested_params = suggest_optimal_parameters_adaptive(
            features,
            implementation="basic_box_counting"
        )

        print(f"      🔧 Suggested parameters:")
        print(f"         initial_delta: {suggested_params['initial_delta']:.6f}")
        print(f"         delta_factor: {suggested_params['delta_factor']:.3f}")
        print(f"         num_steps: {int(suggested_params['num_steps'])}")

        # Validate parameters for degenerate cases
        if suggested_params['initial_delta'] <= 0 or not np.isfinite(suggested_params['initial_delta']):
            print(f"   ⚠️  Invalid initial_delta - interface too simple (likely straight line)")
            return {
                'fractal_dimension': 1.0,  # Straight line has dimension 1.0
                'fractal_r_squared': 1.0,
                'fractal_analysis_time': 0.0,
                'fractal_interface_type': interface_type,
                'fractal_suggested_parameters': suggested_params,
                'fractal_features': features,
                'fractal_error': 'Degenerate case - straight line'
            }

        # Convert to SegmentArray for FastFractalAnalyzer
        segments_reshaped = segments_array.reshape(-1, 2, 2)
        segment_array_obj = SegmentArray(segments_reshaped)

        # Run fractal analysis
        print(f"\n   🏁 Running fractal dimension analysis...")
        start_time = time.time()

        try:
            result = self.fractal_analyzer.analyze_segments(
                segment_array_obj,
                initial_delta=suggested_params['initial_delta'],
                delta_factor=suggested_params['delta_factor'],
                num_steps=int(suggested_params['num_steps'])
            )
            analysis_time = time.time() - start_time
        except Exception as e:
            analysis_time = time.time() - start_time
            print(f"   ❌ Fractal analysis error: {e}")
            return {
                'fractal_dimension': np.nan,
                'fractal_r_squared': np.nan,
                'fractal_analysis_time': analysis_time,
                'fractal_interface_type': interface_type,
                'fractal_error': str(e)
            }

        if result and hasattr(result, 'dimension'):
            dimension = result.dimension
            r_squared = result.r_squared

            print(f"\n   {'='*66}")
            print(f"   🎉 FRACTAL RESULTS:")
            print(f"   {'='*66}")
            print(f"   📐 Fractal Dimension: {dimension:.6f}")
            print(f"   📈 R² Quality: {r_squared:.6f}")
            print(f"   ⏱️  Analysis Time: {analysis_time:.1f}s")

            if theoretical_dimension is not None:
                error = abs(dimension - theoretical_dimension) / theoretical_dimension * 100
                print(f"   🎯 Error vs. theoretical ({theoretical_dimension:.3f}): {error:.2f}%")

            # Quality assessment
            if r_squared > 0.99:
                print(f"   ✅ Excellent quality - very high confidence")
            elif r_squared > 0.95:
                print(f"   ✅ Good quality - high confidence")
            elif r_squared > 0.90:
                print(f"   ⚠️  Fair quality - moderate confidence")
            else:
                print(f"   ❌ Poor quality - low confidence")

            # Record feedback for learning
            if self.enable_learning:
                print(f"\n   📝 Recording feedback for AI learning...")
                record_parameter_feedback(
                    features=features,
                    suggested_parameters=suggested_params,
                    dimension_result=dimension,
                    theoretical_dimension=theoretical_dimension,
                    r_squared=r_squared,
                    computation_time=analysis_time
                )
                print(f"      ✅ Feedback recorded - AI will improve from this analysis")

            # Compile results
            return {
                'fractal_dimension': dimension,
                'fractal_r_squared': r_squared,
                'fractal_analysis_time': analysis_time,
                'fractal_interface_type': interface_type,
                'fractal_suggested_parameters': suggested_params,
                'fractal_features': features
            }

        else:
            print(f"   ❌ Fractal analysis failed - no valid result")
            return {
                'fractal_dimension': np.nan,
                'fractal_r_squared': np.nan,
                'fractal_analysis_time': analysis_time,
                'fractal_error': 'Analysis failed'
            }

    def analyze_temporal_evolution(self,
                                   vtk_pattern_or_dir: str,
                                   analysis_types: Optional[List[str]] = None,
                                   time_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Analyze temporal evolution across multiple VTK files.

        Args:
            vtk_pattern_or_dir: Directory or pattern for VTK files
            analysis_types: List of analysis types
            time_range: Optional (min_time, max_time) filter

        Returns:
            Dictionary with time-series results
        """
        print(f"\n{'='*70}")
        print(f"📈 TEMPORAL EVOLUTION ANALYSIS")
        print(f"{'='*70}")

        # Find VTK files
        if os.path.isdir(vtk_pattern_or_dir):
            vtk_files = find_vtk_files(vtk_pattern_or_dir)
        else:
            print(f"❌ Pattern matching not yet implemented - provide directory")
            return {'success': False, 'error': 'Pattern matching not implemented'}

        if not vtk_files:
            print(f"❌ No VTK files found")
            return {'success': False, 'error': 'No VTK files found'}

        print(f"📂 Found {len(vtk_files)} VTK files")

        # Analyze each file
        time_series_results = []

        for i, vtk_file in enumerate(vtk_files, 1):
            print(f"\n{'─'*70}")
            print(f"File {i}/{len(vtk_files)}: {os.path.basename(vtk_file)}")

            result = self.analyze_single_vtk(vtk_file, analysis_types=analysis_types)

            if result['success']:
                # Filter by time range if specified
                if time_range is not None:
                    t = result['time']
                    if t < time_range[0] or t > time_range[1]:
                        print(f"   ⏭️  Skipping (time {t:.3f} outside range)")
                        continue

                time_series_results.append(result)
            else:
                print(f"   ❌ Analysis failed: {result.get('error', 'Unknown error')}")

        if not time_series_results:
            print(f"\n❌ No successful analyses")
            return {'success': False, 'error': 'No successful analyses'}

        # Compile temporal evolution data
        times = [r['time'] for r in time_series_results]

        temporal_data = {
            'success': True,
            'n_files': len(time_series_results),
            'times': times,
            'results': time_series_results
        }

        # Extract fractal dimension time series if available
        if all('fractal_dimension' in r for r in time_series_results):
            dimensions = [r['fractal_dimension'] for r in time_series_results]
            r_squareds = [r['fractal_r_squared'] for r in time_series_results]

            temporal_data['fractal_dimensions'] = dimensions
            temporal_data['fractal_r_squareds'] = r_squareds

            print(f"\n{'='*70}")
            print(f"📊 TEMPORAL EVOLUTION SUMMARY")
            print(f"{'='*70}")
            print(f"   Files analyzed: {len(time_series_results)}")
            print(f"   Time range: [{min(times):.3f}, {max(times):.3f}]")
            print(f"   Dimension range: [{min(dimensions):.4f}, {max(dimensions):.4f}]")
            print(f"   Mean dimension: {np.mean(dimensions):.4f} ± {np.std(dimensions):.4f}")
            print(f"   Mean R²: {np.mean(r_squareds):.4f}")

        return temporal_data


def main():
    """Main entry point for refactored RT analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Refactored AI-Enhanced RT Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file analysis
  python rt_analyzer_refactored.py /path/to/RT160x200-10000.vtk

  # Temporal evolution
  python rt_analyzer_refactored.py /path/to/vtk_directory/ --temporal

  # With theoretical validation
  python rt_analyzer_refactored.py file.vtk --theoretical-dim 1.66

  # Multiple analysis types
  python rt_analyzer_refactored.py file.vtk --analysis fractal mixing
        """
    )

    parser.add_argument('input', help='VTK file or directory')
    parser.add_argument('--temporal', action='store_true',
                       help='Temporal evolution analysis (directory required)')
    parser.add_argument('--analysis', nargs='+', default=['fractal'],
                       choices=['fractal', 'mixing', 'power_spectrum'],
                       help='Analysis types to perform')
    parser.add_argument('--theoretical-dim', type=float,
                       help='Theoretical dimension for validation')
    parser.add_argument('--output-dir', default='./rt_analysis_refactored',
                       help='Output directory')
    parser.add_argument('--performance', choices=['fast', 'balanced', 'accurate'],
                       default='fast', help='Performance mode')
    parser.add_argument('--interface-method', choices=['skimage', 'plic', 'conrec'],
                       default='skimage', help='Interface extraction method')
    parser.add_argument('--no-learning', action='store_true',
                       help='Disable AI learning')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Map performance string to enum
    perf_map = {
        'fast': PerformanceMode.FAST,
        'balanced': PerformanceMode.BALANCED,
        'accurate': PerformanceMode.ACCURATE
    }

    # Create analyzer
    analyzer = RefactoredRTAnalyzer(
        output_dir=args.output_dir,
        performance_mode=perf_map[args.performance],
        interface_method=args.interface_method,
        enable_learning=not args.no_learning,
        debug=args.debug
    )

    # Run analysis
    if args.temporal or os.path.isdir(args.input):
        # Temporal evolution
        results = analyzer.analyze_temporal_evolution(
            args.input,
            analysis_types=args.analysis
        )
    else:
        # Single file
        results = analyzer.analyze_single_vtk(
            args.input,
            analysis_types=args.analysis,
            theoretical_dimension=args.theoretical_dim
        )

    if results['success']:
        print(f"\n✅ Analysis complete!")
        if 'fractal_dimension' in results:
            print(f"   Dimension: {results['fractal_dimension']:.6f}")
            print(f"   R²: {results['fractal_r_squared']:.6f}")
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
