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
import json
import csv
from datetime import datetime

# Add FractalParameterAI root to path (allows running from anywhere)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRACTAL_PARAM_AI_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from integration/ to FractalParameterAI/
if FRACTAL_PARAM_AI_ROOT not in sys.path:
    sys.path.insert(0, FRACTAL_PARAM_AI_ROOT)

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
from integration.rt_vtk_parser import VTKParser, VTKData, find_vtk_files, find_vtk_files_with_pattern
from integration.rt_interface_extraction import InterfaceExtractor, InterfaceData, extract_multiple_levels
from integration.rt_classification import RTClassifier, compute_growth_rate
from integration.rt_power_spectrum import PowerSpectrumAnalyzer
from integration.rt_spectrum_plotting import SpectrumPlotter
from integration.rt_velocity_statistics import VelocityAnalyzer, VelocityStatistics
from integration.rt_mixing_analysis import MixingAnalyzer, MixingStatistics
from integration.rt_multifractal_analysis import MultifractalAnalyzer, MultifractalSpectrum


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
                 interface_method: str = "conrec",
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
        self.rt_classifier = RTClassifier(debug=debug)
        self.power_spectrum_analyzer = PowerSpectrumAnalyzer(debug=debug)
        self.velocity_analyzer = VelocityAnalyzer(debug=debug)
        self.mixing_analyzer = MixingAnalyzer(debug=debug)
        self.multifractal_analyzer = MultifractalAnalyzer(debug=debug)

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
                vtk_data,
                theoretical_dimension
            )
            results.update(fractal_results)

        # Step 4: RT classification (if requested)
        if 'rt_classification' in analysis_types:
            rt_results = self._analyze_rt_classification(interface_data)
            results['rt_classification'] = rt_results

        # Step 5: Mixing analysis (if requested)
        if 'mixing' in analysis_types:
            mixing_results = self._analyze_mixing(vtk_data)
            results['mixing'] = mixing_results

        # Step 6: Power spectrum (if requested)
        if 'power_spectrum' in analysis_types:
            spectrum_results = self._analyze_power_spectrum(vtk_data, interface_data)
            results['power_spectrum'] = spectrum_results

        # Step 7: Velocity statistics (if requested)
        if 'velocity' in analysis_types:
            velocity_results = self._analyze_velocity_statistics(vtk_data)
            results['velocity'] = velocity_results

        # Step 8: Multifractal analysis (if requested)
        if 'multifractal' in analysis_types:
            multifractal_results = self._analyze_multifractal(interface_data)
            results['multifractal'] = multifractal_results

        # Total time
        total_time = time.time() - total_start
        results['total_analysis_time'] = total_time

        print(f"\n{'='*70}")
        print(f"✅ ANALYSIS COMPLETE: {total_time:.1f}s total")
        print(f"{'='*70}")

        return results

    def _analyze_fractal_dimension(self,
                                   interface_data: InterfaceData,
                                   vtk_data: VTKData,
                                   theoretical_dimension: Optional[float] = None) -> Dict:
        """
        Perform AI-enhanced fractal dimension analysis.

        Args:
            interface_data: Extracted interface data
            vtk_data: VTK data (for grid spacing)
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
        print(f"      📐 Linearity R²: {features['linearity_r_squared']:.3f}")
        print(f"      🔄 Direction change rate: {features['direction_change_rate']:.3f}")
        print(f"      🔗 Connectivity: {features['connectivity_ratio']:.1%}")

        # Get AI-suggested parameters
        print(f"\n   🎯 AI Parameter Selection...")
        suggested_params = suggest_optimal_parameters_adaptive(
            features,
            implementation="basic_box_counting"
        )

        # Adjust num_steps and delta_factor based on interface complexity to avoid excessive computation
        # For smooth interfaces (D~1.0), we don't need as many scales as for rough fractals (D~1.5+)
        complexity = features['complexity_score']
        base_num_steps = suggested_params['num_steps']
        base_delta_factor = suggested_params['delta_factor']

        if complexity < 0.5:
            # Very smooth interface - reduce steps by 50%, increase delta_factor
            adjusted_num_steps = max(6, int(base_num_steps * 0.5))
            adjusted_delta_factor = max(base_delta_factor, 2.0)  # Use larger scaling
            print(f"      ⚡ Low complexity ({complexity:.2f}) - aggressive optimization for efficiency")
            print(f"         Reducing num_steps: {base_num_steps} → {adjusted_num_steps}")
            print(f"         Increasing delta_factor: {base_delta_factor:.1f} → {adjusted_delta_factor:.1f}")
        elif complexity < 0.75:
            # Moderately smooth - reduce steps by 40%, increase delta_factor moderately
            adjusted_num_steps = max(8, int(base_num_steps * 0.6))
            adjusted_delta_factor = max(base_delta_factor, 1.8)  # Slightly larger scaling
            print(f"      ⚡ Moderate complexity ({complexity:.2f}) - optimizing for smooth interface")
            print(f"         Reducing num_steps: {base_num_steps} → {adjusted_num_steps}")
            if adjusted_delta_factor > base_delta_factor:
                print(f"         Increasing delta_factor: {base_delta_factor:.1f} → {adjusted_delta_factor:.1f}")
        else:
            # Complex interface - use full suggested parameters
            adjusted_num_steps = int(base_num_steps)
            adjusted_delta_factor = base_delta_factor

        suggested_params['num_steps'] = adjusted_num_steps
        suggested_params['delta_factor'] = adjusted_delta_factor

        # Grid-aware parameter adjustment:
        # For near-straight interfaces, CONREC produces artificially small segments
        # This leads to unreasonably small initial_delta (smaller than grid resolution)
        # Enforce minimum initial_delta based on computational grid spacing
        #
        # Estimate grid spacing from coordinate grids
        dx_samples = np.abs(np.diff(vtk_data.x_grid[:min(10, vtk_data.x_grid.shape[0]), 0]))
        dy_samples = np.abs(np.diff(vtk_data.y_grid[0, :min(10, vtk_data.y_grid.shape[1])]))

        dx_samples = dx_samples[dx_samples > 0]
        dy_samples = dy_samples[dy_samples > 0]

        if len(dx_samples) > 0 and len(dy_samples) > 0:
            typical_dx = np.median(dx_samples)
            typical_dy = np.median(dy_samples)
            grid_spacing = min(typical_dx, typical_dy)

            # Minimum initial_delta should be at least 2× grid spacing
            # For smooth interfaces (complexity < 0.5), use even larger minimum (4× grid)
            if complexity < 0.5:
                min_initial_delta = grid_spacing * 4.0
            else:
                min_initial_delta = grid_spacing * 2.0

            # Apply grid-aware constraint
            original_initial_delta = suggested_params['initial_delta']
            if original_initial_delta < min_initial_delta:
                suggested_params['initial_delta'] = min_initial_delta
                print(f"      🔧 Grid-aware adjustment:")
                print(f"         Grid spacing: {grid_spacing:.6f}")
                print(f"         Original initial_delta: {original_initial_delta:.6f} ({original_initial_delta/grid_spacing:.2f}× grid)")
                print(f"         Adjusted initial_delta: {min_initial_delta:.6f} ({min_initial_delta/grid_spacing:.1f}× grid)")
                print(f"         Reason: Box sizes smaller than grid cannot resolve physical features")

        print(f"      🔧 Final parameters:")
        print(f"         initial_delta: {suggested_params['initial_delta']:.6f}")
        print(f"         delta_factor: {suggested_params['delta_factor']:.3f}")
        print(f"         num_steps: {suggested_params['num_steps']}")

        # Validate parameters for degenerate cases
        # Check 1: Invalid or extremely small initial_delta (< 1e-6)
        if (suggested_params['initial_delta'] <= 0 or
            not np.isfinite(suggested_params['initial_delta']) or
            suggested_params['initial_delta'] < 1e-6):
            print(f"   ⚠️  Invalid initial_delta ({suggested_params['initial_delta']:.2e}) - interface too simple")
            return {
                'fractal_dimension': 1.0,  # Straight line has dimension 1.0
                'fractal_r_squared': 1.0,
                'fractal_analysis_time': 0.0,
                'fractal_interface_type': interface_type,
                'fractal_suggested_parameters': suggested_params,
                'fractal_features': features,
                'fractal_error': 'Degenerate case - straight line or initial condition'
            }

        # Check 2: Near-linear interface (very low complexity and minimal tortuosity)
        # Early timesteps in RT simulations (t < ~1.0) are essentially straight lines
        # Skip expensive box counting for these cases
        # Based on empirical data: t=0.0-0.999 have complexity=0.33-0.42, tortuosity=1.000-1.025
        if (complexity < 0.5 and features['tortuosity'] < 1.03):
            print(f"   ⚠️  Near-linear interface detected (complexity={complexity:.3f}, tortuosity={features['tortuosity']:.3f})")
            print(f"   ⚡ Skipping box counting - returning D=1.0 for early-time straight line")
            return {
                'fractal_dimension': 1.0,  # Near-straight line has dimension ≈ 1.0
                'fractal_r_squared': 1.0,
                'fractal_analysis_time': 0.001,  # Negligible time
                'fractal_interface_type': interface_type,
                'fractal_suggested_parameters': suggested_params,
                'fractal_features': features,
                'fractal_error': 'Near-linear interface - early timestep'
            }

        # Convert to SegmentArray for FastFractalAnalyzer
        segments_reshaped = segments_array.reshape(-1, 2, 2)
        segment_array_obj = SegmentArray(segments_reshaped)

        # Run fractal analysis
        print(f"\n   🏁 Running fractal dimension analysis...")
        start_time = time.time()

        # Calculate minimum box size (stop at 0.5× grid spacing)
        # This prevents box counting at sub-grid scales which are:
        # 1) Physically meaningless (can't resolve features smaller than grid)
        # 2) Computationally expensive (exponentially slower at small scales)
        # Using 0.5× grid as safety margin to avoid cutting off valid scaling region
        if len(dx_samples) > 0 and len(dy_samples) > 0:
            min_box_size = grid_spacing * 0.5  # Stop at 0.5× grid spacing
        else:
            min_box_size = 1e-4  # Fallback safety

        try:
            result = self.fractal_analyzer.analyze_segments(
                segment_array_obj,
                initial_delta=suggested_params['initial_delta'],
                delta_factor=suggested_params['delta_factor'],
                num_steps=int(suggested_params['num_steps']),
                min_box_size=min_box_size
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

    def _analyze_rt_classification(self, interface_data: InterfaceData) -> Dict:
        """
        Perform RT-specific classification (bubble/spike detection).

        Args:
            interface_data: Extracted interface data

        Returns:
            Dictionary with RT classification results
        """
        print(f"\n🫧 Step 4: RT Classification Analysis")

        # Run RT classification
        rt_results = self.rt_classifier.analyze_rt_interface(interface_data)

        print(f"\n   {'='*66}")
        print(f"   🎉 RT CLASSIFICATION RESULTS:")
        print(f"   {'='*66}")
        print(f"   🫧 Bubbles: {rt_results['bubble_count']}")
        print(f"   📌 Spikes: {rt_results['spike_count']}")
        print(f"   ⬆️  Bubble penetration: {rt_results['bubble_penetration_height']:.6f}")
        print(f"   ⬇️  Spike penetration: {rt_results['spike_penetration_depth']:.6f}")
        print(f"   📏 Mixing width: {rt_results['mixing_width']:.6f}")
        print(f"   📊 Interface amplitude: {rt_results['interface_amplitude']:.6f}")

        return rt_results

    def _analyze_power_spectrum(self, vtk_data: VTKData,
                               interface_data: Optional[InterfaceData] = None) -> Dict:
        """
        Perform comprehensive power spectrum analysis.

        Analyzes power spectra for:
        - Interface height fluctuations h(x) [if interface_data provided]
        - VOF field F(x,y)
        - Velocity components u(x,y), v(x,y) [if available]
        - Turbulent kinetic energy (TKE)

        Args:
            vtk_data: Parsed VTK data containing field grids
            interface_data: Extracted interface data (optional)

        Returns:
            Dictionary with power spectrum results for all available fields
        """
        print(f"\n📊 Step 5: Power Spectrum Analysis")

        start_time = time.time()

        # Analyze all available spectra
        spectrum_results = self.power_spectrum_analyzer.analyze_full_spectrum(
            vtk_data,
            interface_data=interface_data,
            detrend=True
        )

        analysis_time = time.time() - start_time

        if not spectrum_results:
            print(f"   ❌ Power spectrum analysis failed")
            return {
                'power_spectrum_analysis_time': analysis_time,
                'power_spectrum_error': 'Analysis failed'
            }

        # Print results for each spectrum type
        print(f"\n   {'='*66}")
        print(f"   🎉 POWER SPECTRUM RESULTS:")
        print(f"   {'='*66}")

        # Helper to print spectrum results
        def print_spectrum(name: str, spectrum: Dict):
            if spectrum:
                print(f"\n   {name}:")
                print(f"      📐 Power law slope: {spectrum['power_law_slope']:.3f}")
                print(f"      📈 Fit R²: {spectrum['power_law_r_squared']:.4f}")
                print(f"      🌊 Dominant wavelength: {spectrum['dominant_wavelength']:.6f}")
                print(f"      📏 Cutoff wavelength: {spectrum['cutoff_wavelength']:.6f}")
                print(f"      ⚡ Total energy: {spectrum['total_energy']:.6e}")
                print(f"      🎯 Peak power: {spectrum['peak_power']:.6e}")

        # Print results for each field
        if 'interface_spectrum' in spectrum_results:
            print_spectrum("Interface h(x)", spectrum_results['interface_spectrum'])

        if 'vof_spectrum' in spectrum_results:
            print_spectrum("VOF Field F(x,y)", spectrum_results['vof_spectrum'])

        if 'u_velocity_spectrum' in spectrum_results:
            print_spectrum("U-velocity u(x,y)", spectrum_results['u_velocity_spectrum'])

        if 'v_velocity_spectrum' in spectrum_results:
            print_spectrum("V-velocity v(x,y)", spectrum_results['v_velocity_spectrum'])

        if 'tke_spectrum' in spectrum_results:
            print_spectrum("Turbulent Kinetic Energy", spectrum_results['tke_spectrum'])

        print(f"\n   ⏱️  Total spectrum analysis time: {analysis_time:.2f}s")

        # Add timing to results
        spectrum_results['power_spectrum_analysis_time'] = analysis_time

        return spectrum_results

    def _analyze_velocity_statistics(self, vtk_data: VTKData) -> Dict:
        """
        Perform velocity field statistical analysis.

        Computes:
        - Spatial mean velocities ⟨u⟩, ⟨v⟩
        - RMS velocities u_rms, v_rms (relative to mean)
        - Turbulent kinetic energy (TKE)
        - Reynolds stress ⟨u'v'⟩
        - Velocity magnitude statistics

        Args:
            vtk_data: Parsed VTK data containing velocity fields

        Returns:
            Dictionary with velocity statistics results
        """
        print(f"\n🌊 Step 6: Velocity Statistics Analysis")

        start_time = time.time()

        # Analyze velocity statistics
        velocity_stats = self.velocity_analyzer.analyze_velocity_field_from_vtk(vtk_data)

        analysis_time = time.time() - start_time

        if not velocity_stats:
            print(f"   ⚠️  Velocity fields not available - skipping velocity analysis")
            return {
                'velocity_analysis_time': analysis_time,
                'velocity_error': 'Velocity fields not available'
            }

        # Print results
        print(f"\n   {'='*66}")
        print(f"   🎉 VELOCITY STATISTICS RESULTS:")
        print(f"   {'='*66}")

        print(f"\n   📍 Mean Velocities:")
        print(f"      ⟨u⟩ = {velocity_stats.u_mean:.6e}")
        print(f"      ⟨v⟩ = {velocity_stats.v_mean:.6e}")

        print(f"\n   📊 RMS Velocities (relative to mean):")
        print(f"      u_rms = {velocity_stats.u_rms:.6e}")
        print(f"      v_rms = {velocity_stats.v_rms:.6e}")

        print(f"\n   🌊 Velocity Magnitude:")
        print(f"      |V|_mean = {velocity_stats.velocity_magnitude_mean:.6e}")
        print(f"      |V|_rms = {velocity_stats.velocity_magnitude_rms:.6e}")
        print(f"      |V|_max = {velocity_stats.velocity_magnitude_max:.6e}")

        print(f"\n   ⚡ Turbulence Quantities:")
        print(f"      TKE = {velocity_stats.turbulent_kinetic_energy:.6e}")
        print(f"      ⟨u'v'⟩ = {velocity_stats.reynolds_stress:.6e}")
        print(f"      Turbulence intensity = {velocity_stats.turbulence_intensity:.6f}")

        print(f"\n   ⏱️  Analysis time: {velocity_stats.analysis_time:.4f}s")

        # Convert to dictionary for results
        return {
            'u_mean': velocity_stats.u_mean,
            'v_mean': velocity_stats.v_mean,
            'u_rms': velocity_stats.u_rms,
            'v_rms': velocity_stats.v_rms,
            'velocity_magnitude_mean': velocity_stats.velocity_magnitude_mean,
            'velocity_magnitude_rms': velocity_stats.velocity_magnitude_rms,
            'velocity_magnitude_max': velocity_stats.velocity_magnitude_max,
            'turbulent_kinetic_energy': velocity_stats.turbulent_kinetic_energy,
            'reynolds_stress': velocity_stats.reynolds_stress,
            'turbulence_intensity': velocity_stats.turbulence_intensity,
            'velocity_grid_shape': velocity_stats.grid_shape,
            'velocity_n_points': velocity_stats.n_points,
            'velocity_analysis_time': velocity_stats.analysis_time
        }

    def _analyze_mixing(self, vtk_data: VTKData) -> Dict:
        """
        Perform mixing analysis on VOF field.

        Computes:
        - Mixing zone width and boundaries
        - Mixed/unmixed fractions
        - Mixing efficiency
        - Segregation index
        - Concentration variance

        Args:
            vtk_data: Parsed VTK data containing VOF field

        Returns:
            Dictionary with mixing analysis results
        """
        print(f"\n🧪 Step 7: Mixing Analysis")

        start_time = time.time()

        # Analyze mixing
        mixing_stats = self.mixing_analyzer.analyze_mixing_from_vtk(vtk_data)

        analysis_time = time.time() - start_time

        if not mixing_stats:
            print(f"   ⚠️  VOF field not available - skipping mixing analysis")
            return {
                'mixing_analysis_time': analysis_time,
                'mixing_error': 'VOF field not available'
            }

        # Print results
        print(f"\n   {'='*66}")
        print(f"   🎉 MIXING ANALYSIS RESULTS:")
        print(f"   {'='*66}")

        print(f"\n   📏 Mixing Zone Geometry:")
        print(f"      Width: {mixing_stats.mixing_zone_width:.6f}")
        print(f"      Upper boundary (F=0.95): y = {mixing_stats.upper_boundary:.6f}")
        print(f"      Interface (F=0.5): y = {mixing_stats.interface_position:.6f}")
        print(f"      Lower boundary (F=0.05): y = {mixing_stats.lower_boundary:.6f}")

        print(f"\n   📊 Fractions:")
        print(f"      Mixed (0.1 < F < 0.9): {mixing_stats.mixed_fraction:.4f} ({mixing_stats.mixed_fraction*100:.2f}%)")
        print(f"      Unmixed light (F < 0.05): {mixing_stats.unmixed_light_fraction:.4f}")
        print(f"      Unmixed heavy (F > 0.95): {mixing_stats.unmixed_heavy_fraction:.4f}")

        print(f"\n   🎯 Mixing Quality:")
        print(f"      Mixing efficiency: {mixing_stats.mixing_efficiency:.4f}")
        print(f"      Segregation index: {mixing_stats.segregation_index:.4f}")
        print(f"      Concentration variance: {mixing_stats.concentration_variance:.6e}")

        print(f"\n   ⏱️  Analysis time: {mixing_stats.analysis_time:.4f}s")

        # Convert to dictionary for results
        return {
            'mixing_zone_width': mixing_stats.mixing_zone_width,
            'upper_boundary': mixing_stats.upper_boundary,
            'lower_boundary': mixing_stats.lower_boundary,
            'interface_position': mixing_stats.interface_position,
            'mixed_fraction': mixing_stats.mixed_fraction,
            'unmixed_light_fraction': mixing_stats.unmixed_light_fraction,
            'unmixed_heavy_fraction': mixing_stats.unmixed_heavy_fraction,
            'mixing_efficiency': mixing_stats.mixing_efficiency,
            'concentration_variance': mixing_stats.concentration_variance,
            'segregation_index': mixing_stats.segregation_index,
            'mixing_analysis_time': mixing_stats.analysis_time
        }

    def _analyze_multifractal(self, interface_data: InterfaceData) -> Dict:
        """
        Perform multifractal spectrum analysis on interface.

        Computes:
        - Generalized dimensions D_q (D₀, D₁, D₂, D_∞, D₋∞)
        - Singularity spectrum f(α)
        - Spectrum width and asymmetry

        Args:
            interface_data: Extracted interface data

        Returns:
            Dictionary with multifractal analysis results
        """
        print(f"\n🌀 Step 8: Multifractal Spectrum Analysis")

        start_time = time.time()

        # Check if interface is suitable for multifractal analysis
        # Near-linear interfaces (t < ~3s) are not truly fractal
        # Multifractal analysis produces artifacts (hierarchy violations, invalid dimensions)
        from core.interface_features import extract_interface_features

        segments_array = interface_data.segments
        features = extract_interface_features(segments_array)

        # Skip multifractal for near-linear interfaces
        # Based on empirical data: t < 3s have tortuosity < 1.5, complexity < 0.5
        if features['tortuosity'] < 1.5 or features['complexity_score'] < 0.5:
            analysis_time = time.time() - start_time
            print(f"   ⚠️  Near-linear interface (tortuosity={features['tortuosity']:.3f}, complexity={features['complexity_score']:.3f})")
            print(f"   ⏭️  Skipping multifractal analysis - interface not sufficiently complex")
            return {
                'multifractal_d0': 1.0,
                'multifractal_d1': 1.0,
                'multifractal_d2': 1.0,
                'multifractal_d_inf': 1.0,
                'multifractal_d_minus_inf': 1.0,
                'multifractal_alpha_min': 1.0,
                'multifractal_alpha_max': 1.0,
                'multifractal_alpha_0': 1.0,
                'multifractal_f_alpha_max': 1.0,
                'multifractal_width': 0.0,
                'multifractal_asymmetry': 0.0,
                'multifractal_mean_r_squared': 1.0,
                'multifractal_n_scales': 0,
                'multifractal_analysis_time': analysis_time,
                'multifractal_error': 'Near-linear interface - early timestep'
            }

        # Use characteristic length for initial box size
        char_length = interface_data.metadata['bounds']['width']
        initial_delta = char_length / 20

        # Analyze multifractal spectrum
        spectrum = self.multifractal_analyzer.analyze_multifractal(
            interface_data.segments,
            initial_delta=initial_delta,
            delta_factor=1.5,
            num_steps=12
        )

        analysis_time = time.time() - start_time

        if not spectrum:
            print(f"   ⚠️  Multifractal analysis failed")
            return {
                'multifractal_analysis_time': analysis_time,
                'multifractal_error': 'Analysis failed'
            }

        # Print results
        print(f"\n   {'='*66}")
        print(f"   🎉 MULTIFRACTAL SPECTRUM RESULTS:")
        print(f"   {'='*66}")

        print(f"\n   📊 Generalized Dimensions:")
        print(f"      D₀ (capacity): {spectrum.d0:.6f}")
        print(f"      D₁ (information): {spectrum.d1:.6f}")
        print(f"      D₂ (correlation): {spectrum.d2:.6f}")
        print(f"      D_∞: {spectrum.d_inf:.6f}")
        print(f"      D₋∞: {spectrum.d_minus_inf:.6f}")

        print(f"\n   🌀 Singularity Spectrum:")
        print(f"      α_min: {spectrum.alpha_min:.6f}")
        print(f"      α_0: {spectrum.alpha_0:.6f}")
        print(f"      α_max: {spectrum.alpha_max:.6f}")
        print(f"      f(α)_max: {spectrum.f_alpha_max:.6f}")

        print(f"\n   📐 Spectrum Characteristics:")
        print(f"      Width Δα: {spectrum.spectrum_width:.6f}")
        print(f"      Asymmetry: {spectrum.spectrum_asymmetry:.6f}")

        print(f"\n   ✅ Quality:")
        print(f"      Mean R²: {spectrum.mean_r_squared:.6f}")
        print(f"      Scales used: {spectrum.n_scales}")

        print(f"\n   ⏱️  Analysis time: {spectrum.analysis_time:.4f}s")

        # Convert to dictionary for results
        return {
            'multifractal_d0': spectrum.d0,
            'multifractal_d1': spectrum.d1,
            'multifractal_d2': spectrum.d2,
            'multifractal_d_inf': spectrum.d_inf,
            'multifractal_d_minus_inf': spectrum.d_minus_inf,
            'multifractal_alpha_min': spectrum.alpha_min,
            'multifractal_alpha_max': spectrum.alpha_max,
            'multifractal_alpha_0': spectrum.alpha_0,
            'multifractal_f_alpha_max': spectrum.f_alpha_max,
            'multifractal_width': spectrum.spectrum_width,
            'multifractal_asymmetry': spectrum.spectrum_asymmetry,
            'multifractal_mean_r_squared': spectrum.mean_r_squared,
            'multifractal_n_scales': spectrum.n_scales,
            'multifractal_analysis_time': spectrum.analysis_time
        }

    def analyze_temporal_evolution(self,
                                   vtk_pattern_or_dir: str,
                                   file_pattern: Optional[str] = None,
                                   analysis_types: Optional[List[str]] = None,
                                   time_range: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Analyze temporal evolution across multiple VTK files.

        Args:
            vtk_pattern_or_dir: Directory path
            file_pattern: Optional file pattern (glob or brace expansion)
                         Examples: "RT160x200-1*.vtk", "RT160x200-{200,1999,2999}.vtk"
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
            if file_pattern:
                # Use pattern matching
                print(f"📋 Using pattern: {file_pattern}")
                vtk_files = find_vtk_files_with_pattern(vtk_pattern_or_dir, file_pattern)
            else:
                # Find all VTK files in directory
                vtk_files = find_vtk_files(vtk_pattern_or_dir)
        else:
            print(f"❌ Input must be a directory")
            return {'success': False, 'error': 'Input must be a directory'}

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

    def save_results_json(self, results: Dict, output_path: str) -> None:
        """
        Save analysis results to JSON file.

        Args:
            results: Analysis results dictionary
            output_path: Path to output JSON file
        """
        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(item) for item in obj)
            else:
                return obj

        results_serializable = convert_for_json(results)

        # Add metadata
        results_serializable['_metadata'] = {
            'generated_by': 'RefactoredRTAnalyzer',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }

        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"   💾 Results saved to: {output_path}")

    def save_temporal_csv(self, temporal_results: Dict, output_path: str) -> None:
        """
        Save temporal evolution results to CSV file.

        CSV format provides a comprehensive digital record of all analysis results
        across time steps. As new analysis types are implemented (power spectrum,
        multifractal, mixing, velocity), additional columns will be added.

        Args:
            temporal_results: Temporal analysis results dictionary
            output_path: Path to output CSV file
        """
        if not temporal_results.get('success') or 'results' not in temporal_results:
            print(f"   ⚠️  No temporal results to save to CSV")
            return

        results_list = temporal_results['results']

        if not results_list:
            print(f"   ⚠️  Empty results list, no CSV written")
            return

        # Define CSV columns
        # NOTE: This list will expand as we implement additional analysis types
        # Current: 19 fields (Phase 1)
        # Target: ~50-60 fields (Phase 2-4 complete)
        fieldnames = [
            # === PHASE 1: Current Implementation ===
            # Basic metadata
            'time',
            'file_path',
            'grid_nx',
            'grid_ny',
            'interface_segments',
            'interface_points',
            # Fractal analysis
            'fractal_dimension',
            'fractal_r_squared',
            'fractal_interface_type',
            'fractal_analysis_time',
            'interface_extraction_time',
            'total_analysis_time',
            # Geometric features
            'characteristic_length',
            'tortuosity',
            'complexity_score',
            'connectivity_ratio',
            # AI-suggested parameters
            'initial_delta',
            'delta_factor',
            'num_steps',

            # === PHASE 2: RT-Specific Classification ===
            'bubble_count',
            'spike_count',
            'bubble_penetration_height',
            'spike_penetration_depth',
            'mixing_width',
            'interface_mean_height',
            'interface_amplitude',

            # === PHASE 3: Power Spectrum Analysis ===
            # Interface spectrum h(x)
            'interface_power_law_slope',
            'interface_power_law_r_squared',
            'interface_dominant_wavelength',
            'interface_total_energy',
            'interface_peak_power',
            # VOF field spectrum F(x,y)
            'vof_power_law_slope',
            'vof_power_law_r_squared',
            'vof_dominant_wavelength',
            'vof_total_energy',
            'vof_peak_power',
            # U-velocity spectrum u(x,y)
            'u_velocity_power_law_slope',
            'u_velocity_power_law_r_squared',
            'u_velocity_dominant_wavelength',
            'u_velocity_total_energy',
            'u_velocity_peak_power',
            # V-velocity spectrum v(x,y)
            'v_velocity_power_law_slope',
            'v_velocity_power_law_r_squared',
            'v_velocity_dominant_wavelength',
            'v_velocity_total_energy',
            'v_velocity_peak_power',
            # TKE spectrum
            'tke_power_law_slope',
            'tke_power_law_r_squared',
            'tke_dominant_wavelength',
            'tke_total_energy',
            'tke_peak_power',
            'power_spectrum_analysis_time',

            # === PHASE 3: Multifractal Analysis ===
            'multifractal_d0',           # Capacity dimension (D₀)
            'multifractal_d1',           # Information dimension (D₁)
            'multifractal_d2',           # Correlation dimension (D₂)
            'multifractal_d_inf',        # D_∞
            'multifractal_d_minus_inf',  # D_-∞
            'multifractal_alpha_min',    # Minimum singularity strength
            'multifractal_alpha_max',    # Maximum singularity strength
            'multifractal_alpha_0',      # α at f(α) maximum
            'multifractal_f_alpha_max',  # Maximum f(α)
            'multifractal_width',        # Spectrum width Δα
            'multifractal_asymmetry',    # Spectrum asymmetry
            'multifractal_mean_r_squared', # Mean R² for D_q fits
            'multifractal_n_scales',     # Number of scales used
            'multifractal_analysis_time', # Analysis time

            # === PHASE 3: Mixing Analysis ===
            'mixing_zone_width',         # Total width of mixing zone
            'upper_boundary',            # Y-coordinate of upper boundary (F ≈ 0.95)
            'lower_boundary',            # Y-coordinate of lower boundary (F ≈ 0.05)
            'interface_position',        # Y-coordinate of interface (F ≈ 0.5)
            'mixed_fraction',            # Fraction that is mixed (0.1 < F < 0.9)
            'unmixed_light_fraction',    # Fraction that is pure light fluid (F < 0.05)
            'unmixed_heavy_fraction',    # Fraction that is pure heavy fluid (F > 0.95)
            'mixing_efficiency',         # Mixing efficiency (0 = unmixed, 1 = perfect)
            'concentration_variance',    # Variance of VOF field
            'segregation_index',         # Danckwerts segregation index
            'mixing_analysis_time',      # Analysis time

            # === PHASE 3: Velocity Statistics ===
            'u_mean',                    # Spatial mean ⟨u⟩
            'v_mean',                    # Spatial mean ⟨v⟩
            'u_rms',                     # RMS horizontal velocity (WRT mean)
            'v_rms',                     # RMS vertical velocity (WRT mean)
            'velocity_magnitude_mean',   # Mean |V|
            'velocity_magnitude_rms',    # RMS total velocity magnitude
            'velocity_magnitude_max',    # Maximum velocity magnitude
            'turbulent_kinetic_energy',  # TKE = 0.5(u_rms² + v_rms²)
            'reynolds_stress',           # ⟨u'v'⟩ correlation
            'turbulence_intensity',      # Turbulence intensity
            'velocity_analysis_time',    # Analysis time
        ]

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for result in results_list:
                # Flatten nested dictionaries for CSV
                # Structure follows the same phased approach as fieldnames above
                row = {
                    # === PHASE 1: Current Implementation ===
                    'time': result.get('time', ''),
                    'file_path': os.path.basename(result.get('file_path', '')),
                    'grid_nx': result.get('grid_shape', [None, None])[0],
                    'grid_ny': result.get('grid_shape', [None, None])[1],
                    'interface_segments': result.get('interface_segments', ''),
                    'interface_points': result.get('interface_points', ''),
                    'fractal_dimension': result.get('fractal_dimension', ''),
                    'fractal_r_squared': result.get('fractal_r_squared', ''),
                    'fractal_interface_type': result.get('fractal_interface_type', ''),
                    'fractal_analysis_time': result.get('fractal_analysis_time', ''),
                    'interface_extraction_time': result.get('interface_extraction_time', ''),
                    'total_analysis_time': result.get('total_analysis_time', '')
                }

                # Extract fractal features if available
                features = result.get('fractal_features', {})
                if features:
                    row['characteristic_length'] = features.get('characteristic_length', '')
                    row['tortuosity'] = features.get('tortuosity', '')
                    row['complexity_score'] = features.get('complexity_score', '')
                    row['connectivity_ratio'] = features.get('connectivity_ratio', '')

                # Extract AI-suggested parameters if available
                params = result.get('fractal_suggested_parameters', {})
                if params:
                    row['initial_delta'] = params.get('initial_delta', '')
                    row['delta_factor'] = params.get('delta_factor', '')
                    row['num_steps'] = params.get('num_steps', '')

                # === PHASE 2: RT-Specific Classification ===
                rt_results = result.get('rt_classification', {})
                if rt_results:
                    row['bubble_count'] = rt_results.get('bubble_count', '')
                    row['spike_count'] = rt_results.get('spike_count', '')
                    row['bubble_penetration_height'] = rt_results.get('bubble_penetration_height', '')
                    row['spike_penetration_depth'] = rt_results.get('spike_penetration_depth', '')
                    row['mixing_width'] = rt_results.get('mixing_width', '')
                    row['interface_mean_height'] = rt_results.get('interface_mean_height', '')
                    row['interface_amplitude'] = rt_results.get('interface_amplitude', '')

                # === PHASE 3: Power Spectrum Analysis ===
                spectrum_results = result.get('power_spectrum', {})
                if spectrum_results:
                    # Interface spectrum h(x)
                    interface_spec = spectrum_results.get('interface_spectrum', {})
                    if interface_spec:
                        row['interface_power_law_slope'] = interface_spec.get('power_law_slope', '')
                        row['interface_power_law_r_squared'] = interface_spec.get('power_law_r_squared', '')
                        row['interface_dominant_wavelength'] = interface_spec.get('dominant_wavelength', '')
                        row['interface_total_energy'] = interface_spec.get('total_energy', '')
                        row['interface_peak_power'] = interface_spec.get('peak_power', '')

                    # VOF field spectrum F(x,y)
                    vof_spec = spectrum_results.get('vof_spectrum', {})
                    if vof_spec:
                        row['vof_power_law_slope'] = vof_spec.get('power_law_slope', '')
                        row['vof_power_law_r_squared'] = vof_spec.get('power_law_r_squared', '')
                        row['vof_dominant_wavelength'] = vof_spec.get('dominant_wavelength', '')
                        row['vof_total_energy'] = vof_spec.get('total_energy', '')
                        row['vof_peak_power'] = vof_spec.get('peak_power', '')

                    # U-velocity spectrum u(x,y)
                    u_spec = spectrum_results.get('u_velocity_spectrum', {})
                    if u_spec:
                        row['u_velocity_power_law_slope'] = u_spec.get('power_law_slope', '')
                        row['u_velocity_power_law_r_squared'] = u_spec.get('power_law_r_squared', '')
                        row['u_velocity_dominant_wavelength'] = u_spec.get('dominant_wavelength', '')
                        row['u_velocity_total_energy'] = u_spec.get('total_energy', '')
                        row['u_velocity_peak_power'] = u_spec.get('peak_power', '')

                    # V-velocity spectrum v(x,y)
                    v_spec = spectrum_results.get('v_velocity_spectrum', {})
                    if v_spec:
                        row['v_velocity_power_law_slope'] = v_spec.get('power_law_slope', '')
                        row['v_velocity_power_law_r_squared'] = v_spec.get('power_law_r_squared', '')
                        row['v_velocity_dominant_wavelength'] = v_spec.get('dominant_wavelength', '')
                        row['v_velocity_total_energy'] = v_spec.get('total_energy', '')
                        row['v_velocity_peak_power'] = v_spec.get('peak_power', '')

                    # TKE spectrum
                    tke_spec = spectrum_results.get('tke_spectrum', {})
                    if tke_spec:
                        row['tke_power_law_slope'] = tke_spec.get('power_law_slope', '')
                        row['tke_power_law_r_squared'] = tke_spec.get('power_law_r_squared', '')
                        row['tke_dominant_wavelength'] = tke_spec.get('dominant_wavelength', '')
                        row['tke_total_energy'] = tke_spec.get('total_energy', '')
                        row['tke_peak_power'] = tke_spec.get('peak_power', '')

                    # Analysis time
                    row['power_spectrum_analysis_time'] = spectrum_results.get('power_spectrum_analysis_time', '')

                # === PHASE 3: Multifractal Analysis ===
                multifractal = result.get('multifractal', {})
                if multifractal:
                    row['multifractal_d0'] = multifractal.get('multifractal_d0', '')
                    row['multifractal_d1'] = multifractal.get('multifractal_d1', '')
                    row['multifractal_d2'] = multifractal.get('multifractal_d2', '')
                    row['multifractal_d_inf'] = multifractal.get('multifractal_d_inf', '')
                    row['multifractal_d_minus_inf'] = multifractal.get('multifractal_d_minus_inf', '')
                    row['multifractal_alpha_min'] = multifractal.get('multifractal_alpha_min', '')
                    row['multifractal_alpha_max'] = multifractal.get('multifractal_alpha_max', '')
                    row['multifractal_alpha_0'] = multifractal.get('multifractal_alpha_0', '')
                    row['multifractal_f_alpha_max'] = multifractal.get('multifractal_f_alpha_max', '')
                    row['multifractal_width'] = multifractal.get('multifractal_width', '')
                    row['multifractal_asymmetry'] = multifractal.get('multifractal_asymmetry', '')
                    row['multifractal_mean_r_squared'] = multifractal.get('multifractal_mean_r_squared', '')
                    row['multifractal_n_scales'] = multifractal.get('multifractal_n_scales', '')
                    row['multifractal_analysis_time'] = multifractal.get('multifractal_analysis_time', '')

                # === PHASE 3: Mixing Analysis ===
                mixing = result.get('mixing', {})
                if mixing:
                    row['mixing_zone_width'] = mixing.get('mixing_zone_width', '')
                    row['upper_boundary'] = mixing.get('upper_boundary', '')
                    row['lower_boundary'] = mixing.get('lower_boundary', '')
                    row['interface_position'] = mixing.get('interface_position', '')
                    row['mixed_fraction'] = mixing.get('mixed_fraction', '')
                    row['unmixed_light_fraction'] = mixing.get('unmixed_light_fraction', '')
                    row['unmixed_heavy_fraction'] = mixing.get('unmixed_heavy_fraction', '')
                    row['mixing_efficiency'] = mixing.get('mixing_efficiency', '')
                    row['concentration_variance'] = mixing.get('concentration_variance', '')
                    row['segregation_index'] = mixing.get('segregation_index', '')
                    row['mixing_analysis_time'] = mixing.get('mixing_analysis_time', '')

                # === PHASE 3: Velocity Statistics ===
                velocity = result.get('velocity', {})
                if velocity:
                    row['u_mean'] = velocity.get('u_mean', '')
                    row['v_mean'] = velocity.get('v_mean', '')
                    row['u_rms'] = velocity.get('u_rms', '')
                    row['v_rms'] = velocity.get('v_rms', '')
                    row['velocity_magnitude_mean'] = velocity.get('velocity_magnitude_mean', '')
                    row['velocity_magnitude_rms'] = velocity.get('velocity_magnitude_rms', '')
                    row['velocity_magnitude_max'] = velocity.get('velocity_magnitude_max', '')
                    row['turbulent_kinetic_energy'] = velocity.get('turbulent_kinetic_energy', '')
                    row['reynolds_stress'] = velocity.get('reynolds_stress', '')
                    row['turbulence_intensity'] = velocity.get('turbulence_intensity', '')
                    row['velocity_analysis_time'] = velocity.get('velocity_analysis_time', '')

                writer.writerow(row)

        print(f"   💾 CSV saved to: {output_path}")


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
    parser.add_argument('--pattern', type=str, default=None,
                       help='File pattern for temporal mode (e.g., "RT160x200-1*.vtk" or "RT160x200-{200,1999,2999}.vtk")')
    parser.add_argument('--analysis', nargs='+', default=['fractal'],
                       choices=['fractal', 'rt_classification', 'mixing', 'power_spectrum', 'velocity', 'multifractal', 'all'],
                       help='Analysis types to perform')
    parser.add_argument('--theoretical-dim', type=float,
                       help='Theoretical dimension for validation')
    parser.add_argument('--output-dir', default='./rt_analysis_refactored',
                       help='Output directory')
    parser.add_argument('--performance', choices=['fast', 'balanced', 'accurate'],
                       default='fast', help='Performance mode')
    parser.add_argument('--interface-method', choices=['skimage', 'plic', 'conrec'],
                       default='conrec', help='Interface extraction method')
    parser.add_argument('--no-learning', action='store_true',
                       help='Disable AI learning')
    parser.add_argument('--save-json', action='store_true',
                       help='Save results to JSON file')
    parser.add_argument('--save-csv', action='store_true',
                       help='Save temporal results to CSV file (temporal mode only)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots (requires matplotlib)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Expand 'all' to all analysis types
    if 'all' in args.analysis:
        args.analysis = ['fractal', 'rt_classification', 'mixing', 'power_spectrum', 'velocity', 'multifractal']

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
            file_pattern=args.pattern,
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

        # Save outputs if requested
        if args.save_json:
            json_path = os.path.join(args.output_dir, 'results.json')
            analyzer.save_results_json(results, json_path)

        if args.save_csv and (args.temporal or os.path.isdir(args.input)):
            csv_path = os.path.join(args.output_dir, 'temporal_evolution.csv')
            analyzer.save_temporal_csv(results, csv_path)

        if args.plot:
            print(f"\n📊 Generating plots...")
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt

                # Generate plots based on analysis type
                if args.temporal or os.path.isdir(args.input):
                    _plot_temporal_evolution(results, args.output_dir)
                else:
                    _plot_single_analysis(results, args.output_dir)

                print(f"   ✅ Plots saved to {args.output_dir}/")
            except ImportError:
                print(f"   ❌ matplotlib not available - cannot generate plots")
            except Exception as e:
                print(f"   ⚠️  Plotting failed: {e}")

    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


def _plot_temporal_evolution(temporal_results: Dict, output_dir: str) -> None:
    """
    Generate temporal evolution plots.

    Args:
        temporal_results: Temporal analysis results
        output_dir: Output directory for plots
    """
    import matplotlib.pyplot as plt

    if not temporal_results.get('success') or 'results' not in temporal_results:
        return

    results = temporal_results['results']

    # Extract time series data
    times = [r['time'] for r in results if 'time' in r]
    dimensions = [r.get('fractal_dimension') for r in results if 'fractal_dimension' in r]
    r_squareds = [r.get('fractal_r_squared') for r in results if 'fractal_r_squared' in r]

    # Remove None values
    valid_indices = [i for i, d in enumerate(dimensions) if d is not None and not np.isnan(d)]
    times = [times[i] for i in valid_indices]
    dimensions = [dimensions[i] for i in valid_indices]
    r_squareds = [r_squareds[i] for i in valid_indices]

    if times:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot fractal dimension evolution
        ax1.plot(times, dimensions, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Fractal Dimension')
        ax1.set_title('Fractal Dimension Evolution')
        ax1.grid(True, alpha=0.3)

        # Plot R² quality evolution
        ax2.plot(times, r_squareds, 'r-s', linewidth=2, markersize=4)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('R² (Fit Quality)')
        ax2.set_title('Fit Quality Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.9, 1.0])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot mixing evolution if available
    if any('mixing' in r for r in results):
        print(f"   📊 Generating mixing evolution plots...")
        _plot_mixing_evolution(results, output_dir)

    # Plot velocity evolution if available
    if any('velocity' in r for r in results):
        print(f"   📊 Generating velocity evolution plots...")
        _plot_velocity_evolution(results, output_dir)

    # Plot RT classification evolution if available
    if any('rt_classification' in r for r in results):
        print(f"   📊 Generating RT classification evolution plots...")
        _plot_rt_classification_evolution(results, output_dir)

    # Plot multifractal evolution if available
    if any('multifractal' in r for r in results):
        print(f"   📊 Generating multifractal evolution plots...")
        _plot_multifractal_evolution(results, output_dir)

    # Generate power spectrum plots if available
    if any('power_spectrum' in r for r in results):
        print(f"   📊 Generating power spectrum plots...")
        spectrum_plotter = SpectrumPlotter(output_dir=output_dir, dpi=300)

        # Plot temporal evolution of spectra for each field type
        spectrum_keys = ['interface_spectrum', 'vof_spectrum', 'u_velocity_spectrum',
                        'v_velocity_spectrum', 'tke_spectrum']

        for spectrum_key in spectrum_keys:
            # Check if this spectrum type exists in any result
            if any(spectrum_key in r.get('power_spectrum', {}) for r in results):
                try:
                    filename = spectrum_plotter.plot_temporal_spectrum_evolution(
                        results,
                        spectrum_key=spectrum_key
                    )
                    print(f"      ✅ {spectrum_key} evolution: {filename}")
                except Exception as e:
                    print(f"      ⚠️  Failed to plot {spectrum_key}: {e}")


def _plot_mixing_evolution(results: List[Dict], output_dir: str) -> None:
    """Plot mixing analysis temporal evolution."""
    import matplotlib.pyplot as plt

    times = []
    mixing_widths = []
    upper_bounds = []
    lower_bounds = []
    interface_positions = []
    mixed_fractions = []
    mixing_efficiencies = []

    for r in results:
        if 'mixing' in r and 'time' in r:
            mixing = r['mixing']
            times.append(r['time'])
            mixing_widths.append(mixing.get('mixing_zone_width', np.nan))
            upper_bounds.append(mixing.get('upper_boundary', np.nan))
            lower_bounds.append(mixing.get('lower_boundary', np.nan))
            interface_positions.append(mixing.get('interface_position', np.nan))
            mixed_fractions.append(mixing.get('mixed_fraction', np.nan))
            mixing_efficiencies.append(mixing.get('mixing_efficiency', np.nan))

    if not times:
        return

    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mixing zone boundaries
    ax1.plot(times, upper_bounds, 'r-o', label='Upper boundary (F=0.95)', linewidth=2, markersize=4)
    ax1.plot(times, interface_positions, 'g-s', label='Interface (F=0.5)', linewidth=2, markersize=4)
    ax1.plot(times, lower_bounds, 'b-^', label='Lower boundary (F=0.05)', linewidth=2, markersize=4)
    ax1.fill_between(times, lower_bounds, upper_bounds, alpha=0.2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Mixing Zone Boundaries Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mixing zone width
    ax2.plot(times, mixing_widths, 'purple', marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mixing Zone Width')
    ax2.set_title('Mixing Zone Width Evolution')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mixed fraction
    ax3.plot(times, mixed_fractions, 'orange', marker='s', linewidth=2, markersize=4)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mixed Fraction (0.1 < F < 0.9)')
    ax3.set_title('Mixed Fraction Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])

    # Plot 4: Mixing efficiency
    ax4.plot(times, mixing_efficiencies, 'teal', marker='^', linewidth=2, markersize=4)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Mixing Efficiency')
    ax4.set_title('Mixing Efficiency Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mixing_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Mixing evolution: mixing_evolution.png")


def _plot_velocity_evolution(results: List[Dict], output_dir: str) -> None:
    """Plot velocity statistics temporal evolution."""
    import matplotlib.pyplot as plt

    times = []
    u_means = []
    v_means = []
    u_rms_vals = []
    v_rms_vals = []
    tke_vals = []
    reynolds_stresses = []

    for r in results:
        if 'velocity' in r and 'time' in r:
            velocity = r['velocity']
            times.append(r['time'])
            u_means.append(velocity.get('u_mean', np.nan))
            v_means.append(velocity.get('v_mean', np.nan))
            u_rms_vals.append(velocity.get('u_rms', np.nan))
            v_rms_vals.append(velocity.get('v_rms', np.nan))
            tke_vals.append(velocity.get('turbulent_kinetic_energy', np.nan))
            reynolds_stresses.append(velocity.get('reynolds_stress', np.nan))

    if not times:
        return

    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean velocities
    ax1.plot(times, u_means, 'b-o', label='⟨u⟩', linewidth=2, markersize=4)
    ax1.plot(times, v_means, 'r-s', label='⟨v⟩', linewidth=2, markersize=4)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Mean Velocity')
    ax1.set_title('Mean Velocity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: RMS velocities
    ax2.plot(times, u_rms_vals, 'b-o', label='u_rms', linewidth=2, markersize=4)
    ax2.plot(times, v_rms_vals, 'r-s', label='v_rms', linewidth=2, markersize=4)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RMS Velocity')
    ax2.set_title('RMS Velocity Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Turbulent kinetic energy
    ax3.plot(times, tke_vals, 'green', marker='^', linewidth=2, markersize=4)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('TKE')
    ax3.set_title('Turbulent Kinetic Energy Evolution')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Reynolds stress
    ax4.plot(times, reynolds_stresses, 'purple', marker='d', linewidth=2, markersize=4)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Time')
    ax4.set_ylabel("⟨u'v'⟩")
    ax4.set_title('Reynolds Stress Evolution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'velocity_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Velocity evolution: velocity_evolution.png")


def _plot_rt_classification_evolution(results: List[Dict], output_dir: str) -> None:
    """Plot RT classification temporal evolution."""
    import matplotlib.pyplot as plt

    times = []
    bubble_counts = []
    spike_counts = []
    bubble_heights = []
    spike_depths = []
    mixing_widths = []
    amplitudes = []

    for r in results:
        if 'rt_classification' in r and 'time' in r:
            rt = r['rt_classification']
            times.append(r['time'])
            bubble_counts.append(rt.get('bubble_count', 0))
            spike_counts.append(rt.get('spike_count', 0))
            bubble_heights.append(rt.get('bubble_penetration_height', np.nan))
            spike_depths.append(rt.get('spike_penetration_depth', np.nan))
            mixing_widths.append(rt.get('mixing_width', np.nan))
            amplitudes.append(rt.get('interface_amplitude', np.nan))

    if not times:
        return

    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Bubble and spike counts
    ax1.plot(times, bubble_counts, 'b-o', label='Bubbles', linewidth=2, markersize=5)
    ax1.plot(times, spike_counts, 'r-s', label='Spikes', linewidth=2, markersize=5)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Count')
    ax1.set_title('Bubble & Spike Count Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Penetration heights/depths
    ax2.plot(times, bubble_heights, 'b-o', label='Bubble penetration', linewidth=2, markersize=4)
    ax2.plot(times, spike_depths, 'r-s', label='Spike penetration', linewidth=2, markersize=4)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Penetration Distance')
    ax2.set_title('Penetration Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Mixing width from RT
    ax3.plot(times, mixing_widths, 'green', marker='^', linewidth=2, markersize=4)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Mixing Width')
    ax3.set_title('Mixing Width Evolution (RT)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Interface amplitude
    ax4.plot(times, amplitudes, 'purple', marker='d', linewidth=2, markersize=4)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Interface Amplitude')
    ax4.set_title('Interface Amplitude Evolution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rt_classification_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ RT classification evolution: rt_classification_evolution.png")


def _plot_multifractal_evolution(results: List[Dict], output_dir: str) -> None:
    """Plot multifractal spectrum temporal evolution."""
    import matplotlib.pyplot as plt

    times = []
    d0_vals = []
    d1_vals = []
    d2_vals = []
    d_inf_vals = []
    d_minus_inf_vals = []
    widths = []
    asymmetries = []

    for r in results:
        if 'multifractal' in r and 'time' in r:
            mf = r['multifractal']
            times.append(r['time'])
            d0_vals.append(mf.get('multifractal_d0', np.nan))
            d1_vals.append(mf.get('multifractal_d1', np.nan))
            d2_vals.append(mf.get('multifractal_d2', np.nan))
            d_inf_vals.append(mf.get('multifractal_d_inf', np.nan))
            d_minus_inf_vals.append(mf.get('multifractal_d_minus_inf', np.nan))
            widths.append(mf.get('multifractal_width', np.nan))
            asymmetries.append(mf.get('multifractal_asymmetry', np.nan))

    if not times:
        return

    # Create 2x2 subplot figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Generalized dimensions
    ax1.plot(times, d0_vals, 'b-o', label='D₀ (capacity)', linewidth=2, markersize=4)
    ax1.plot(times, d1_vals, 'g-s', label='D₁ (information)', linewidth=2, markersize=4)
    ax1.plot(times, d2_vals, 'r-^', label='D₂ (correlation)', linewidth=2, markersize=4)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Generalized Dimensions Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: D_∞ and D_-∞
    ax2.plot(times, d_inf_vals, 'purple', marker='o', label='D_∞', linewidth=2, markersize=4)
    ax2.plot(times, d_minus_inf_vals, 'orange', marker='s', label='D_-∞', linewidth=2, markersize=4)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Dimension')
    ax2.set_title('Extreme Dimensions Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Spectrum width
    ax3.plot(times, widths, 'teal', marker='^', linewidth=2, markersize=4)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Spectrum Width Δα')
    ax3.set_title('Multifractal Spectrum Width Evolution')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Spectrum asymmetry
    ax4.plot(times, asymmetries, 'brown', marker='d', linewidth=2, markersize=4)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Symmetric')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Asymmetry')
    ax4.set_title('Multifractal Spectrum Asymmetry Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multifractal_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      ✅ Multifractal evolution: multifractal_evolution.png")


def _plot_single_analysis(results: Dict, output_dir: str) -> None:
    """
    Generate plots for single file analysis.

    Args:
        results: Single analysis results
        output_dir: Output directory for plots
    """
    import matplotlib.pyplot as plt

    # Create a simple summary plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Summary text
    summary_text = f"""
Fractal Dimension Analysis Summary

Dimension: {results.get('fractal_dimension', 'N/A'):.6f}
R²: {results.get('fractal_r_squared', 'N/A'):.6f}
Interface Type: {results.get('fractal_interface_type', 'N/A')}
Segments: {results.get('interface_segments', 'N/A')}
Analysis Time: {results.get('fractal_analysis_time', 'N/A'):.1f}s
    """

    ax.text(0.5, 0.5, summary_text.strip(),
           ha='center', va='center', fontsize=12, family='monospace')
    ax.axis('off')
    ax.set_title('Analysis Summary', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate power spectrum plots if available
    if 'power_spectrum' in results:
        print(f"   📊 Generating power spectrum plots...")
        spectrum_plotter = SpectrumPlotter(output_dir=output_dir, dpi=300)
        spectrum_results = results['power_spectrum']

        # Plot multi-field comparison if multiple spectra available
        try:
            filename = spectrum_plotter.plot_multi_field_comparison(
                spectrum_results,
                time=results.get('time', 0.0)
            )
            print(f"      ✅ Multi-field comparison: {filename}")
        except Exception as e:
            print(f"      ⚠️  Failed to plot multi-field comparison: {e}")


if __name__ == "__main__":
    main()
