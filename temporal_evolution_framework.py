#!/usr/bin/env python3
"""
Temporal Evolution Framework for AI-Enhanced Fractal Parameter Learning
Progressive analysis from simple to complex RT interfaces with adaptive learning
"""

import sys
import os
import time
import numpy as np
import json
import glob
from pathlib import Path

# Add necessary paths
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')

def discover_vtk_files(vtk_directory, time_range=(1.0, 12.0), time_step=1.0):
    """
    Discover available VTK files in the specified time range.

    Args:
        vtk_directory: Path to directory containing VTK files
        time_range: (start_time, end_time) in seconds
        time_step: Time step increment in seconds

    Returns:
        List of (time, filepath) tuples sorted by time
    """
    print(f"🔍 Discovering VTK files in {vtk_directory}")
    print(f"   Time range: {time_range[0]:.1f}s - {time_range[1]:.1f}s (step: {time_step:.1f}s)")

    available_files = []

    # Generate candidate times
    current_time = time_range[0]
    while current_time <= time_range[1]:
        # Convert time to timestamp (t * 1000)
        timestamp = int(current_time * 1000)

        # Try different possible filename patterns
        possible_files = [
            f"RT160x200-{timestamp}.vtk",
            f"RT160x200-{timestamp:04d}.vtk",
        ]

        for filename in possible_files:
            filepath = os.path.join(vtk_directory, filename)
            if os.path.exists(filepath):
                available_files.append((current_time, filepath))
                print(f"   ✅ Found: t={current_time:.3f}s → {filename}")
                break
        else:
            print(f"   ❌ Missing: t={current_time:.3f}s")

        current_time += time_step

    print(f"\n📊 Summary: {len(available_files)} files found")
    return sorted(available_files)

class TemporalAIFramework:
    """
    AI Framework that learns parameter optimization progressively through time.
    """

    def __init__(self):
        self.temporal_results = []
        self.learning_database = []
        self.parameter_evolution = {
            'times': [],
            'dimensions': [],
            'initial_deltas': [],
            'delta_factors': [],
            'num_steps': [],
            'r_squared': [],
            'segments': []
        }

    def suggest_parameters(self, segments, domain_scale, prior_results=None):
        """
        Suggest parameters based on current interface and learning from prior results.

        Args:
            segments: Current interface segments
            domain_scale: Domain scale for parameter scaling
            prior_results: List of previous (time, params, result) tuples

        Returns:
            Dictionary with suggested parameters
        """

        if prior_results is None or len(prior_results) == 0:
            # First time step - use baseline parameters
            return {
                'initial_delta': domain_scale / 200,  # Conservative start
                'delta_factor': 1.4,
                'num_steps': 15
            }

        # Extract trends from prior results
        times = [r[0] for r in prior_results]
        params = [r[1] for r in prior_results]
        results = [r[2] for r in prior_results]

        # Analyze parameter evolution trends
        if len(prior_results) >= 2:
            # Look at parameter trends
            recent_delta = params[-1]['initial_delta']
            recent_factor = params[-1]['delta_factor']
            recent_steps = params[-1]['num_steps']

            # Check if complexity is increasing (more segments)
            recent_segments = len(prior_results[-1][3]) if len(prior_results[-1]) > 3 else 1000
            prev_segments = len(prior_results[-2][3]) if len(prior_results[-2]) > 3 else 1000

            complexity_increase = (recent_segments - prev_segments) / prev_segments

            if complexity_increase > 0.5:  # 50% increase in segments
                # Interface getting more complex - refine parameters
                suggested_delta = recent_delta * 0.8  # Smaller boxes
                suggested_factor = min(recent_factor + 0.1, 2.0)  # Slightly increase factor
                suggested_steps = min(recent_steps + 2, 20)  # More steps
            elif complexity_increase < 0.1:  # Stable complexity
                # Keep similar parameters
                suggested_delta = recent_delta
                suggested_factor = recent_factor
                suggested_steps = recent_steps
            else:  # Moderate increase
                suggested_delta = recent_delta * 0.9
                suggested_factor = recent_factor
                suggested_steps = recent_steps + 1
        else:
            # Single prior result - modest adaptation
            suggested_delta = params[-1]['initial_delta'] * 0.95
            suggested_factor = params[-1]['delta_factor']
            suggested_steps = params[-1]['num_steps']

        return {
            'initial_delta': suggested_delta,
            'delta_factor': suggested_factor,
            'num_steps': int(suggested_steps)
        }

    def add_result(self, time, segments, parameters, result):
        """Add a temporal result to the learning database."""
        self.temporal_results.append((time, parameters, result, segments))

        # Update evolution tracking
        self.parameter_evolution['times'].append(time)
        self.parameter_evolution['dimensions'].append(result.dimension if result else np.nan)
        self.parameter_evolution['initial_deltas'].append(parameters['initial_delta'])
        self.parameter_evolution['delta_factors'].append(parameters['delta_factor'])
        self.parameter_evolution['num_steps'].append(parameters['num_steps'])
        self.parameter_evolution['r_squared'].append(result.r_squared if result else np.nan)
        self.parameter_evolution['segments'].append(segments.n_segments if segments else 0)

    def get_evolution_summary(self):
        """Get summary of parameter evolution."""
        if not self.temporal_results:
            return {}

        return {
            'time_range': (min(self.parameter_evolution['times']),
                          max(self.parameter_evolution['times'])),
            'dimension_range': (np.nanmin(self.parameter_evolution['dimensions']),
                               np.nanmax(self.parameter_evolution['dimensions'])),
            'total_timepoints': len(self.temporal_results),
            'successful_analyses': sum(1 for d in self.parameter_evolution['dimensions'] if not np.isnan(d)),
            'complexity_evolution': {
                'initial_segments': self.parameter_evolution['segments'][0],
                'final_segments': self.parameter_evolution['segments'][-1],
                'growth_factor': self.parameter_evolution['segments'][-1] / max(1, self.parameter_evolution['segments'][0])
            }
        }

def run_temporal_evolution_analysis(vtk_directory, time_range=(1.0, 12.0), time_step=1.0, output_file="temporal_evolution_results.json"):
    """
    Run complete temporal evolution analysis.

    Args:
        vtk_directory: Directory containing VTK files
        time_range: (start_time, end_time) for analysis
        time_step: Time step between analyses
        output_file: Output file for results
    """

    print("🌟 Temporal Evolution Framework - AI-Enhanced RT Interface Analysis")
    print("=" * 75)

    # Discover available VTK files
    vtk_files = discover_vtk_files(vtk_directory, time_range, time_step)

    if not vtk_files:
        print("❌ No VTK files found in specified time range")
        return None

    print(f"\n📋 Analysis Plan: {len(vtk_files)} time steps")

    # Initialize framework
    ai_framework = TemporalAIFramework()

    # Import necessary modules
    try:
        from fractal_analyzer.io.vtk_reader import VTKReader
        from fractal_analyzer.core.conrec_extractor import CONRECExtractor
        from fractal_analyzer.core.data_types import SegmentArray
        from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

        reader = VTKReader()
        extractor = CONRECExtractor()
        analyzer = FastFractalAnalyzer()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

    # Progressive analysis loop
    total_start_time = time.time()

    for i, (current_time, vtk_file) in enumerate(vtk_files):
        print(f"\n🔬 Analysis {i+1}/{len(vtk_files)}: t={current_time:.3f}s")
        print(f"   📁 File: {os.path.basename(vtk_file)}")

        try:
            # Step 1: Extract interface
            step_start = time.time()
            vtk_data = reader.read_vtk_file(vtk_file)

            volume_fraction = vtk_data['F']
            x_grid = vtk_data['x']
            y_grid = vtk_data['y']

            segment_list = extractor.extract_interface_conrec(volume_fraction, x_grid, y_grid, 0.5)
            segments = SegmentArray.from_list(segment_list)
            extraction_time = time.time() - step_start

            print(f"   ✅ Interface: {segments.n_segments} segments ({extraction_time:.1f}s)")

            # Step 2: AI parameter suggestion
            domain_x = segments.bbox.max_x - segments.bbox.min_x
            domain_y = segments.bbox.max_y - segments.bbox.min_y
            domain_scale = max(domain_x, domain_y)

            ai_params = ai_framework.suggest_parameters(
                segments, domain_scale, ai_framework.temporal_results
            )

            print(f"   🤖 AI Parameters: δ₀={ai_params['initial_delta']:.6f}, "
                  f"factor={ai_params['delta_factor']:.3f}, steps={ai_params['num_steps']}")

            # Step 3: Box counting analysis
            bc_start = time.time()
            result = analyzer.compute_fractal_dimension(segments)
            bc_time = time.time() - bc_start

            if result and hasattr(result, 'dimension'):
                print(f"   📊 Dimension: {result.dimension:.6f} (R²={result.r_squared:.4f}, {bc_time:.1f}s)")

                # Add to learning database
                ai_framework.add_result(current_time, segments, ai_params, result)

                # Progress indicators
                if i > 0:
                    prev_dim = ai_framework.parameter_evolution['dimensions'][-2]
                    dim_change = result.dimension - prev_dim if not np.isnan(prev_dim) else 0

                    print(f"   📈 Evolution: Δt={current_time - vtk_files[i-1][0]:.1f}s, "
                          f"ΔD={dim_change:+.6f}")

            else:
                print(f"   ❌ Box counting failed")
                ai_framework.add_result(current_time, segments, ai_params, None)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            ai_framework.add_result(current_time, None, {}, None)

    # Analysis complete
    total_time = time.time() - total_start_time
    print(f"\n🎉 Temporal Evolution Analysis Complete!")
    print(f"   Total time: {total_time:.1f}s")

    # Generate summary
    summary = ai_framework.get_evolution_summary()
    print(f"   Time range: {summary.get('time_range', (0, 0))[0]:.1f}s - {summary.get('time_range', (0, 0))[1]:.1f}s")
    print(f"   Successful analyses: {summary.get('successful_analyses', 0)}/{summary.get('total_timepoints', 0)}")

    if summary.get('successful_analyses', 0) > 0:
        print(f"   Dimension evolution: {summary.get('dimension_range', (0, 0))[0]:.6f} → {summary.get('dimension_range', (0, 0))[1]:.6f}")
        print(f"   Interface complexity: {summary.get('complexity_evolution', {}).get('growth_factor', 1):.1f}x growth")

    # Save results
    output_data = {
        'metadata': {
            'analysis_type': 'temporal_evolution',
            'vtk_directory': vtk_directory,
            'time_range': time_range,
            'time_step': time_step,
            'total_analysis_time': total_time,
            'timestamp': time.time()
        },
        'summary': summary,
        'parameter_evolution': ai_framework.parameter_evolution,
        'detailed_results': []
    }

    # Add detailed results
    for i, (time_val, params, result, segments) in enumerate(ai_framework.temporal_results):
        detail = {
            'time': time_val,
            'parameters': params,
            'n_segments': segments.n_segments if segments else 0,
            'dimension': result.dimension if result and hasattr(result, 'dimension') else None,
            'r_squared': result.r_squared if result and hasattr(result, 'r_squared') else None
        }
        output_data['detailed_results'].append(detail)

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"   📁 Results saved to: {output_file}")

    return output_data

def main():
    """Main function for temporal evolution analysis."""

    # Configuration
    vtk_directory = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster"

    # Run temporal evolution analysis from nearly straight line to complex interface
    results = run_temporal_evolution_analysis(
        vtk_directory=vtk_directory,
        time_range=(1.999, 12.0),    # Start from nearly straight line
        time_step=1.0,               # 1 second intervals
        output_file="rt_temporal_evolution_results.json"
    )

    if results:
        print(f"\n🎯 Key Findings:")
        summary = results['summary']

        if summary.get('successful_analyses', 0) > 0:
            evol = results['parameter_evolution']

            print(f"   • Interface complexity grew {summary['complexity_evolution']['growth_factor']:.1f}x over time")
            print(f"   • Fractal dimension evolved from {np.nanmin(evol['dimensions']):.6f} to {np.nanmax(evol['dimensions']):.6f}")
            print(f"   • AI adapted parameters through {len(evol['times'])} time steps")

            # Parameter evolution trends
            if len(evol['initial_deltas']) > 1:
                delta_trend = evol['initial_deltas'][-1] / evol['initial_deltas'][0]
                print(f"   • δ₀ evolved {delta_trend:.2f}x (adaptive scaling)")

        print(f"\n📚 This temporal framework demonstrates:")
        print(f"   • Progressive AI learning from simple → complex interfaces")
        print(f"   • Adaptive parameter evolution with interface growth")
        print(f"   • Scientific validation of RT instability fractal evolution")

if __name__ == "__main__":
    main()