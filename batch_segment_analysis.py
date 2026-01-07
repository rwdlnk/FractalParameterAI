#!/usr/bin/env python3
"""
Batch Segment File Analysis - Uses existing AI Framework

Analyzes all segment files in a directory and produces temporal evolution results.
"""

import sys
import os
import re
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Add FastFractalAnalyzer to path
FAST_FRACTAL_PATH = '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer'
if FAST_FRACTAL_PATH not in sys.path:
    sys.path.insert(0, FAST_FRACTAL_PATH)

from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer
from fractal_analyzer.core.performance_config import PerformanceMode

from core.interface_features import (
    extract_interface_features,
    classify_interface_type,
    suggest_optimal_parameters_adaptive,
)


def analyze_single_file(filepath, verbose=False):
    """Analyze a single segment file and return results."""
    try:
        segments = SegmentArray.from_file(filepath)

        # Extract features
        segments_array = np.column_stack([
            segments.segments[:, 0, 0],
            segments.segments[:, 0, 1],
            segments.segments[:, 1, 0],
            segments.segments[:, 1, 1]
        ])
        
        features = extract_interface_features(segments_array)
        interface_type = classify_interface_type(features)
        suggested_params = suggest_optimal_parameters_adaptive(features, implementation="basic_box_counting")

        # Run analysis
        analyzer = FastFractalAnalyzer(PerformanceMode.BALANCED)
        result = analyzer.analyze_segments(
            segments,
            initial_delta=suggested_params.get('initial_delta'),
            delta_factor=suggested_params.get('delta_factor', 1.3),
            num_steps=int(suggested_params.get('num_steps', 12)),
        )

        return {
            'n_segments': segments.n_segments,
            'D': result.dimension,
            'r_squared': result.r_squared,
            'interface_type': interface_type,
            'complexity': features.get('complexity_score', 0),
        }
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_segment_analysis.py <segment_directory> [output_csv] [output_plot]")
        sys.exit(1)

    segment_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'batch_results.csv'
    output_plot = sys.argv[3] if len(sys.argv) > 3 else 'batch_evolution.png'

    print("=" * 70)
    print("BATCH 2D FRACTAL ANALYSIS (AI Framework)")
    print("=" * 70)
    print(f"Directory: {segment_dir}")

    # Find segment files
    files = []
    for f in os.listdir(segment_dir):
        if f.endswith('.txt'):
            match = re.search(r't(\d+\.?\d*)', f)
            if match:
                t = float(match.group(1))
                files.append((t, os.path.join(segment_dir, f)))
    
    files.sort(key=lambda x: x[0])
    print(f"Found {len(files)} segment files")
    print()

    results = []
    for i, (t, filepath) in enumerate(files):
        print(f"[{i+1}/{len(files)}] t = {t:6.2f}...", end=" ", flush=True)
        
        r = analyze_single_file(filepath)
        if r:
            results.append({'time': t, **r})
            print(f"D = {r['D']:.4f}, R² = {r['r_squared']:.4f}, segments = {r['n_segments']}")
        else:
            print("FAILED")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Time':>8}  {'Segments':>10}  {'D':>8}  {'R²':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['time']:8.2f}  {r['n_segments']:10d}  {r['D']:8.4f}  {r['r_squared']:10.5f}")

    # Save CSV
    with open(output_csv, 'w') as f:
        f.write("time,n_segments,D,r_squared,interface_type,complexity\n")
        for r in results:
            f.write(f"{r['time']},{r['n_segments']},{r['D']},{r['r_squared']},{r['interface_type']},{r['complexity']}\n")
    print(f"\nResults saved to {output_csv}")

    # Generate plot
    if len(results) > 0:
        times = [r['time'] for r in results]
        Ds = [r['D'] for r in results]
        r2s = [r['r_squared'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(times, Ds, 'b.-', markersize=4)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='D=1.0 (smooth line)')
        ax1.set_ylabel('Fractal Dimension D')
        ax1.set_title('2D RT Interface Temporal Evolution (AI Framework)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, r2s, 'g.-', markersize=4)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('R²')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_plot, dpi=150)
        print(f"Plot saved to {output_plot}")


if __name__ == '__main__':
    main()
