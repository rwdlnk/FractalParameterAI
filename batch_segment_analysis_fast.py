#!/usr/bin/env python3
"""
Fast Batch Segment Analysis - Simple box counting without heavy AI optimization
"""

import sys
import os
import re
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')
from fractal_analyzer.core.data_types import SegmentArray
from fractal_analyzer.core.box_counting import VectorizedBoxCounter


def analyze_single_file(filepath):
    """Fast analysis using basic box counting."""
    try:
        segments = SegmentArray.from_file(filepath)
        
        # Use basic box counter with fixed parameters
        counter = VectorizedBoxCounter()
        
        # Get domain scale
        char_length = max(segments.bbox.width, segments.bbox.height)
        initial_delta = char_length / 5
        
        deltas = []
        counts = []
        delta = initial_delta
        
        for _ in range(12):
            n = counter.count_boxes(segments, delta)
            if n > 0:
                deltas.append(delta)
                counts.append(n)
            delta /= 1.5
            if delta < char_length / 500:
                break
        
        if len(deltas) < 3:
            return None
            
        log_inv_delta = np.log(1.0 / np.array(deltas))
        log_n = np.log(np.array(counts))
        slope, intercept, r, p, se = stats.linregress(log_inv_delta, log_n)
        
        return {
            'n_segments': segments.n_segments,
            'D': slope,
            'r_squared': r**2,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_segment_analysis_fast.py <segment_directory> [output_csv] [output_plot]")
        sys.exit(1)

    segment_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'batch_results.csv'
    output_plot = sys.argv[3] if len(sys.argv) > 3 else 'batch_evolution.png'

    print("=" * 70)
    print("FAST BATCH 2D FRACTAL ANALYSIS")
    print("=" * 70)
    print(f"Directory: {segment_dir}")

    files = []
    for f in os.listdir(segment_dir):
        if f.endswith('.txt'):
            match = re.search(r't(\d+\.?\d*)', f)
            if match:
                t = float(match.group(1))
                files.append((t, os.path.join(segment_dir, f)))
    
    files.sort(key=lambda x: x[0])
    print(f"Found {len(files)} segment files\n")

    results = []
    for i, (t, filepath) in enumerate(files):
        print(f"[{i+1}/{len(files)}] t = {t:6.2f}...", end=" ", flush=True)
        
        r = analyze_single_file(filepath)
        if r:
            results.append({'time': t, **r})
            print(f"D = {r['D']:.4f}, R² = {r['r_squared']:.4f}")
        else:
            print("FAILED")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Time':>8}  {'Segments':>10}  {'D':>8}  {'R²':>10}")
    print("-" * 40)
    for r in results:
        print(f"{r['time']:8.2f}  {r['n_segments']:10d}  {r['D']:8.4f}  {r['r_squared']:10.5f}")

    with open(output_csv, 'w') as f:
        f.write("time,n_segments,D,r_squared\n")
        for r in results:
            f.write(f"{r['time']},{r['n_segments']},{r['D']},{r['r_squared']}\n")
    print(f"\nResults saved to {output_csv}")

    if len(results) > 0:
        times = [r['time'] for r in results]
        Ds = [r['D'] for r in results]
        r2s = [r['r_squared'] for r in results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, Ds, 'b.-', markersize=4)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='D=1.0 (smooth line)')
        ax1.set_ylabel('Fractal Dimension D')
        ax1.set_title('2D RT Interface Temporal Evolution')
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
