#!/usr/bin/env python3
"""
Test the improved CONREC filtering with adaptive thresholds.
"""

import numpy as np
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser
from integration.rt_interface_extraction import InterfaceExtractor

# Parse VTK file
vtk_file = '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-4999.vtk'
parser = VTKParser(debug=False)
vtk_data = parser.parse_vtk_file(vtk_file)

print("=" * 80)
print("TESTING IMPROVED CONREC SEGMENT FILTERING")
print("=" * 80)

# Extract interface with debug enabled to see filtering
extractor = InterfaceExtractor(method='conrec', debug=True)
interface_data = extractor.extract_interface(vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5)

if interface_data:
    segments = interface_data.segments
    print(f"\n✅ Extracted {len(segments)} segments after filtering")

    # Analyze remaining segment lengths
    segment_lengths = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segment_lengths.append(length)

    segment_lengths = np.array(segment_lengths)

    print(f"\nSegment length statistics (after filtering):")
    print(f"  Min length: {np.min(segment_lengths):.10f}")
    print(f"  Max length: {np.max(segment_lengths):.10f}")
    print(f"  Mean length: {np.mean(segment_lengths):.10f}")
    print(f"  Median length: {np.median(segment_lengths):.10f}")

    # Show distribution
    print(f"\nSegment length distribution:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(segment_lengths, p)
        print(f"  {p:3d}th percentile: {val:.6f}")

    # Grid spacing for reference
    dx = 0.0025
    dy = 0.0025
    print(f"\nComparison to grid spacing (dx=dy={dx}):")
    print(f"  Min/grid: {np.min(segment_lengths)/dx:.4f}× grid spacing")
    print(f"  Mean/grid: {np.mean(segment_lengths)/dx:.4f}× grid spacing")
    print(f"  Median/grid: {np.median(segment_lengths)/dx:.4f}× grid spacing")

    # Check if any segments are unreasonably small
    unreasonably_small = np.sum(segment_lengths < dx * 0.01)  # < 1% of grid
    if unreasonably_small > 0:
        print(f"\n⚠️  WARNING: {unreasonably_small} segments still smaller than 1% of grid spacing")
    else:
        print(f"\n✅ All segments are >= 1% of grid spacing (physically reasonable)")
else:
    print("❌ Failed to extract interface")
