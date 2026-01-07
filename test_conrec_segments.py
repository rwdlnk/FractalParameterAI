#!/usr/bin/env python3
"""
Test CONREC segment extraction to understand zero-length segment issue.
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

if vtk_data is None:
    print("Failed to parse VTK file")
    sys.exit(1)

# Extract interface with conrec
print(f"Extracting interface from {vtk_file} with CONREC")
extractor = InterfaceExtractor(method='conrec', debug=False)
interface_data = extractor.extract_interface(vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5)

if interface_data is None:
    print("Failed to extract interface")
    sys.exit(1)

segments = interface_data.segments
print(f"\nTotal segments: {len(segments)}")

# Analyze segment lengths
segment_lengths = []
for seg in segments:
    x1, y1, x2, y2 = seg
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    segment_lengths.append(length)

segment_lengths = np.array(segment_lengths)

print(f"\nSegment length statistics:")
print(f"  Min length: {np.min(segment_lengths):.10f}")
print(f"  Max length: {np.max(segment_lengths):.10f}")
print(f"  Mean length: {np.mean(segment_lengths):.10f}")
print(f"  Median length: {np.median(segment_lengths):.10f}")
print(f"  Std dev: {np.std(segment_lengths):.10f}")

# Count segments by length threshold
thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3]
print(f"\nSegment count by length threshold:")
for thresh in thresholds:
    count = np.sum(segment_lengths < thresh)
    pct = count / len(segments) * 100
    print(f"  < {thresh:.0e}: {count:5d} segments ({pct:5.2f}%)")

# Show some very small segments
print(f"\nFirst 20 segments with length < 1e-6:")
small_segments = [(i, seg, segment_lengths[i]) for i, seg in enumerate(segments) if segment_lengths[i] < 1e-6]
for i, (idx, seg, length) in enumerate(small_segments[:20]):
    x1, y1, x2, y2 = seg
    print(f"  Segment {idx}: ({x1:.10f}, {y1:.10f}) -> ({x2:.10f}, {y2:.10f}), length = {length:.10e}")

# Grid spacing for reference
dx = vtk_data.x_grid[0, 1] - vtk_data.x_grid[0, 0]
dy = vtk_data.y_grid[1, 0] - vtk_data.y_grid[0, 0]
print(f"\nGrid spacing for reference:")
print(f"  dx = {dx:.10f}")
print(f"  dy = {dy:.10f}")
print(f"  Expected segment length: ~{(dx**2 + dy**2)**0.5:.10f}")

# Calculate what percentage of segments are smaller than 1% of grid spacing
grid_spacing = min(abs(dx), abs(dy))
tiny_threshold = grid_spacing * 0.01
tiny_count = np.sum(segment_lengths < tiny_threshold)
print(f"\nSegments smaller than 1% of grid spacing ({tiny_threshold:.10e}):")
print(f"  Count: {tiny_count} ({tiny_count/len(segments)*100:.2f}%)")
