#!/usr/bin/env python3
"""
Debug connectivity calculation for RT160x200-4999.vtk
"""

import numpy as np
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser
from integration.rt_interface_extraction import InterfaceExtractor
from core.interface_features import compute_connectivity

# Parse VTK file
vtk_file = '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-4999.vtk'
parser = VTKParser(debug=False)
vtk_data = parser.parse_vtk_file(vtk_file)

if vtk_data is None:
    print("Failed to parse VTK file")
    sys.exit(1)

# Extract interface with both methods for comparison
print(f"Extracting interface from {vtk_file}")

# Try skimage first
print("\n=== SKIMAGE METHOD ===")
extractor_skimage = InterfaceExtractor(method='skimage', debug=False)
interface_data_skimage = extractor_skimage.extract_interface(vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5)
segments_skimage = interface_data_skimage.segments

# Try conrec
print("\n=== CONREC METHOD ===")
extractor_conrec = InterfaceExtractor(method='conrec', debug=False)
interface_data_conrec = extractor_conrec.extract_interface(vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5)
segments_conrec = interface_data_conrec.segments

# Use conrec for analysis
segments = segments_conrec
print(f"\nUsing CONREC segments for connectivity analysis")

print(f"\nInterface segments: {len(segments)}")
print(f"First few segments:")
for i in range(min(5, len(segments))):
    print(f"  Segment {i}: ({segments[i,0]:.6f}, {segments[i,1]:.6f}) -> ({segments[i,2]:.6f}, {segments[i,3]:.6f})")

# Test connectivity calculation with detailed output
print("\n" + "="*70)
print("CONNECTIVITY ANALYSIS")
print("="*70)

n = len(segments)
tolerance = 0.001

# Build adjacency graph
start_points = {}
for i in range(n):
    start_key = (round(segments[i, 0] / tolerance), round(segments[i, 1] / tolerance))
    if start_key not in start_points:
        start_points[start_key] = []
    start_points[start_key].append(i)

adjacency = [[] for _ in range(n)]
for i in range(n):
    end_key = (round(segments[i, 2] / tolerance), round(segments[i, 3] / tolerance))
    if end_key in start_points:
        for j in start_points[end_key]:
            if j != i:
                adjacency[i].append(j)
                adjacency[j].append(i)

# Count connected components
visited = [False] * n
n_components = 0
component_sizes = []

for i in range(n):
    if not visited[i]:
        n_components += 1
        component_size = 0
        queue = [i]
        visited[i] = True

        while queue:
            current = queue.pop(0)
            component_size += 1
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        component_sizes.append(component_size)

print(f"Number of connected components: {n_components}")
print(f"Component sizes: {component_sizes}")

connectivity = 1.0 - (n_components - 1) / n
print(f"Connectivity: {connectivity:.4f} ({connectivity * 100:.1f}%)")

# Also show what the old method would give
connected_count = 0
for i in range(n - 1):
    end_point = segments[i, 2:4]
    next_start = segments[i+1, 0:2]
    distance = np.sqrt(np.sum((end_point - next_start)**2))
    if distance < 0.001:
        connected_count += 1

old_connectivity = connected_count / (n - 1)
print(f"\nOld method connectivity: {old_connectivity:.4f} ({old_connectivity * 100:.1f}%)")

# Show some disconnections
print("\nLooking for large gaps between consecutive segments:")
gap_count = 0
for i in range(n - 1):
    end_point = segments[i, 2:4]
    next_start = segments[i+1, 0:2]
    distance = np.sqrt(np.sum((end_point - next_start)**2))
    if distance > 0.001:
        gap_count += 1
        if gap_count <= 10:
            print(f"  Gap at segment {i} -> {i+1}: distance = {distance:.6f}")
print(f"Total gaps found: {gap_count}")

# Check if the interface forms a closed loop
first_start = segments[0, 0:2]
last_end = segments[-1, 2:4]
loop_distance = np.sqrt(np.sum((first_start - last_end)**2))
print(f"\nDistance from last segment end to first segment start: {loop_distance:.6f}")
if loop_distance < 0.001:
    print("✅ Interface forms a closed loop")
else:
    print("❌ Interface is NOT a closed loop")

# Analyze spatial distribution to detect separate bubbles
print("\n" + "="*70)
print("BUBBLE DETECTION ANALYSIS")
print("="*70)

# Compute center points of segments
centers = (segments[:, 0:2] + segments[:, 2:4]) / 2.0
y_coords = centers[:, 1]

print(f"Interface Y-coordinate range: [{np.min(y_coords):.6f}, {np.max(y_coords):.6f}]")
print(f"Interface Y-coordinate mean: {np.mean(y_coords):.6f}")
print(f"Interface Y-coordinate std: {np.std(y_coords):.6f}")

# Look for segments that might indicate separate bubbles
# Bubbles would show up as isolated regions in X-Y space
x_coords = centers[:, 0]
print(f"Interface X-coordinate range: [{np.min(x_coords):.6f}, {np.max(x_coords):.6f}]")

# Simple heuristic: look for segments far apart in the sequence
print("\nLooking for distant segment pairs (might indicate wrapping around bubbles):")
for i in range(min(20, n - 1)):
    end_point = segments[i, 2:4]
    next_start = segments[i+1, 0:2]
    center_distance = np.sqrt(np.sum((segments[i, 0:2] - segments[i+1, 0:2])**2))
    if center_distance > 0.1:  # Large spatial jump
        print(f"  Segment {i} at ({segments[i,0]:.6f}, {segments[i,1]:.6f}) to {i+1} at ({segments[i+1,0]:.6f}, {segments[i+1,1]:.6f}): jump = {center_distance:.6f}")
