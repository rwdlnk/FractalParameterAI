#!/usr/bin/env python3
"""
Check grid spacing to understand segment sizes.
"""

import numpy as np
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser

# Parse VTK file
vtk_file = '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-4999.vtk'
parser = VTKParser(debug=False)
vtk_data = parser.parse_vtk_file(vtk_file)

print(f"Grid shape: {vtk_data.x_grid.shape}")
print(f"Is cell data: {vtk_data.is_cell_data}")

# Check x-grid
print(f"\nX-grid analysis:")
print(f"  Range: [{np.min(vtk_data.x_grid):.6f}, {np.max(vtk_data.x_grid):.6f}]")
print(f"  x_grid[0,0] = {vtk_data.x_grid[0,0]:.10f}")
print(f"  x_grid[0,1] = {vtk_data.x_grid[0,1]:.10f}")
print(f"  x_grid[1,0] = {vtk_data.x_grid[1,0]:.10f}")
print(f"  x_grid[1,1] = {vtk_data.x_grid[1,1]:.10f}")

# Calculate actual spacings
x_spacing_i = []
for i in range(min(10, vtk_data.x_grid.shape[0] - 1)):
    dx = vtk_data.x_grid[i+1, 0] - vtk_data.x_grid[i, 0]
    x_spacing_i.append(dx)

x_spacing_j = []
for j in range(min(10, vtk_data.x_grid.shape[1] - 1)):
    dx = vtk_data.x_grid[0, j+1] - vtk_data.x_grid[0, j]
    x_spacing_j.append(dx)

print(f"\nX-spacing in i-direction (first 10):")
for i, dx in enumerate(x_spacing_i):
    print(f"  x_grid[{i+1},0] - x_grid[{i},0] = {dx:.10f}")

print(f"\nX-spacing in j-direction (first 10):")
for j, dx in enumerate(x_spacing_j):
    print(f"  x_grid[0,{j+1}] - x_grid[0,{j}] = {dx:.10f}")

# Check y-grid
print(f"\nY-grid analysis:")
print(f"  Range: [{np.min(vtk_data.y_grid):.6f}, {np.max(vtk_data.y_grid):.6f}]")
print(f"  y_grid[0,0] = {vtk_data.y_grid[0,0]:.10f}")
print(f"  y_grid[0,1] = {vtk_data.y_grid[0,1]:.10f}")
print(f"  y_grid[1,0] = {vtk_data.y_grid[1,0]:.10f}")
print(f"  y_grid[1,1] = {vtk_data.y_grid[1,1]:.10f}")

# Calculate actual spacings
y_spacing_i = []
for i in range(min(10, vtk_data.y_grid.shape[0] - 1)):
    dy = vtk_data.y_grid[i+1, 0] - vtk_data.y_grid[i, 0]
    y_spacing_i.append(dy)

y_spacing_j = []
for j in range(min(10, vtk_data.y_grid.shape[1] - 1)):
    dy = vtk_data.y_grid[0, j+1] - vtk_data.y_grid[0, j]
    y_spacing_j.append(dy)

print(f"\nY-spacing in i-direction (first 10):")
for i, dy in enumerate(y_spacing_i):
    print(f"  y_grid[{i+1},0] - y_grid[{i},0] = {dy:.10f}")

print(f"\nY-spacing in j-direction (first 10):")
for j, dy in enumerate(y_spacing_j):
    print(f"  y_grid[0,{j+1}] - y_grid[0,{j}] = {dy:.10f}")

# Typical grid spacing
typical_dx = np.median(np.abs(np.diff(vtk_data.x_grid[vtk_data.x_grid.shape[0]//2, :])))
typical_dy = np.median(np.abs(np.diff(vtk_data.y_grid[:, vtk_data.y_grid.shape[1]//2])))
print(f"\nTypical grid spacing (median):")
print(f"  dx = {typical_dx:.10f}")
print(f"  dy = {typical_dy:.10f}")
print(f"  diagonal = {(typical_dx**2 + typical_dy**2)**0.5:.10f}")
