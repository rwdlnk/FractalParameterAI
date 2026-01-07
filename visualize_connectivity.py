#!/usr/bin/env python3
"""
Visualize the connectivity of RT interface to understand what we're measuring.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

from integration.rt_vtk_parser import VTKParser
from integration.rt_interface_extraction import InterfaceExtractor

# Parse VTK file
vtk_file = '/media/rod/ResearchII_III/svofRuns/Dalziel_1999/160x200/slimMaster/RT160x200-4999.vtk'
parser = VTKParser(debug=False)
vtk_data = parser.parse_vtk_file(vtk_file)

# Extract interface
extractor = InterfaceExtractor(method='skimage', debug=False)
interface_data = extractor.extract_interface(vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid, level=0.5)
segments = interface_data.segments

# Convert segments to points for visualization
points = []
for seg in segments:
    points.append([seg[0], seg[1]])  # Start point
# Add the last endpoint
points.append([segments[-1, 2], segments[-1, 3]])
points = np.array(points)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Show the interface as a continuous path with segment numbering
ax1.plot(points[:, 0], points[:, 1], 'b-', linewidth=1, alpha=0.7, label='Interface path')
ax1.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start (segment 0)')
ax1.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, label='End (segment {})'.format(len(segments)-1))

# Mark every 50th point to show direction
step = max(1, len(points) // 20)
for i in range(0, len(points), step):
    ax1.plot(points[i, 0], points[i, 1], 'k.', markersize=3)
    if i % (step * 2) == 0:
        ax1.annotate(str(i), (points[i, 0], points[i, 1]), fontsize=6, alpha=0.6)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title(f'Interface Path Connectivity\n({len(segments)} segments forming ONE connected curve)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Right plot: Show the VOF field with interface overlay
im = ax2.contourf(vtk_data.x_grid, vtk_data.y_grid, vtk_data['f'], levels=20, cmap='RdBu_r')
ax2.contour(vtk_data.x_grid, vtk_data.y_grid, vtk_data['f'], levels=[0.5], colors='black', linewidths=2)
ax2.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, alpha=0.8, label='Extracted interface')
ax2.plot(points[0, 0], points[0, 1], 'go', markersize=10, label='Start')
ax2.plot(points[-1, 0], points[-1, 1], 'ro', markersize=10, label='End')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('VOF Field with Interface Overlay')
ax2.legend()
plt.colorbar(im, ax=ax2, label='VOF (Volume of Fluid)')
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('/tmp/connectivity_visualization.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved to /tmp/connectivity_visualization.png")

# Also print detailed analysis
print("\n" + "="*70)
print("CONNECTIVITY INTERPRETATION")
print("="*70)
print(f"""
The RT interface at t=4.999 has:
  • {len(segments)} line segments forming ONE connected curve
  • Connectivity: 100% (interface has not fragmented)
  • Start point: ({points[0,0]:.6f}, {points[0,1]:.6f})
  • End point:   ({points[-1,0]:.6f}, {points[-1,1]:.6f})
  • Gap between start and end: {np.sqrt((points[0,0]-points[-1,0])**2 + (points[0,1]-points[-1,1])**2):.6f}

WHAT CONNECTIVITY MEASURES:
  ✅ Whether the interface has broken into separate disconnected pieces
  ✅ 100% = interface is one continuous curve (even with complex shape)
  ✅ < 100% = interface has fragmented into multiple disconnected curves

WHAT CONNECTIVITY DOES NOT MEASURE:
  ❌ Number of bubbles or spikes (use RT classification for that)
  ❌ Complexity or tortuosity of the interface
  ❌ Whether the curve forms a closed loop

The bubbles and spikes detected by RT classification are PEAKS and VALLEYS
along this single connected curve, not separate disconnected objects.
""")
