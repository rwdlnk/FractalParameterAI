#!/usr/bin/env python3
"""
Compute Dalziel's h_1,1 mixing integral from VTK files.

h_1,1 = ∫ C̄(z) * (1 - C̄(z)) dz

where C̄(z) is the horizontal average of volume fraction F at height z.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add FractalParameterAI to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'FractalParameterAI'))

from integration.rt_vtk_parser import VTKParser

def compute_h11_from_vtk(vtk_file):
    """Compute h_1,1 from a single VTK file."""
    parser = VTKParser(debug=False)
    vtk_data = parser.parse_vtk_file(vtk_file)

    if not vtk_data:
        return None

    # Get F field (try both lowercase and uppercase)
    f_grid = None
    for key in ['f', 'F', 'vof', 'VOF']:
        if key in vtk_data.scalar_fields:
            f_grid = vtk_data.scalar_fields[key]
            break

    if f_grid is None:
        print(f"Warning: No F field found in {vtk_file}")
        return None

    # Get y-coordinates (assume first column since y uniform in x)
    y_coords = vtk_data.y_grid[0, :]

    # Compute horizontal average C̄(z)
    mean_f_profile = np.mean(f_grid, axis=0)  # Average along x (axis 0)

    # Compute integrand: C̄(1-C̄)
    integrand = mean_f_profile * (1.0 - mean_f_profile)

    # Integrate using trapezoidal rule
    h_11 = np.trapz(integrand, y_coords)

    return h_11

def process_resolution(run_dir, output_file):
    """Process all VTK files in a resolution directory."""
    vtk_dir = Path(run_dir) 
    vtk_files = sorted(vtk_dir.glob('*.vtk'))

    if not vtk_files:
        print(f"No VTK files found in {vtk_dir}")
        return

    print(f"Processing {len(vtk_files)} VTK files from {run_dir}...")

    results = []
    for vtk_file in vtk_files:
        # Extract time from filename (e.g., rt.0050.vtk -> 50)
        # Adjust this based on your actual filename format
        # Extract time from filename (e.g., RT160x200-1999.vtk -> 1999)
        try:
            # Split on '-' to get the number part
            filename_parts = vtk_file.stem.split('-')
            if len(filename_parts) >= 2:
                time_index = int(filename_parts[-1])
            else:
                print(f"Warning: Could not extract time from {vtk_file.name}")
                continue
        except:
            print(f"Warning: Could not extract time from {vtk_file.name}")
            continue

        h_11 = compute_h11_from_vtk(str(vtk_file))

        if h_11 is not None:
            results.append((time_index, h_11))

            if len(results) % 50 == 0:
                print(f"  Processed {len(results)} files...")

    # Save results
    results = np.array(results)
    np.savetxt(output_file, results,
               header='time_index h_1,1(m)',
               fmt='%d %.6e')

    print(f"✅ Saved {len(results)} values to {output_file}")
    print(f"   h_1,1 range: [{results[:, 1].min():.6f}, {results[:, 1].max():.6f}] m")

if __name__ == "__main__":
    # Define your resolution directories
    base_dir = Path("/media/rod/ResearchII_III/svofRuns/Dalziel_1999")

    resolutions = [
        "160x200",
        "320x400",
        "640x800",
        "960x1200",
        "1280x1600"
    ]

    for res in resolutions:
        run_dir = base_dir / res / "slimMaster"
        output_file = run_dir / "rt_analysis_ai/h11_mixing_integral.txt"

        print(f"\n{'='*70}")
        print(f"Processing resolution: {res}")
        print(f"{'='*70}")

        process_resolution(run_dir, output_file)
