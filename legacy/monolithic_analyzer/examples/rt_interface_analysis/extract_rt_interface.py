#!/usr/bin/env python3
"""
RT Interface Extraction Tool

Reads RT VTK files, extracts interfaces using conrec contour extraction,
saves them as segment files, and creates journal-ready plots.

Usage:
    python extract_rt_interface.py input.vtk [output.dat] [--contour_value 0.5] [--domain x0,x1,y0,y1]

Author: Generated for RT interface analysis
"""

import sys
import os
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.0


def read_vtk_file(filename):
    """
    Read VTK rectilinear grid file with cell-centered scalars.

    Args:
        filename: Path to VTK file

    Returns:
        tuple: (x_coords, y_coords, z_coords, scalar_data, dims)
               where coords are 1D arrays and scalar_data is 3D array
    """
    # Read VTK rectilinear grid file
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()

    # Get the data
    data = reader.GetOutput()

    # Get dimensions (number of points in each direction)
    dims = data.GetDimensions()
    nx, ny, nz = dims[0], dims[1], dims[2]
    print(f"Grid dimensions: {nx} x {ny} x {nz} vertices")
    print(f"Cell dimensions: {nx-1} x {ny-1} x {nz-1}")

    # For rectilinear grids, coordinates are stored separately
    x_coords = vtk_to_numpy(data.GetXCoordinates())
    y_coords = vtk_to_numpy(data.GetYCoordinates())
    z_coords = vtk_to_numpy(data.GetZCoordinates())

    print(f"Coordinate ranges:")
    print(f"  X: {x_coords.min():.6f} to {x_coords.max():.6f}")
    print(f"  Y: {y_coords.min():.6f} to {y_coords.max():.6f}")
    print(f"  Z: {z_coords.min():.6f} to {z_coords.max():.6f}")

    # Get cell-centered scalar data
    cell_data = data.GetCellData()
    if cell_data.GetNumberOfArrays() == 0:
        raise ValueError("No cell data found in VTK file")

    # List available arrays
    print("Available cell data arrays:")
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        print(f"  {i}: {array_name}")

    # Get the first scalar array (or look for 'F' if available)
    scalar_array = None
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        if array_name == 'F':
            scalar_array = cell_data.GetArray(i)
            break

    if scalar_array is None:
        scalar_array = cell_data.GetArray(0)

    scalar_name = scalar_array.GetName()
    print(f"Using scalar field: '{scalar_name}'")

    # Convert to numpy and reshape to 3D cell array
    scalar_data = vtk_to_numpy(scalar_array)

    # Handle 2D case (nz=1) by reshaping appropriately
    if nz == 1:
        scalar_data = scalar_data.reshape((1, ny-1, nx-1))  # Single layer 2D grid
    else:
        scalar_data = scalar_data.reshape((nz-1, ny-1, nx-1))  # Cell-centered data

    print(f"Scalar range: {scalar_data.min():.6f} to {scalar_data.max():.6f}")

    return x_coords, y_coords, z_coords, scalar_data, dims


def extract_2d_slice_contour(x_coords, y_coords, z_coords, scalar_data, contour_value, z_level=None):
    """
    Extract 2D contour from 3D structured grid at specified z-level.

    Args:
        x_coords: X coordinates (1D array, nx points)
        y_coords: Y coordinates (1D array, ny points)
        z_coords: Z coordinates (1D array, nz points)
        scalar_data: Cell-centered scalar field (3D array, shape (nz-1, ny-1, nx-1))
        contour_value: Value to extract contour at
        z_level: Z-level index to extract (default: middle layer)

    Returns:
        list: List of line segments [(x1, y1, x2, y2), ...]
    """
    # Determine which z-level to extract
    nz_cells = scalar_data.shape[0]
    if z_level is None:
        z_level = nz_cells // 2  # Middle layer
        print(f"Using middle z-level: {z_level} of {nz_cells}")
    elif z_level >= nz_cells:
        z_level = nz_cells - 1
        print(f"Adjusting z_level to maximum: {z_level}")

    # Extract 2D slice at specified z-level
    z_slice = scalar_data[z_level, :, :]  # Shape: (ny-1, nx-1)
    ny_cells, nx_cells = z_slice.shape

    print(f"Extracting contour from 2D slice: {nx_cells} x {ny_cells} cells")
    print(f"Slice scalar range: {z_slice.min():.6f} to {z_slice.max():.6f}")

    segments = []

    # Convert cell centers to grid coordinates for conrec
    # Cell centers are at midpoints between vertices
    x_cell_centers = (x_coords[:-1] + x_coords[1:]) / 2
    y_cell_centers = (y_coords[:-1] + y_coords[1:]) / 2

    # Implement basic conrec-style contour extraction on 2D slice
    for i in range(ny_cells - 1):
        for j in range(nx_cells - 1):
            # Get the four corners of the cell (cell-centered values)
            f00 = z_slice[i, j]
            f10 = z_slice[i, j+1]
            f01 = z_slice[i+1, j]
            f11 = z_slice[i+1, j+1]

            # Get cell center coordinates
            x0, x1 = x_cell_centers[j], x_cell_centers[j+1]
            y0, y1 = y_cell_centers[i], y_cell_centers[i+1]

            # Check if contour passes through this cell
            fmin = min(f00, f10, f01, f11)
            fmax = max(f00, f10, f01, f11)

            if fmin <= contour_value <= fmax:
                # Find intersections with cell edges
                intersections = []

                # Bottom edge (y=y0)
                if (f00 <= contour_value <= f10) or (f10 <= contour_value <= f00):
                    if f10 != f00:
                        t = (contour_value - f00) / (f10 - f00)
                        x_int = x0 + t * (x1 - x0)
                        intersections.append((x_int, y0))

                # Right edge (x=x1)
                if (f10 <= contour_value <= f11) or (f11 <= contour_value <= f10):
                    if f11 != f10:
                        t = (contour_value - f10) / (f11 - f10)
                        y_int = y0 + t * (y1 - y0)
                        intersections.append((x1, y_int))

                # Top edge (y=y1)
                if (f01 <= contour_value <= f11) or (f11 <= contour_value <= f01):
                    if f11 != f01:
                        t = (contour_value - f01) / (f11 - f01)
                        x_int = x0 + t * (x1 - x0)
                        intersections.append((x_int, y1))

                # Left edge (x=x0)
                if (f00 <= contour_value <= f01) or (f01 <= contour_value <= f00):
                    if f01 != f00:
                        t = (contour_value - f00) / (f01 - f00)
                        y_int = y0 + t * (y1 - y0)
                        intersections.append((x0, y_int))

                # Remove duplicates (within tolerance)
                unique_intersections = []
                for pt in intersections:
                    is_duplicate = False
                    for existing_pt in unique_intersections:
                        if abs(pt[0] - existing_pt[0]) < 1e-10 and abs(pt[1] - existing_pt[1]) < 1e-10:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_intersections.append(pt)

                # Create segments from pairs of intersections
                if len(unique_intersections) >= 2:
                    # For multiple intersections, pair them up
                    for k in range(0, len(unique_intersections)-1, 2):
                        if k+1 < len(unique_intersections):
                            pt1 = unique_intersections[k]
                            pt2 = unique_intersections[k+1]
                            segments.append((pt1[0], pt1[1], pt2[0], pt2[1]))

    return segments


def save_segments(segments, filename):
    """
    Save segments to file in standard format.

    Args:
        segments: List of segments [(x1, y1, x2, y2), ...]
        filename: Output filename
    """
    with open(filename, 'w') as f:
        f.write(f"# RT Interface segments extracted from VTK\n")
        f.write(f"# Format: x1 y1 x2 y2\n")
        f.write(f"# Number of segments: {len(segments)}\n")

        for seg in segments:
            f.write(f"{seg[0]:.8f} {seg[1]:.8f} {seg[2]:.8f} {seg[3]:.8f}\n")

    print(f"Saved {len(segments)} segments to {filename}")


def create_interface_plots(segments, domain, base_filename):
    """
    Create journal-ready plots of the interface within the solution domain.

    Args:
        segments: List of segments [(x1, y1, x2, y2), ...]
        domain: Domain bounds (x0, x1, y0, y1) or None for auto
        base_filename: Base filename for output plots
    """
    if len(segments) == 0:
        print("No segments to plot")
        return

    # Set up the plot with journal-ready formatting
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot each segment
    for seg in segments:
        x1, y1, x2, y2 = seg
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, color='#2ca02c')

    # Set domain limits
    if domain:
        x0, x1, y0, y1 = domain
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    else:
        # Auto-determine domain with some padding
        all_x = []
        all_y = []
        for seg in segments:
            all_x.extend([seg[0], seg[2]])
            all_y.extend([seg[1], seg[3]])

        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)

    # Journal formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Save PNG
    png_filename = f"{base_filename}_interface.png"
    plt.savefig(png_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved PNG plot: {png_filename}")

    # Save EPS
    eps_filename = f"{base_filename}_interface.eps"
    plt.savefig(eps_filename, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved EPS plot: {eps_filename}")

    plt.close()


def parse_domain(domain_str):
    """Parse domain string 'x0,x1,y0,y1' into tuple of floats."""
    if not domain_str:
        return None
    try:
        parts = domain_str.split(',')
        if len(parts) != 4:
            raise ValueError("Domain must have exactly 4 values: x0,x1,y0,y1")
        return tuple(float(x.strip()) for x in parts)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid domain format '{domain_str}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract RT interface from VTK file, save as segments, and create plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_rt_interface.py data.vtk
  python extract_rt_interface.py data.vtk interface.dat
  python extract_rt_interface.py data.vtk --contour_value 0.5 --domain "0,0.4,0,0.5"
  python extract_rt_interface.py data.vtk interface.dat --contour_value 0.3
        """
    )

    parser.add_argument('input_file', help='Input VTK file')
    parser.add_argument('output_file', nargs='?', help='Output segment file (default: input_file.dat)')
    parser.add_argument('--contour_value', type=float, default=0.5,
                       help='Contour value to extract (default: 0.5)')
    parser.add_argument('--domain', type=str,
                       help='Plot domain as "x0,x1,y0,y1" (default: auto-fit)')
    parser.add_argument('--z_level', type=int,
                       help='Z-level index to extract (default: middle layer)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Determine output filename
    if args.output_file:
        output_file = args.output_file
        base_name = os.path.splitext(args.output_file)[0]
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_interface.dat"

    # Parse domain if provided
    domain = parse_domain(args.domain) if args.domain else None

    try:
        print(f"Reading VTK file: {args.input_file}")
        x_coords, y_coords, z_coords, scalar_data, dims = read_vtk_file(args.input_file)

        if args.verbose:
            nx, ny, nz = dims
            print(f"Grid dimensions: {nx} x {ny} x {nz} vertices")
            print(f"X range: {x_coords[0]:.6f} to {x_coords[-1]:.6f}")
            print(f"Y range: {y_coords[0]:.6f} to {y_coords[-1]:.6f}")
            print(f"Z range: {z_coords[0]:.6f} to {z_coords[-1]:.6f}")
            print(f"Scalar range: {scalar_data.min():.6f} to {scalar_data.max():.6f}")

        print(f"Extracting contour at value: {args.contour_value}")
        segments = extract_2d_slice_contour(x_coords, y_coords, z_coords, scalar_data, args.contour_value, args.z_level)

        if len(segments) == 0:
            print(f"Warning: No contour found at value {args.contour_value}")
            print(f"Scalar field range: {scalar_data.min():.6f} to {scalar_data.max():.6f}")
        else:
            save_segments(segments, output_file)

            if args.verbose:
                total_length = sum(np.sqrt((s[2]-s[0])**2 + (s[3]-s[1])**2) for s in segments)
                print(f"Total interface length: {total_length:.6f}")

            # Create plots
            print("Creating journal-ready plots...")
            create_interface_plots(segments, domain, base_name)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()