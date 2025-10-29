#!/usr/bin/env python3
"""
Standalone VTK Parser for RT Simulations

Parses VTK RECTILINEAR_GRID files commonly used in Rayleigh-Taylor simulations.
Extracts coordinate grids and scalar fields (F, u, v, etc.) without external dependencies.

This module is part of the FractalParameterAI framework and provides
standalone VTK parsing independent of the original FractalAnalyzer codebase.
"""

import numpy as np
import re
import os
import glob
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VTKData:
    """Container for parsed VTK data."""
    x_grid: np.ndarray  # 2D mesh grid of x-coordinates
    y_grid: np.ndarray  # 2D mesh grid of y-coordinates
    scalar_fields: Dict[str, np.ndarray]  # Dictionary of scalar fields (F, u, v, etc.)
    dimensions: Tuple[int, int, int]  # (nx, ny, nz)
    time: float  # Simulation time
    is_cell_data: bool  # True if data is cell-centered, False if node-centered
    file_path: str  # Original file path

    def __getitem__(self, key: str):
        """Allow dictionary-style access for backward compatibility."""
        if key == 'x':
            return self.x_grid
        elif key == 'y':
            return self.y_grid
        elif key in self.scalar_fields:
            return self.scalar_fields[key]
        elif key == 'f' and 'F' in self.scalar_fields:
            return self.scalar_fields['F']
        elif key == 'dims':
            return self.dimensions
        elif key == 'time':
            return self.time
        elif key == 'is_cell_data':
            return self.is_cell_data
        else:
            raise KeyError(f"Key '{key}' not found in VTKData")

    def get(self, key: str, default=None):
        """Get item with default value."""
        try:
            return self[key]
        except KeyError:
            return default


class VTKParser:
    """
    Standalone VTK parser for RT simulation files.

    Supports:
    - RECTILINEAR_GRID format
    - Cell-centered and node-centered data
    - Multiple scalar fields (F, u, v, etc.)
    - Automatic time extraction from filename
    """

    def __init__(self, debug: bool = False):
        """
        Initialize VTK parser.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def parse_vtk_file(self, vtk_file_path: str) -> Optional[VTKData]:
        """
        Parse a VTK RECTILINEAR_GRID file.

        Args:
            vtk_file_path: Path to VTK file

        Returns:
            VTKData object containing parsed data, or None on error
        """
        if not os.path.exists(vtk_file_path):
            print(f"❌ VTK file not found: {vtk_file_path}")
            return None

        try:
            with open(vtk_file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ Error reading VTK file: {e}")
            return None

        if self.debug:
            print(f"📂 Parsing: {os.path.basename(vtk_file_path)}")
            print(f"   Total lines: {len(lines)}")

        # Extract dimensions
        dimensions = self._extract_dimensions(lines)
        if dimensions is None:
            print(f"❌ Could not extract dimensions from VTK file")
            return None

        nx, ny, nz = dimensions
        if self.debug:
            print(f"   Dimensions: {nx} × {ny} × {nz}")

        # Extract coordinates
        x_coords = self._extract_coordinates(lines, "X_COORDINATES")
        y_coords = self._extract_coordinates(lines, "Y_COORDINATES")

        if x_coords is None or y_coords is None:
            print(f"❌ Could not extract coordinates from VTK file")
            return None

        if self.debug:
            print(f"   X range: [{x_coords[0]:.6f}, {x_coords[-1]:.6f}]")
            print(f"   Y range: [{y_coords[0]:.6f}, {y_coords[-1]:.6f}]")

        # Check if cell-centered or node-centered
        is_cell_data = any("CELL_DATA" in line for line in lines)

        # Create coordinate grids
        if is_cell_data:
            nx_cells, ny_cells = nx - 1, ny - 1
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx_cells, ny_cells)
        else:
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx, ny)

        if self.debug:
            print(f"   Grid type: {'CELL_DATA' if is_cell_data else 'POINT_DATA'}")
            print(f"   Grid shape: {grid_shape}")

        # Extract all scalar fields
        scalar_fields = self._extract_scalar_fields(lines, grid_shape)

        if not scalar_fields:
            print(f"⚠️  No scalar fields found in VTK file")
        elif self.debug:
            print(f"   Scalar fields: {list(scalar_fields.keys())}")
            for name, field in scalar_fields.items():
                print(f"      {name}: [{np.min(field):.6f}, {np.max(field):.6f}]")

        # Ensure VOF field is available
        if 'F' not in scalar_fields and 'f' not in scalar_fields:
            print(f"⚠️  No VOF field (F or f) found in VTK file")
        elif 'F' in scalar_fields and 'f' not in scalar_fields:
            # Standardize field name
            scalar_fields['f'] = scalar_fields['F']

        # Extract simulation time
        sim_time = self._extract_simulation_time(vtk_file_path, lines)
        if self.debug:
            print(f"   Simulation time: {sim_time:.6f}")

        # Create VTKData object
        vtk_data = VTKData(
            x_grid=x_grid,
            y_grid=y_grid,
            scalar_fields=scalar_fields,
            dimensions=dimensions,
            time=sim_time,
            is_cell_data=is_cell_data,
            file_path=vtk_file_path
        )

        return vtk_data

    def _extract_dimensions(self, lines: List[str]) -> Optional[Tuple[int, int, int]]:
        """Extract grid dimensions from VTK file."""
        for line in lines:
            if "DIMENSIONS" in line:
                parts = line.strip().split()
                try:
                    nx = int(parts[1])
                    ny = int(parts[2])
                    nz = int(parts[3])
                    return (nx, ny, nz)
                except (IndexError, ValueError):
                    return None
        return None

    def _extract_coordinates(self, lines: List[str], coord_type: str) -> Optional[np.ndarray]:
        """
        Extract coordinate array from VTK file.

        Args:
            lines: VTK file lines
            coord_type: "X_COORDINATES", "Y_COORDINATES", or "Z_COORDINATES"

        Returns:
            numpy array of coordinates
        """
        for i, line in enumerate(lines):
            if coord_type in line:
                parts = line.strip().split()
                try:
                    n_coords = int(parts[1])
                except (IndexError, ValueError):
                    return None

                # Read coordinate values (may span multiple lines)
                coords_data = []
                j = i + 1
                while len(coords_data) < n_coords and j < len(lines):
                    line_values = lines[j].strip().split()
                    if line_values:  # Skip empty lines
                        try:
                            coords_data.extend([float(v) for v in line_values])
                        except ValueError:
                            break
                    j += 1

                if len(coords_data) == n_coords:
                    return np.array(coords_data)
                else:
                    return None

        return None

    def _extract_scalar_fields(self, lines: List[str], grid_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Extract all scalar fields from VTK file.

        Args:
            lines: VTK file lines
            grid_shape: Expected grid shape (nx, ny)

        Returns:
            Dictionary mapping field names to 2D numpy arrays
        """
        scalar_fields = {}
        expected_size = grid_shape[0] * grid_shape[1]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith("SCALARS"):
                parts = line.split()
                if len(parts) >= 2:
                    field_name = parts[1]

                    # Skip LOOKUP_TABLE line
                    i += 1
                    if i >= len(lines):
                        break

                    # Read data values
                    i += 1
                    data_values = []
                    while i < len(lines) and len(data_values) < expected_size:
                        line = lines[i].strip()

                        # Stop if we hit another field
                        if line.startswith("SCALARS") or line.startswith("VECTORS"):
                            break

                        if line:  # Skip empty lines
                            try:
                                values = [float(v) for v in line.split()]
                                data_values.extend(values)
                            except ValueError:
                                break

                        i += 1

                    # Reshape to 2D grid if we got the right amount of data
                    if len(data_values) == expected_size:
                        # Reshape: data comes in row-major order (y varies fastest)
                        field_2d = np.array(data_values).reshape(grid_shape[1], grid_shape[0]).T
                        scalar_fields[field_name] = field_2d

                        if self.debug:
                            print(f"      Extracted {field_name}: {len(data_values)} values → {field_2d.shape}")
                    else:
                        if self.debug:
                            print(f"      ⚠️  {field_name}: size mismatch ({len(data_values)} vs {expected_size})")

                    continue

            i += 1

        return scalar_fields

    def _extract_simulation_time(self, vtk_file_path: str, lines: List[str]) -> float:
        """
        Extract simulation time from filename or VTK content.

        Priority:
        1. Filename pattern like RT160x200-10000.vtk → time = 10.000
        2. Explicit time field in VTK file
        3. Default to 0.0

        Args:
            vtk_file_path: VTK file path
            lines: VTK file lines

        Returns:
            Simulation time as float
        """
        filename = os.path.basename(vtk_file_path)

        # Try to extract from filename (e.g., RT160x200-10000.vtk)
        match = re.search(r'-(\d+)\.vtk$', filename)
        if match:
            time_int = int(match.group(1))
            return time_int / 1000.0  # Convert to physical time

        # Try to find time field in VTK file content
        for line in lines:
            if "TIME" in line.upper():
                parts = line.strip().split()
                for i, part in enumerate(parts):
                    if "TIME" in part.upper() and i + 1 < len(parts):
                        try:
                            return float(parts[i + 1])
                        except ValueError:
                            pass

        # Default to 0.0
        return 0.0


def expand_brace_pattern(pattern: str) -> List[str]:
    """
    Expand brace patterns like RT160x200-{200,1999,2999}.vtk

    Args:
        pattern: Pattern with brace expansion

    Returns:
        List of expanded patterns
    """
    # Find brace patterns
    brace_match = re.search(r'\{([^}]+)\}', pattern)

    if not brace_match:
        return [pattern]

    # Extract comma-separated values
    brace_content = brace_match.group(1)
    values = [v.strip() for v in brace_content.split(',')]

    # Generate expanded patterns
    expanded = []
    for value in values:
        expanded_pattern = pattern[:brace_match.start()] + value + pattern[brace_match.end():]
        expanded.append(expanded_pattern)

    return expanded


def find_vtk_files_with_pattern(directory: str, file_pattern: str, skip_mesh: bool = True) -> List[str]:
    """
    Find VTK files matching glob or brace patterns.

    Supports:
    - Glob patterns: RT160x200-1*.vtk
    - Brace expansion: RT160x200-{200,1999,2999}.vtk
    - Mixed: RT160x200-{1,2}*.vtk

    Args:
        directory: Directory to search
        file_pattern: File pattern with glob/brace syntax
        skip_mesh: Skip mesh files

    Returns:
        Sorted list of VTK file paths
    """
    # Expand brace patterns first
    expanded_patterns = expand_brace_pattern(file_pattern)

    # Collect all matching files
    all_files = []
    for pattern in expanded_patterns:
        # Create full path pattern
        full_pattern = os.path.join(directory, pattern)

        # Use glob to find matching files
        matched_files = glob.glob(full_pattern)
        all_files.extend(matched_files)

    # Filter out mesh files if requested
    if skip_mesh:
        all_files = [f for f in all_files if 'Mesh' not in os.path.basename(f)]

    # Remove duplicates and sort numerically
    all_files = list(set(all_files))

    def extract_time_number(filepath):
        """Extract numerical time value from filename for sorting."""
        filename = os.path.basename(filepath)
        match = re.search(r'-(\d+)\.vtk$', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')

    all_files.sort(key=extract_time_number)

    return all_files


def find_vtk_files(directory: str, pattern: Optional[str] = None, skip_mesh: bool = True) -> List[str]:
    """
    Find and sort VTK files in a directory.

    Args:
        directory: Directory to search
        pattern: Optional regex pattern to filter filenames
        skip_mesh: Skip mesh files (files with 'Mesh' in name)

    Returns:
        Sorted list of VTK file paths
    """
    if not os.path.isdir(directory):
        print(f"❌ Directory not found: {directory}")
        return []

    vtk_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.vtk'):
            # Skip mesh files if requested
            if skip_mesh and 'Mesh' in filename:
                continue

            if pattern is None or re.search(pattern, filename):
                vtk_files.append(os.path.join(directory, filename))

    # Sort by numerical time value extracted from filename (not alphabetically)
    # This handles filenames like RT160x200-0.vtk, RT160x200-1199.vtk, RT160x200-10000.vtk correctly
    def extract_time_number(filepath):
        """Extract numerical time value from filename for sorting."""
        filename = os.path.basename(filepath)
        match = re.search(r'-(\d+)\.vtk$', filename)
        if match:
            return int(match.group(1))
        else:
            # If no number found, sort alphabetically (fallback)
            return float('inf')

    vtk_files.sort(key=extract_time_number)

    return vtk_files


def main():
    """Test VTK parser."""
    import argparse

    parser = argparse.ArgumentParser(description='Test VTK parser')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    vtk_parser = VTKParser(debug=args.debug)
    vtk_data = vtk_parser.parse_vtk_file(args.vtk_file)

    if vtk_data:
        print(f"\n✅ Successfully parsed VTK file")
        print(f"   Grid: {vtk_data.dimensions[0]} × {vtk_data.dimensions[1]}")
        print(f"   Time: {vtk_data.time:.6f}")
        print(f"   Scalar fields: {list(vtk_data.scalar_fields.keys())}")
        print(f"   X range: [{np.min(vtk_data.x_grid):.6f}, {np.max(vtk_data.x_grid):.6f}]")
        print(f"   Y range: [{np.min(vtk_data.y_grid):.6f}, {np.max(vtk_data.y_grid):.6f}]")

        if 'f' in vtk_data.scalar_fields or 'F' in vtk_data.scalar_fields:
            f_field = vtk_data['f']
            print(f"   VOF field: [{np.min(f_field):.6f}, {np.max(f_field):.6f}]")
    else:
        print(f"\n❌ Failed to parse VTK file")


if __name__ == "__main__":
    main()
