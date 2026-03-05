"""
VTK file reader for FastFractalAnalyzer v3.

Optimized reader for RECTILINEAR_GRID format with cell-centered data,
specifically designed for Rayleigh-Taylor simulation output.
"""

import os
import re
from typing import Dict, Optional, Tuple, List
import numpy as np

from ..core.data_types import SegmentArray, BoundingBox
from .conrec_optimizer import optimize_conrec_for_vtk


class VTKReader:
    """
    High-performance VTK reader for RECTILINEAR_GRID format.

    Optimized for RT simulation data with cell-centered volume fraction fields.
    """

    def __init__(self):
        self.interface_value = 0.5  # Default volume fraction for interface extraction

    @staticmethod
    def parse_filename(vtk_file: str) -> Dict:
        """
        Parse VTK filename to extract grid resolution and simulation time.

        Expected format: RTnnnxmmm-tttt.vtk or similar patterns.

        Returns:
            Dictionary with resolution (nx, ny) and simulation time info
        """
        filename = os.path.basename(vtk_file)

        # Try RT format: RTnnnxmmm-tttt.vtk
        pattern = r'RT(\d+)x(\d+)-(\d+)\.vtk'
        match = re.match(pattern, filename)

        if match:
            nx = int(match.group(1))
            ny = int(match.group(2))
            time_step = int(match.group(3))

            return {
                'valid': True,
                'nx': nx,
                'ny': ny,
                'time_step': time_step,
                'filename': filename
            }

        return {'valid': False, 'filename': filename}

    def read_vtk_file(self, vtk_file: str) -> Dict:
        """
        Read VTK RECTILINEAR_GRID file and extract all data fields.

        Based on the proven algorithm from FractalAnalyzer v2 but optimized.

        Args:
            vtk_file: Path to VTK file

        Returns:
            Dictionary containing:
            - x, y: Grid coordinates (cell centers)
            - F: Volume fraction field
            - U, V: Velocity fields (if available)
            - P: Pressure field (if available)
            - nx, ny: Grid dimensions
            - bbox: Domain bounding box
        """
        if not os.path.exists(vtk_file):
            raise FileNotFoundError(f"VTK file not found: {vtk_file}")

        with open(vtk_file, 'r') as f:
            lines = f.readlines()

        # Parse header information
        dimensions = self._parse_dimensions(lines)
        coordinates = self._parse_coordinates(lines)
        scalar_fields = self._parse_scalar_fields(lines)

        # Handle cell-centered vs node-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)

        if is_cell_data:
            # Convert node coordinates to cell centers
            nx_cells = dimensions['nx'] - 1
            ny_cells = dimensions['ny'] - 1

            x_cell = 0.5 * (coordinates['x'][:-1] + coordinates['x'][1:])
            y_cell = 0.5 * (coordinates['y'][:-1] + coordinates['y'][1:])

            grid_shape = (nx_cells, ny_cells)
        else:
            x_cell = coordinates['x']
            y_cell = coordinates['y']
            grid_shape = (dimensions['nx'], dimensions['ny'])

        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(x_cell, y_cell, indexing='ij')

        # Reshape scalar fields to 2D grids
        reshaped_fields = {}
        for field_name, data in scalar_fields.items():
            if len(data) == grid_shape[0] * grid_shape[1]:
                # Reshape: data is stored in column-major order (Fortran-style)
                reshaped_fields[field_name] = data.reshape(grid_shape, order='F')
            else:
                print(f"Warning: Field {field_name} size mismatch. Expected {grid_shape[0] * grid_shape[1]}, got {len(data)}")

        # Create bounding box
        bbox = BoundingBox(
            min_x=float(x_cell[0]),
            min_y=float(y_cell[0]),
            max_x=float(x_cell[-1]),
            max_y=float(y_cell[-1])
        )

        # Prepare output data
        vtk_data = {
            'x': x_grid,
            'y': y_grid,
            'nx': grid_shape[0],
            'ny': grid_shape[1],
            'bbox': bbox,
            'filename': os.path.basename(vtk_file)
        }

        # Add all scalar fields (F, U, V, P, etc.)
        vtk_data.update(reshaped_fields)

        # Ensure volume fraction field is available
        if 'F' not in vtk_data:
            available_fields = list(reshaped_fields.keys())
            raise ValueError(f"Volume fraction field 'F' not found. Available fields: {available_fields}")

        return vtk_data

    def _parse_dimensions(self, lines: List[str]) -> Dict:
        """Parse DIMENSIONS line."""
        for line in lines:
            if "DIMENSIONS" in line:
                parts = line.strip().split()
                return {
                    'nx': int(parts[1]),
                    'ny': int(parts[2]),
                    'nz': int(parts[3])
                }
        raise ValueError("DIMENSIONS not found in VTK file")

    def _parse_coordinates(self, lines: List[str]) -> Dict:
        """Parse X_COORDINATES and Y_COORDINATES."""
        coordinates = {}

        for i, line in enumerate(lines):
            if "X_COORDINATES" in line:
                coordinates['x'] = self._read_coordinate_data(lines, i)
            elif "Y_COORDINATES" in line:
                coordinates['y'] = self._read_coordinate_data(lines, i)

        if 'x' not in coordinates or 'y' not in coordinates:
            raise ValueError("Coordinate data not found in VTK file")

        return coordinates

    def _read_coordinate_data(self, lines: List[str], start_line: int) -> np.ndarray:
        """Read coordinate data starting from specified line."""
        # Parse the coordinate header line
        parts = lines[start_line].strip().split()
        n_coords = int(parts[1])

        # Read coordinate values
        coords_data = []
        j = start_line + 1

        while len(coords_data) < n_coords and j < len(lines):
            line_data = lines[j].strip()
            if line_data and not line_data.startswith(('X_COORDINATES', 'Y_COORDINATES', 'Z_COORDINATES')):
                coords_data.extend(list(map(float, line_data.split())))
            j += 1

        if len(coords_data) != n_coords:
            raise ValueError(f"Expected {n_coords} coordinates, got {len(coords_data)}")

        return np.array(coords_data)

    def _parse_scalar_fields(self, lines: List[str]) -> Dict:
        """Parse all SCALARS fields (F, U, V, P, etc.)."""
        scalar_fields = {}

        for i, line in enumerate(lines):
            if line.strip().startswith("SCALARS"):
                field_name, field_data = self._read_scalar_field(lines, i)
                scalar_fields[field_name] = field_data

        return scalar_fields

    def _read_scalar_field(self, lines: List[str], start_line: int) -> Tuple[str, np.ndarray]:
        """Read a single SCALARS field."""
        # Parse SCALARS header
        parts = lines[start_line].strip().split()
        if len(parts) < 2:
            raise ValueError(f"Invalid SCALARS line: {lines[start_line]}")

        field_name = parts[1]

        # Read data values (skip LOOKUP_TABLE line)
        data_values = []
        j = start_line + 2  # Skip SCALARS and LOOKUP_TABLE lines

        while j < len(lines):
            line = lines[j].strip()

            # Stop if we hit another SCALARS field or end of data
            if line.startswith("SCALARS") or not line:
                if not line:  # Empty line - continue reading
                    j += 1
                    continue
                else:  # New SCALARS field - stop
                    break

            # Parse numeric data
            try:
                data_values.extend(list(map(float, line.split())))
            except ValueError:
                # Non-numeric line - stop reading this field
                break

            j += 1

        return field_name, np.array(data_values)

    def extract_interface(self, volume_fraction: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray,
                         interface_value: float = 0.5, use_conrec: bool = True) -> SegmentArray:
        """
        Extract interface contour from volume fraction field.

        Uses optimized CONREC algorithm with collinear segment merging to solve
        the over-segmentation problem for early-time RT interfaces.

        Args:
            volume_fraction: 2D volume fraction field
            x_grid, y_grid: Coordinate grids
            interface_value: Volume fraction value for interface (default: 0.5)
            use_conrec: If True, use optimized CONREC; if False, fall back to marching squares

        Returns:
            SegmentArray containing optimally merged interface line segments
        """
        if use_conrec:
            # Use optimized CONREC algorithm with collinear merging
            return optimize_conrec_for_vtk(volume_fraction, x_grid, y_grid, interface_value)
        else:
            # Fall back to marching squares (original method)
            return self._extract_interface_marching_squares(volume_fraction, x_grid, y_grid, interface_value)

    def _extract_interface_marching_squares(self, volume_fraction: np.ndarray, x_grid: np.ndarray,
                                          y_grid: np.ndarray, interface_value: float = 0.5) -> SegmentArray:
        """
        Original marching squares implementation (kept as fallback).
        """
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image required for interface extraction. Install with: pip install scikit-image")

        # Extract contours using marching squares
        contours = measure.find_contours(volume_fraction, interface_value)

        if not contours:
            print(f"Warning: No interface found at F = {interface_value}")
            return SegmentArray.from_list([])

        # Convert contours to line segments
        segments = []

        for contour in contours:
            # Convert pixel coordinates to physical coordinates
            # contour contains (row, col) indices, need to interpolate to (x, y)

            for i in range(len(contour) - 1):
                # Get pixel indices
                r1, c1 = contour[i]
                r2, c2 = contour[i + 1]

                # Convert to integer indices for grid lookup
                r1_int, c1_int = int(np.round(r1)), int(np.round(c1))
                r2_int, c2_int = int(np.round(r2)), int(np.round(c2))

                # Bounds checking
                if (0 <= r1_int < volume_fraction.shape[0] and 0 <= c1_int < volume_fraction.shape[1] and
                    0 <= r2_int < volume_fraction.shape[0] and 0 <= c2_int < volume_fraction.shape[1]):

                    # Get physical coordinates
                    x1, y1 = x_grid[r1_int, c1_int], y_grid[r1_int, c1_int]
                    x2, y2 = x_grid[r2_int, c2_int], y_grid[r2_int, c2_int]

                    segments.append(((x1, y1), (x2, y2)))

        if not segments:
            print("Warning: No valid segments extracted from interface")
            return SegmentArray.from_list([])

        return SegmentArray.from_list(segments)

    def analyze_vtk_file(self, vtk_file: str, interface_value: float = 0.5) -> Dict:
        """
        Complete analysis: read VTK file and extract interface for fractal analysis.

        Args:
            vtk_file: Path to VTK file
            interface_value: Volume fraction value for interface extraction

        Returns:
            Dictionary containing VTK data and extracted interface segments
        """
        # Read VTK file
        vtk_data = self.read_vtk_file(vtk_file)

        # Extract interface
        segments = self.extract_interface(
            vtk_data['F'],
            vtk_data['x'],
            vtk_data['y'],
            interface_value
        )

        # Add interface data to results
        result = vtk_data.copy()
        result['interface_segments'] = segments
        result['interface_value'] = interface_value
        result['n_interface_segments'] = segments.n_segments

        return result