# rt_analyzer.py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
import re
import time
import glob
from typing import Tuple, List, Dict, Optional
from skimage import measure

# GPU acceleration for multifractal box counting
try:
    from .fast_counting_gpu_2d import HAS_CUDA, count_segments_per_box_gpu, _prepare_segments_array
except ImportError:
    try:
        from fast_counting_gpu_2d import HAS_CUDA, count_segments_per_box_gpu, _prepare_segments_array
    except ImportError:
        HAS_CUDA = False

def _parse_openfoam_points(points_file):
    """Parse OpenFOAM constant/polyMesh/points → (x_unique_sorted, y_unique_sorted).

    Args:
        points_file: Path to the points file

    Returns:
        (x_unique, y_unique): Sorted unique x and y node coordinates as numpy arrays
    """
    with open(points_file, 'r') as f:
        content = f.read()

    # Find the data block: count followed by ( ... )
    # Skip header, find the integer count before the opening '('
    lines = content.split('\n')
    count = None
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip header/comments
        if stripped.startswith('/*') or stripped.startswith('\\') or stripped.startswith('//'):
            continue
        if stripped.startswith('FoamFile') or stripped == '{' or stripped == '}':
            continue
        if stripped.startswith('format') or stripped.startswith('class') or \
           stripped.startswith('location') or stripped.startswith('object'):
            continue
        if stripped == '':
            continue
        # First non-header integer is the count
        if count is None:
            try:
                count = int(stripped)
                continue
            except ValueError:
                continue
        # Next should be '('
        if stripped == '(':
            data_start = i + 1
            break

    if count is None or data_start is None:
        raise ValueError(f"Could not parse points file: {points_file}")

    # Parse (x y z) tuples
    x_vals = []
    y_vals = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ')':
            break
        # Remove parentheses: "(x y z)" → "x y z"
        line = line.strip('()')
        parts = line.split()
        if len(parts) >= 2:
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))

    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)
    return x_unique, y_unique


def _parse_openfoam_scalar_field(field_file):
    """Parse an OpenFOAM volScalarField file → numpy array of values.

    Args:
        field_file: Path to the field file (e.g. alpha.water)

    Returns:
        np.ndarray of scalar values
    """
    with open(field_file, 'r') as f:
        lines = f.readlines()

    # Find 'internalField' line, then count and ( ... ) block
    count = None
    data_start = None
    for i, line in enumerate(lines):
        if 'internalField' in line and 'nonuniform' in line:
            # Count is on the next line (or same line for some formats)
            # Typically: "internalField   nonuniform List<scalar>"
            # Then: count
            # Then: (
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped == '':
                    j += 1
                    continue
                try:
                    count = int(stripped)
                    j += 1
                    break
                except ValueError:
                    j += 1
                    continue
            # Find opening '('
            while j < len(lines):
                if lines[j].strip() == '(':
                    data_start = j + 1
                    break
                j += 1
            break

    if count is None or data_start is None:
        raise ValueError(f"Could not parse scalar field: {field_file}")

    # Read values
    values = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ')':
            break
        if line == '':
            continue
        try:
            values.append(float(line))
        except ValueError:
            # Handle multiple values per line
            for v in line.split():
                try:
                    values.append(float(v))
                except ValueError:
                    pass

    return np.array(values[:count])


def _parse_openfoam_vector_field(field_file):
    """Parse an OpenFOAM volVectorField file -> (Ux, Uy, Uz) numpy arrays.

    Args:
        field_file: Path to the field file (e.g. U)

    Returns:
        tuple of (Ux, Uy, Uz) numpy arrays
    """
    with open(field_file, 'r') as f:
        content = f.read()

    match = re.search(
        r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(\s*\n(.*?)\n\)\s*;',
        content, re.DOTALL)

    if not match:
        raise ValueError(f"Could not parse vector field: {field_file}")

    count = int(match.group(1))
    data_str = match.group(2)
    vectors = re.findall(r'\(([^)]+)\)', data_str)

    ux = np.zeros(count)
    uy = np.zeros(count)
    uz = np.zeros(count)
    for i, vec in enumerate(vectors[:count]):
        parts = vec.split()
        ux[i] = float(parts[0])
        uy[i] = float(parts[1])
        uz[i] = float(parts[2])

    return ux, uy, uz


def _find_openfoam_time_dir(case_dir, time):
    """Find the time directory matching a float time value.

    Args:
        case_dir: Path to OpenFOAM case directory
        time: Time value (float or str)

    Returns:
        Path to the matching time directory
    """
    time_val = float(time)

    # List all entries in case_dir
    candidates = []
    for entry in os.listdir(case_dir):
        entry_path = os.path.join(case_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            t = float(entry)
            candidates.append((t, entry))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"No time directories found in {case_dir}")

    # Find closest match
    candidates.sort(key=lambda x: abs(x[0] - time_val))
    best_t, best_name = candidates[0]

    if abs(best_t - time_val) > 1e-6:
        # Check if it's a reasonable match (within 1% or 0.01)
        if abs(best_t - time_val) > max(0.01, abs(time_val) * 0.01):
            raise ValueError(
                f"No time directory matching t={time_val} in {case_dir}. "
                f"Closest: {best_name} (t={best_t})")

    return os.path.join(case_dir, best_name)


class RTAnalyzer:
    """Complete Rayleigh-Taylor simulation analyzer with fractal dimension calculation."""

    def __init__(self, output_dir="./rt_analysis", rt_physics=None):
        """Initialize the RT analyzer.

        Args:
            output_dir: Output directory for results
            rt_physics: Optional RTPhysics instance for nondimensionalization.
                        When provided, results include dimensionless quantities.
        """
        self.output_dir = output_dir
        self.rt_physics = rt_physics
        os.makedirs(output_dir, exist_ok=True)

        # Create fractal analyzer instance
        try:
            from fractal_analyzer.core.fractal_analyzer import FractalAnalyzer
            self.fractal_analyzer = FractalAnalyzer()
            print("Fractal analyzer initialized successfully")
        except ImportError:
            print("Warning: fractal_analyzer module not found. Fractal dimension calculation may not work.")
            self.fractal_analyzer = None
    
    def read_vtk_file(self, vtk_file):
        """Read VTK rectilinear grid file and extract only the VOF (F) data."""
        with open(vtk_file, 'r') as f:
            lines = f.readlines()
        
        # Extract dimensions
        for i, line in enumerate(lines):
            if "DIMENSIONS" in line:
                parts = line.strip().split()
                nx, ny, nz = int(parts[1]), int(parts[2]), int(parts[3])
                break
        
        # Extract coordinates
        x_coords = []
        y_coords = []
        
        # Find coordinates
        for i, line in enumerate(lines):
            if "X_COORDINATES" in line:
                parts = line.strip().split()
                n_coords = int(parts[1])
                coords_data = []
                j = i + 1
                while len(coords_data) < n_coords:
                    coords_data.extend(list(map(float, lines[j].strip().split())))
                    j += 1
                x_coords = np.array(coords_data)
            
            if "Y_COORDINATES" in line:
                parts = line.strip().split()
                n_coords = int(parts[1])
                coords_data = []
                j = i + 1
                while len(coords_data) < n_coords:
                    coords_data.extend(list(map(float, lines[j].strip().split())))
                    j += 1
                y_coords = np.array(coords_data)
        
        # Extract all scalar field data (F, P, U, V)
        scalar_fields = {}

        for i, line in enumerate(lines):
            if line.strip().startswith("SCALARS "):
                parts = line.strip().split()
                field_name = parts[1]
                data_values = []
                j = i + 2  # Skip the LOOKUP_TABLE line
                while j < len(lines) and not lines[j].strip().startswith("SCALARS"):
                    vals = lines[j].strip().split()
                    if vals:
                        try:
                            data_values.extend(list(map(float, vals)))
                        except ValueError:
                            break
                    j += 1
                scalar_fields[field_name] = np.array(data_values)

        f_data = scalar_fields.get('F')
        
        # Check if this is cell-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)
        
        # For cell data, we need to adjust the grid
        if is_cell_data:
            # The dimensions are one less than the coordinates in each direction
            nx_cells, ny_cells = nx-1, ny-1
            
            # Reshape the data
            f_grid = f_data.reshape(ny_cells, nx_cells).T if f_data is not None else None
            
            # Create cell-centered coordinates
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
            
            # Create 2D meshgrid
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid = x_grid.T  # Transpose to match the data ordering
            y_grid = y_grid.T
        else:
            # For point data, use the coordinates directly
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid = x_grid.T
            y_grid = y_grid.T
            
            # Reshape the data
            f_grid = f_data.reshape(ny, nx).T if f_data is not None else None
        
        # Extract simulation time from filename
        time_match = re.search(r'(\d+)\.vtk$', os.path.basename(vtk_file))
        sim_time = float(time_match.group(1))/1000.0 if time_match else 0.0
        
        # Reshape additional scalar fields (U, V, P) using same logic as F
        extra_fields = {}
        for fname, fdata in scalar_fields.items():
            if fname == 'F' or fdata is None:
                continue
            try:
                if is_cell_data:
                    extra_fields[fname] = fdata.reshape(ny_cells, nx_cells).T
                else:
                    extra_fields[fname] = fdata.reshape(ny, nx).T
            except ValueError:
                pass  # Skip fields that don't match expected size

        # Create output dictionary
        result = {
            'x': x_grid,
            'y': y_grid,
            'f': f_grid,
            'dims': (nx, ny, nz),
            'time': sim_time
        }
        result.update(extra_fields)
        return result

    def read_openfoam_field(self, case_dir, time, field_name='alpha.water'):
        """Read an OpenFOAM 2D field from a case directory.

        Args:
            case_dir: Path to OpenFOAM case directory (contains constant/, system/, time dirs)
            time: Time value (float or str) — used to find the time directory
            field_name: Name of the scalar field file (default: 'alpha.water')

        Returns:
            dict with keys 'x', 'y', 'f', 'dims', 'time' — same contract as read_vtk_file()
        """
        # Parse mesh points to get node coordinates
        points_file = os.path.join(case_dir, 'constant', 'polyMesh', 'points')
        x_nodes, y_nodes = _parse_openfoam_points(points_file)

        # Compute cell centers
        x_cell = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        y_cell = 0.5 * (y_nodes[:-1] + y_nodes[1:])

        nx = len(x_cell)
        ny = len(y_cell)

        # Find and parse the field file
        time_dir = _find_openfoam_time_dir(case_dir, time)
        field_file = os.path.join(time_dir, field_name)

        if not os.path.isfile(field_file):
            raise FileNotFoundError(f"Field file not found: {field_file}")

        f_data = _parse_openfoam_scalar_field(field_file)

        # Reshape: OpenFOAM blockMesh structured grids have x varying fastest
        f_grid = f_data.reshape(ny, nx).T

        # Create meshgrid + transpose (same as VTK cell-data path)
        x_grid, y_grid = np.meshgrid(x_cell, y_cell)
        x_grid = x_grid.T
        y_grid = y_grid.T

        time_val = float(os.path.basename(time_dir))

        result = {
            'x': x_grid,
            'y': y_grid,
            'f': f_grid,
            'dims': (len(x_nodes), len(y_nodes), 2),
            'time': time_val
        }

        # Try to read velocity field U (vector) -> split into U, V components
        u_file = os.path.join(time_dir, 'U')
        if os.path.isfile(u_file):
            try:
                ux, uy, uz = _parse_openfoam_vector_field(u_file)
                result['U'] = ux.reshape(ny, nx).T
                result['V'] = uy.reshape(ny, nx).T
            except Exception:
                pass

        # Try to read pressure field p_rgh
        p_file = os.path.join(time_dir, 'p_rgh')
        if os.path.isfile(p_file):
            try:
                p_data = _parse_openfoam_scalar_field(p_file)
                result['P'] = p_data.reshape(ny, nx).T
            except Exception:
                pass

        return result

    def read_data(self, path, time=None, field_name='alpha.water'):
        """Auto-detect format and read simulation data.

        Args:
            path: Either a .vtk file path OR an OpenFOAM case directory
            time: Required for OpenFOAM (which timestep to read). Ignored for VTK.
            field_name: OpenFOAM field name (default: 'alpha.water')

        Returns:
            dict with 'x', 'y', 'f', 'dims', 'time'
        """
        if path.endswith('.vtk'):
            return self.read_vtk_file(path)
        elif os.path.isdir(path):
            if time is None:
                raise ValueError("time is required for OpenFOAM case directories")
            return self.read_openfoam_field(path, time, field_name)
        else:
            raise ValueError(f"Cannot detect format for: {path}")

    def extract_interface(self, f_grid, x_grid, y_grid, level=0.5):
        """Extract the interface contour at level f=0.5 using marching squares algorithm."""
        # Find contours
        contours = measure.find_contours(f_grid.T, level)
        
        # Convert to physical coordinates
        physical_contours = []
        for contour in contours:
            x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
            y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
            physical_contours.append(np.column_stack([x_physical, y_physical]))
        
        return physical_contours
    
    def convert_contours_to_segments(self, contours):
        """Convert contours to line segments format for fractal analysis."""
        segments = []
        
        for contour in contours:
            for i in range(len(contour) - 1):
                x1, y1 = contour[i]
                x2, y2 = contour[i+1]
                segments.append(((x1, y1), (x2, y2)))
        
        return segments
    
    def find_initial_interface(self, data):
        """Find the initial interface position (y=1.0 for RT)."""
        f_avg = np.mean(data['f'], axis=0)
        y_values = data['y'][0, :]
        
        # Find where f crosses 0.5
        idx = np.argmin(np.abs(f_avg - 0.5))
        return y_values[idx]
    
    def compute_mixing_thickness(self, data, h0, method='geometric'):
        """Compute mixing layer thickness using different methods."""
        if method == 'geometric':
            # Extract interface contours
            contours = self.extract_interface(data['f'], data['x'], data['y'])
            
            # Find maximum displacement above and below initial interface
            ht = 0.0
            hb = 0.0
            
            for contour in contours:
                y_coords = contour[:, 1]
                ht = max(ht, np.max(y_coords - h0))
                hb = max(hb, np.max(h0 - y_coords))
            
            return {'ht': ht, 'hb': hb, 'h_total': ht + hb}
            
        elif method == 'statistical':
            # Use concentration thresholds to define mixing zone
            f_avg = np.mean(data['f'], axis=0)
            y_values = data['y'][0, :]

            epsilon = 0.01  # Threshold for "pure" fluid

            # Detect orientation: if f increases with y, heavy fluid is on top
            # (RT convention: F=1 is tracked/heavy fluid)
            f_increases = f_avg[-1] > f_avg[0]

            if f_increases:
                # F increases with y: heavy on top, light on bottom
                # Upper boundary: highest y where f < 1-ε (light fluid bubbles)
                upper_idx = np.where(f_avg < 1 - epsilon)[0]
                y_upper = y_values[upper_idx[-1]] if len(upper_idx) > 0 else y_values[-1]

                # Lower boundary: lowest y where f > ε (heavy fluid spikes)
                lower_idx = np.where(f_avg > epsilon)[0]
                y_lower = y_values[lower_idx[0]] if len(lower_idx) > 0 else y_values[0]
            else:
                # F decreases with y: heavy on bottom, light on top
                # Upper boundary: highest y where f > ε (heavy fluid rising)
                upper_idx = np.where(f_avg > epsilon)[0]
                y_upper = y_values[upper_idx[-1]] if len(upper_idx) > 0 else y_values[-1]

                # Lower boundary: lowest y where f < 1-ε (light fluid sinking)
                lower_idx = np.where(f_avg < 1 - epsilon)[0]
                y_lower = y_values[lower_idx[0]] if len(lower_idx) > 0 else y_values[0]

            # Calculate thicknesses
            ht = max(0, y_upper - h0)
            hb = max(0, h0 - y_lower)

            return {'ht': ht, 'hb': hb, 'h_total': ht + hb}

        elif method == 'integral':
            # Dalziel et al. (1999) Eq. 7: row-average F then integrate
            # over each half-domain.
            #
            # All 4 fluid/region combinations are computed independently
            # (conservation is NOT imposed — checking h_{m,0}+h_{m,1} = H/2
            # diagnoses numerical mass conservation in the VOF scheme).
            #
            # Fluid 1 = heavy (F=1), Fluid 0 = light (F=0):
            #   h_{1,0} = int_0^{h0} <F>(y) dy     heavy in lower (penetration)
            #   h_{1,1} = int_{h0}^{H} <F>(y) dy    heavy in upper (own region)
            #   h_{0,0} = int_0^{h0} <1-F>(y) dy    light in lower (own region)
            #   h_{0,1} = int_{h0}^{H} <1-F>(y) dy  light in upper (penetration)
            #
            # Conservation checks (should each = H/2 if mass is conserved):
            #   h_{1,0} + h_{0,0} = H/2   (lower half sums to full region)
            #   h_{1,1} + h_{0,1} = H/2   (upper half sums to full region)
            #   h_{1,0} + h_{1,1} = total heavy fluid mass / (rho * L)
            #   h_{0,0} + h_{0,1} = total light fluid mass / (rho * L)
            #
            # Return convention matches geometric/statistical:
            #   hb = h_{1,0}  (heavy penetrating down)
            #   ht = h_{0,1}  (light penetrating up)
            #   h_total = hb + ht

            f_avg = np.mean(data['f'], axis=0)   # <F>(y), shape (ny,)
            y_values = data['y'][0, :]            # y coordinates

            # Detect orientation
            f_increases = f_avg[-1] > f_avg[0]

            # Split into lower (y < h0) and upper (y >= h0) regions
            lower = y_values <= h0
            upper = y_values >= h0

            y_lo = y_values[lower]
            y_up = y_values[upper]
            f_lo = f_avg[lower]
            f_up = f_avg[upper]

            if f_increases:
                # F increases with y → heavy fluid on top, F~1 above h0
                h_10 = float(np.trapz(f_lo, y_lo))         # heavy in lower
                h_11 = float(np.trapz(f_up, y_up))         # heavy in upper
                h_00 = float(np.trapz(1.0 - f_lo, y_lo))   # light in lower
                h_01 = float(np.trapz(1.0 - f_up, y_up))   # light in upper
            else:
                # F decreases with y → heavy on bottom, F~1 below h0
                # "heavy" is the F~1 fluid (bottom), "light" is the F~0 fluid (top)
                h_10 = float(np.trapz(1.0 - f_lo, y_lo))
                h_11 = float(np.trapz(1.0 - f_up, y_up))
                h_00 = float(np.trapz(f_lo, y_lo))
                h_01 = float(np.trapz(f_up, y_up))

            # Map to ht/hb convention:
            #   hb = h_{1,0} (heavy going down)
            #   ht = h_{0,1} (light going up)
            hb = h_10
            ht = h_01

            return {
                'ht': ht, 'hb': hb, 'h_total': ht + hb,
                'h_10': h_10, 'h_11': h_11,
                'h_00': h_00, 'h_01': h_01,
            }

    def compute_fractal_dimension(self, data, min_box_size=0.001):
        """Compute fractal dimension of the interface."""
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping fractal dimension calculation.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }
        
        # Extract contours
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        
        # Convert to segments
        segments = self.convert_contours_to_segments(contours)
        
        if not segments:
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }
        
        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2
        
        # Perform box counting
        box_sizes, box_counts, bounding_box = (
            self.fractal_analyzer.box_counting_unified(
                segments, min_box_size, max_box_size, box_size_factor=1.5)
        )

        # Calculate fractal dimension from box counting data
        dimension, error, intercept = self.fractal_analyzer.calculate_fractal_dimension(
            box_sizes, box_counts)

        # Calculate R-squared
        log_sizes = np.log(box_sizes)
        log_counts = np.log(box_counts)
        _, _, r_value, _, _ = stats.linregress(log_sizes, log_counts)
        r_squared = r_value**2

        result = {
            'dimension': dimension,
            'error': error,
            'r_squared': r_squared,
            'box_sizes': box_sizes,
            'box_counts': box_counts,
            'bounding_box': bounding_box,
            'segments': segments
        }

        if self.rt_physics is not None:
            result['box_sizes_nondim'] = self.rt_physics.nondim_box_size(box_sizes)

        return result
    
    def analyze_vtk_file(self, vtk_file, output_subdir=None):
        """Perform complete analysis on a single VTK file."""
        # Create subdirectory for this file if needed
        if output_subdir:
            file_dir = os.path.join(self.output_dir, output_subdir)
        else:
            basename = os.path.basename(vtk_file).split('.')[0]
            file_dir = os.path.join(self.output_dir, basename)
        
        os.makedirs(file_dir, exist_ok=True)
        
        print(f"Analyzing {vtk_file}...")
        
        # Read VTK file
        start_time = time.time()
        data = self.read_vtk_file(vtk_file)
        print(f"VTK file read in {time.time() - start_time:.2f} seconds")
        
        # Find initial interface position
        h0 = self.find_initial_interface(data)
        print(f"Initial interface position: {h0:.6f}")
        
        # Compute mixing thickness
        mixing = self.compute_mixing_thickness(data, h0, method='geometric')
        print(f"Mixing thickness: {mixing['h_total']:.6f} (ht={mixing['ht']:.6f}, hb={mixing['hb']:.6f})")
        
        # Extract interface for visualization and save to file
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        interface_file = os.path.join(file_dir, 'interface.dat')
        
        with open(interface_file, 'w') as f:
            f.write(f"# Interface data for t = {data['time']:.6f}\n")
            segment_count = 0
            for contour in contours:
                for i in range(len(contour) - 1):
                    f.write(f"{contour[i,0]:.7f},{contour[i,1]:.7f} {contour[i+1,0]:.7f},{contour[i+1,1]:.7f}\n")
                    segment_count += 1
            f.write(f"# Found {segment_count} contour segments\n")
        
        print(f"Interface saved to {interface_file} ({segment_count} segments)")
        
        # Compute fractal dimension
        fd_start_time = time.time()
        fd_results = self.compute_fractal_dimension(data)
        print(f"Fractal dimension: {fd_results['dimension']:.6f} ± {fd_results['error']:.6f} (R²={fd_results['r_squared']:.6f})")
        print(f"Fractal calculation time: {time.time() - fd_start_time:.2f} seconds")
        
        # Visualize interface and box counting
        if fd_results['dimension'] > 0:
            fig = plt.figure(figsize=(12, 10))
            plt.contourf(data['x'], data['y'], data['f'], levels=20, cmap='viridis')
            plt.colorbar(label='Volume Fraction')
            
            # Plot interface
            for contour in contours:
                plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2)
            
            # Plot initial interface position
            plt.axhline(y=h0, color='k', linestyle='--', alpha=0.5, label=f'Initial Interface (y={h0:.4f})')
            
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'Rayleigh-Taylor Interface at t = {data["time"]:.3f}')
            plt.grid(True)
            plt.savefig(os.path.join(file_dir, 'interface_plot.png'), dpi=300)
            plt.close()
            
            # Plot box counting results
            fig = plt.figure(figsize=(10, 8))
            plt.loglog(fd_results['box_sizes'], fd_results['box_counts'], 'bo-', label='Data')
            
            # Linear regression line
            log_sizes = np.log(fd_results['box_sizes'])
            slope = -fd_results['dimension']
            intercept = log_sizes[0] + log_sizes[-1]  # Placeholder if actual intercept not available
            fit_counts = np.exp(intercept + slope * log_sizes)
            plt.loglog(fd_results['box_sizes'], fit_counts, 'r-', label=f"D = {fd_results['dimension']:.4f} ± {fd_results['error']:.4f}")
            
            plt.xlabel('Box Size')
            plt.ylabel('Box Count')
            plt.title(f'Fractal Dimension at t = {data["time"]:.3f}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(file_dir, 'fractal_dimension.png'), dpi=300)
            plt.close()
        
        # Return analysis results
        result = {
            'time': data['time'],
            'h0': h0,
            'ht': mixing['ht'],
            'hb': mixing['hb'],
            'h_total': mixing['h_total'],
            'fractal_dim': fd_results['dimension'],
            'fd_error': fd_results['error'],
            'fd_r_squared': fd_results['r_squared']
        }

        if self.rt_physics is not None:
            result['tau'] = float(self.rt_physics.nondim_time(data['time']))
            result['ht_nondim'] = float(self.rt_physics.nondim_length(mixing['ht']))
            result['hb_nondim'] = float(self.rt_physics.nondim_length(mixing['hb']))
            result['h_total_nondim'] = float(self.rt_physics.nondim_length(mixing['h_total']))

        return result
    
    def process_vtk_series(self, vtk_pattern, resolution=None):
        """Process a series of VTK files matching the given pattern."""
        # Find all matching VTK files
        vtk_files = sorted(glob.glob(vtk_pattern))
        
        if not vtk_files:
            raise ValueError(f"No VTK files found matching pattern: {vtk_pattern}")
        
        print(f"Found {len(vtk_files)} VTK files matching {vtk_pattern}")
        
        # Create subdirectory for this resolution if provided
        if resolution:
            subdir = f"res_{resolution}"
        else:
            subdir = "results"
        
        results_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each file
        results = []
        
        for i, vtk_file in enumerate(vtk_files):
            print(f"\nProcessing file {i+1}/{len(vtk_files)}: {vtk_file}")
            
            try:
                # Analyze this file
                result = self.analyze_vtk_file(vtk_file, subdir)
                results.append(result)
                
                # Print progress
                print(f"Completed {i+1}/{len(vtk_files)} files")
                
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary dataframe
        if results:
            df = pd.DataFrame(results)
            
            # Save results
            csv_file = os.path.join(results_dir, 'results_summary.csv')
            df.to_csv(csv_file, index=False)
            print(f"Results saved to {csv_file}")
            
            # Create summary plots
            self.create_summary_plots(df, results_dir)
            
            return df
        else:
            print("No results to summarize")
            return None
    
    def analyze_resolution_convergence(self, vtk_files, resolutions, target_time=9.0):
        """Analyze how fractal dimension and mixing thickness converge with grid resolution."""
        results = []
        
        for vtk_file, resolution in zip(vtk_files, resolutions):
            print(f"\nAnalyzing resolution {resolution}x{resolution} using {vtk_file}")
            
            try:
                # Read and analyze the file
                data = self.read_vtk_file(vtk_file)
                
                # Check if time matches target
                if abs(data['time'] - target_time) > 0.1:
                    print(f"Warning: File time {data['time']} differs from target {target_time}")
                
                # Find initial interface
                h0 = self.find_initial_interface(data)
                
                # Calculate mixing thickness
                mixing = self.compute_mixing_thickness(data, h0)
                
                # Calculate fractal dimension
                fd_results = self.compute_fractal_dimension(data)
                
                # Save results
                results.append({
                    'resolution': resolution,
                    'time': data['time'],
                    'h0': h0,
                    'ht': mixing['ht'],
                    'hb': mixing['hb'],
                    'h_total': mixing['h_total'],
                    'fractal_dim': fd_results['dimension'],
                    'fd_error': fd_results['error'],
                    'fd_r_squared': fd_results['r_squared']
                })
                
            except Exception as e:
                print(f"Error analyzing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)
            
            # Create output directory
            convergence_dir = os.path.join(self.output_dir, f"convergence_t{target_time}")
            os.makedirs(convergence_dir, exist_ok=True)
            
            # Save results
            csv_file = os.path.join(convergence_dir, 'resolution_convergence.csv')
            df.to_csv(csv_file, index=False)
            
            # Create convergence plots
            self._plot_resolution_convergence(df, target_time, convergence_dir)
            
            return df
        else:
            print("No results to analyze")
            return None
    
    def _plot_resolution_convergence(self, df, target_time, output_dir):
        """Plot resolution convergence results."""
        # Plot fractal dimension vs resolution
        plt.figure(figsize=(10, 8))
        
        plt.errorbar(df['resolution'], df['fractal_dim'], yerr=df['fd_error'],
                    fmt='o-', capsize=5, elinewidth=1, markersize=8)
        
        plt.xscale('log', base=2)  # Use log scale with base 2
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Fractal Dimension at t={target_time}')
        plt.title(f'Fractal Dimension Convergence at t={target_time}')
        plt.grid(True)
        
        # Add grid points as labels
        for i, res in enumerate(df['resolution']):
            plt.annotate(f"{res}×{res}", (df['resolution'].iloc[i], df['fractal_dim'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        # Add asymptote if enough points
        if len(df) >= 3:
            # Extrapolate to infinite resolution (1/N = 0)
            x = 1.0 / np.array(df['resolution'])
            y = df['fractal_dim']
            coeffs = np.polyfit(x[-3:], y[-3:], 1)
            asymptotic_value = coeffs[1]  # y-intercept
            
            plt.axhline(y=asymptotic_value, color='r', linestyle='--',
                       label=f"Extrapolated value: {asymptotic_value:.4f}")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dimension_convergence.png"), dpi=300)
        plt.close()
        
        # Plot mixing layer thickness convergence
        plt.figure(figsize=(10, 8))
        
        plt.plot(df['resolution'], df['h_total'], 'o-', markersize=8, label='Total')
        plt.plot(df['resolution'], df['ht'], 's--', markersize=6, label='Upper')
        plt.plot(df['resolution'], df['hb'], 'd--', markersize=6, label='Lower')
        
        plt.xscale('log', base=2)
        plt.xlabel('Grid Resolution')
        plt.ylabel(f'Mixing Layer Thickness at t={target_time}')
        plt.title(f'Mixing Layer Thickness Convergence at t={target_time}')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mixing_convergence.png"), dpi=300)
        plt.close()
    
    def create_summary_plots(self, df, output_dir):
        """Create summary plots of the time series results."""
        # Plot mixing layer evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['h_total'], 'b-', label='Total', linewidth=2)
        plt.plot(df['time'], df['ht'], 'r--', label='Upper', linewidth=2)
        plt.plot(df['time'], df['hb'], 'g--', label='Lower', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Mixing Layer Thickness')
        plt.title('Mixing Layer Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'mixing_evolution.png'), dpi=300)
        plt.close()
        
        # Plot fractal dimension evolution
        plt.figure(figsize=(10, 6))
        plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt='ko-', capsize=3, linewidth=2, markersize=5)
        plt.fill_between(df['time'], 
                       df['fractal_dim'] - df['fd_error'],
                       df['fractal_dim'] + df['fd_error'],
                       alpha=0.3, color='gray')
        plt.xlabel('Time')
        plt.ylabel('Fractal Dimension')
        plt.title('Fractal Dimension Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'dimension_evolution.png'), dpi=300)
        plt.close()
        
        # Plot R-squared evolution
        plt.figure(figsize=(10, 6))
        plt.plot(df['time'], df['fd_r_squared'], 'm-o', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('R² Value')
        plt.title('Fractal Dimension Fit Quality')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'r_squared_evolution.png'), dpi=300)
        plt.close()
        
        # Combined plot with mixing layer and fractal dimension
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Mixing layer on left axis
        ax1.plot(df['time'], df['h_total'], 'b-', label='Mixing Thickness', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Mixing Layer Thickness', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Fractal dimension on right axis
        ax2 = ax1.twinx()
        ax2.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt='ro-', capsize=3, label='Fractal Dimension')
        ax2.set_ylabel('Fractal Dimension', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Mixing Layer and Fractal Dimension Evolution')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'combined_evolution.png'), dpi=300)
        plt.close()

    def compute_multifractal_spectrum(self, data, min_box_size=0.001, q_values=None, output_dir=None):
        """Compute multifractal spectrum of the interface.
        
        Args:
            data: Data dictionary containing VTK data
            min_box_size: Minimum box size for analysis (default: 0.001)
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            output_dir: Directory to save results (default: None)
            
        Returns:
            dict: Multifractal spectrum results
        """
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping multifractal analysis.")
            return None
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Extract contours and convert to segments
        contours = self.extract_interface(data['f'], data['x'], data['y'])
        segments = self.convert_contours_to_segments(contours)
        
        if not segments:
            print("No interface segments found. Skipping multifractal analysis.")
            return None
        
        # Create output directory if specified and not existing
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)
        
        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2
        
        print(f"Performing multifractal analysis with {len(q_values)} q-values")
        print(f"Box size range: {min_box_size:.6f} to {max_box_size:.6f}")
        
        # Generate box sizes
        box_sizes = []
        current_size = max_box_size
        box_size_factor = 1.5
        
        while current_size >= min_box_size:
            box_sizes.append(current_size)
            current_size /= box_size_factor
            
        box_sizes = np.array(box_sizes)
        num_box_sizes = len(box_sizes)
        
        print(f"Using {num_box_sizes} box sizes for analysis")
        
        # Use spatial index from FractalAnalyzer
        fa = self.fractal_analyzer
        
        # Add small margin to bounding box
        margin = extent * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin
        
        # Initialize data structures for box counting
        all_box_counts = []
        all_probabilities = []

        domain_min = np.array([min_x, min_y], dtype=np.float64)
        domain_max = np.array([max_x, max_y], dtype=np.float64)

        use_gpu = HAS_CUDA
        if use_gpu:
            print("Using GPU-accelerated box counting (CUDA)")
            seg_arr = _prepare_segments_array(segments)
        else:
            print("Using CPU box counting with spatial index")
            # Create spatial index for segments
            start_time = time.time()
            grid_size = min_box_size * 2
            segment_grid, grid_width, grid_height = fa.create_spatial_index(
                segments, min_x, min_y, max_x, max_y, grid_size)
            print(f"Spatial index created in {time.time() - start_time:.2f} seconds")

        # Analyze each box size
        for box_idx, box_size in enumerate(box_sizes):
            box_start_time = time.time()
            print(f"Processing box size {box_idx+1}/{num_box_sizes}: {box_size:.6f}")

            if use_gpu:
                box_counts, num_boxes_x, num_boxes_y = count_segments_per_box_gpu(
                    seg_arr, box_size, domain_min, domain_max)
            else:
                num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
                num_boxes_y = int(np.ceil((max_y - min_y) / box_size))

                box_counts = np.zeros((num_boxes_x, num_boxes_y))

                for i in range(num_boxes_x):
                    for j in range(num_boxes_y):
                        box_xmin = min_x + i * box_size
                        box_ymin = min_y + j * box_size
                        box_xmax = box_xmin + box_size
                        box_ymax = box_ymin + box_size

                        min_cell_x = max(0, int((box_xmin - min_x) / grid_size))
                        max_cell_x = min(int((box_xmax - min_x) / grid_size) + 1, grid_width)
                        min_cell_y = max(0, int((box_ymin - min_y) / grid_size))
                        max_cell_y = min(int((box_ymax - min_y) / grid_size) + 1, grid_height)

                        segments_to_check = set()
                        for cell_x in range(min_cell_x, max_cell_x):
                            for cell_y in range(min_cell_y, max_cell_y):
                                segments_to_check.update(segment_grid.get((cell_x, cell_y), []))

                        count = 0
                        for seg_idx in segments_to_check:
                            (x1, y1), (x2, y2) = segments[seg_idx]
                            if self.fractal_analyzer.liang_barsky_line_box_intersection(
                                    x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                                count += 1

                        box_counts[i, j] = count

            # Keep only non-zero counts and calculate probabilities
            occupied_boxes = box_counts[box_counts > 0].flatten()
            total_segments = occupied_boxes.sum()

            if total_segments > 0:
                probabilities = occupied_boxes / total_segments
            else:
                probabilities = np.array([])

            all_box_counts.append(occupied_boxes)
            all_probabilities.append(probabilities)

            # Report statistics
            box_count = len(occupied_boxes)
            print(f"  Box size: {box_size:.6f}, Occupied boxes: {box_count}, Time: {time.time() - box_start_time:.2f}s")
        
        # Calculate multifractal properties
        print("Calculating multifractal spectrum...")
        
        taus = np.zeros(len(q_values))
        Dqs = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))
        
        for q_idx, q in enumerate(q_values):
            print(f"Processing q = {q:.1f}")
            
            # Skip q=1 as it requires special treatment
            if abs(q - 1.0) < 1e-6:
                continue
                
            # Calculate partition function for each box size
            Z_q = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    Z_q[i] = np.sum(probabilities ** q)
                else:
                    Z_q[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(Z_q)
            if np.sum(valid) < 3:
                print(f"Warning: Not enough valid points for q={q}")
                taus[q_idx] = np.nan
                Dqs[q_idx] = np.nan
                r_squared[q_idx] = np.nan
                continue
                
            log_eps = np.log(box_sizes[valid])
            log_Z_q = np.log(Z_q[valid])
            
            # Linear regression to find tau(q)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_Z_q)
            
            # Calculate tau(q) and D(q)
            taus[q_idx] = slope
            Dqs[q_idx] = taus[q_idx] / (q - 1) if q != 1 else np.nan
            r_squared[q_idx] = r_value ** 2
            
            print(f"  τ({q}) = {taus[q_idx]:.4f}, D({q}) = {Dqs[q_idx]:.4f}, R² = {r_squared[q_idx]:.4f}")
        
        # Handle q=1 case (information dimension) separately
        q1_idx = np.where(np.abs(q_values - 1.0) < 1e-6)[0]
        if len(q1_idx) > 0:
            q1_idx = q1_idx[0]
            print(f"Processing q = 1.0 (information dimension)")
            
            # Calculate using L'Hôpital's rule
            mu_log_mu = np.zeros(num_box_sizes)
            
            for i, probabilities in enumerate(all_probabilities):
                if len(probabilities) > 0:
                    # Use -sum(p*log(p)) for information dimension
                    mu_log_mu[i] = -np.sum(probabilities * np.log(probabilities))
                else:
                    mu_log_mu[i] = np.nan
            
            # Remove NaN values
            valid = ~np.isnan(mu_log_mu)
            if np.sum(valid) >= 3:
                log_eps = np.log(box_sizes[valid])
                log_mu = mu_log_mu[valid]
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_mu)
                
                # Store information dimension
                # τ(1) = 0 exactly (since Z(1,ε) = Σp_i = 1 for all ε)
                # D₁ = lim_{q→1} τ(q)/(q-1) = dτ/dq|_{q=1}
                # From L'Hôpital: D₁ = -slope of entropy vs log(ε)
                taus[q1_idx] = 0.0
                Dqs[q1_idx] = -slope   # Information dimension D₁
                r_squared[q1_idx] = r_value ** 2

                print(f"  τ(1) = 0.0000 (exact), D(1) = {Dqs[q1_idx]:.4f}, R² = {r_squared[q1_idx]:.4f}")
        
        # Calculate alpha and f(alpha) via Legendre transform of τ(q)
        # α = dτ/dq, f(α) = qα - τ(q)
        alpha = np.full(len(q_values), np.nan)
        f_alpha = np.full(len(q_values), np.nan)

        print("Calculating multifractal spectrum f(α)...")

        # Use only valid (non-NaN) τ values for spline fitting
        valid_mask = ~np.isnan(taus)
        if np.sum(valid_mask) >= 4:
            q_valid = q_values[valid_mask]
            tau_valid = taus[valid_mask]

            # Fit τ(q) with a cubic spline for smooth differentiation
            from scipy.interpolate import UnivariateSpline
            try:
                # Use smoothing spline; s=0 interpolates exactly
                spline = UnivariateSpline(q_valid, tau_valid, k=3, s=0)
                tau_smooth = spline(q_valid)
                dtau_dq = spline.derivative()(q_valid)

                # Compute α and f(α) from the smooth derivative
                j = 0
                for i in range(len(q_values)):
                    if valid_mask[i]:
                        alpha[i] = dtau_dq[j]
                        f_alpha[i] = q_values[i] * alpha[i] - taus[i]
                        j += 1

            except Exception as e:
                print(f"  Spline fitting failed ({e}), falling back to finite differences")
                for i in range(len(q_values)):
                    if not valid_mask[i]:
                        continue
                    if i > 0 and i < len(q_values) - 1 and valid_mask[i-1] and valid_mask[i+1]:
                        alpha[i] = (taus[i+1] - taus[i-1]) / (q_values[i+1] - q_values[i-1])
                    elif i == 0 and valid_mask[i+1]:
                        alpha[i] = (taus[i+1] - taus[i]) / (q_values[i+1] - q_values[i])
                    elif i == len(q_values)-1 and valid_mask[i-1]:
                        alpha[i] = (taus[i] - taus[i-1]) / (q_values[i] - q_values[i-1])
                    f_alpha[i] = q_values[i] * alpha[i] - taus[i]

            for i, q in enumerate(q_values):
                if not np.isnan(alpha[i]):
                    print(f"  q = {q:.1f}, α = {alpha[i]:.4f}, f(α) = {f_alpha[i]:.4f}")
        else:
            print("  Not enough valid τ(q) points for Legendre transform")
        
        # Calculate multifractal parameters
        valid_idx = ~np.isnan(Dqs)
        if np.sum(valid_idx) >= 3:
            D0 = Dqs[np.searchsorted(q_values, 0)] if 0 in q_values else np.nan
            D1 = Dqs[np.searchsorted(q_values, 1)] if 1 in q_values else np.nan
            D2 = Dqs[np.searchsorted(q_values, 2)] if 2 in q_values else np.nan
            
            # Width of multifractal spectrum
            valid = ~np.isnan(alpha)
            if np.sum(valid) >= 2:
                alpha_width = np.max(alpha[valid]) - np.min(alpha[valid])
            else:
                alpha_width = np.nan
            
            # Degree of multifractality: D(-∞) - D(+∞) ≈ D(-5) - D(5)
            if -5 in q_values and 5 in q_values:
                degree_multifractality = Dqs[np.searchsorted(q_values, -5)] - Dqs[np.searchsorted(q_values, 5)]
            else:
                degree_multifractality = np.nan
            
            print(f"Multifractal parameters:")
            print(f"  D(0) = {D0:.4f} (capacity dimension)")
            print(f"  D(1) = {D1:.4f} (information dimension)")
            print(f"  D(2) = {D2:.4f} (correlation dimension)")
            print(f"  α width = {alpha_width:.4f}")
            print(f"  Degree of multifractality = {degree_multifractality:.4f}")
        else:
            D0 = D1 = D2 = alpha_width = degree_multifractality = np.nan
            print("Warning: Not enough valid points to calculate multifractal parameters")
        
        # Plot results if output directory provided
        if output_dir:
            # Plot D(q) vs q
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(Dqs)
            plt.plot(q_values[valid], Dqs[valid], 'bo-', markersize=4)
            
            if 0 in q_values:
                plt.axhline(y=Dqs[np.searchsorted(q_values, 0)], color='r', linestyle='--', 
                           label=f"D(0) = {Dqs[np.searchsorted(q_values, 0)]:.4f}")
            
            plt.xlabel('q')
            plt.ylabel('D(q)')
            plt.title(f'Generalized Dimensions D(q) at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "multifractal_dimensions.png"), dpi=300)
            plt.close()
            
            # Plot f(alpha) vs alpha (multifractal spectrum)
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(alpha) & ~np.isnan(f_alpha)
            plt.plot(alpha[valid], f_alpha[valid], 'bo-', markersize=4)
            
            # Add selected q values as annotations
            q_to_highlight = [-5, -2, 0, 2, 5]
            for q_val in q_to_highlight:
                if q_val in q_values:
                    idx = np.searchsorted(q_values, q_val)
                    if idx < len(q_values) and valid[idx]:
                        plt.annotate(f"q={q_values[idx]}", 
                                    (alpha[idx], f_alpha[idx]),
                                    xytext=(5, 0), textcoords='offset points')
            
            plt.xlabel('α')
            plt.ylabel('f(α)')
            plt.title(f'Multifractal Spectrum f(α) at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_spectrum.png"), dpi=300)
            plt.close()
            
            # Plot R² values
            plt.figure(figsize=(10, 6))
            valid = ~np.isnan(r_squared)
            plt.plot(q_values[valid], r_squared[valid], 'go-', markersize=4)
            plt.xlabel('q')
            plt.ylabel('R²')
            plt.title(f'Fit Quality for Different q Values at t = {data["time"]:.2f}')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "multifractal_r_squared.png"), dpi=300)
            plt.close()
            
            # Save results to CSV
            import pandas as pd
            results_df = pd.DataFrame({
                'q': q_values,
                'tau': taus,
                'Dq': Dqs,
                'alpha': alpha,
                'f_alpha': f_alpha,
                'r_squared': r_squared
            })
            results_df.to_csv(os.path.join(output_dir, "multifractal_results.csv"), index=False)
            
            # Save multifractal parameters
            params_df = pd.DataFrame({
                'Parameter': ['Time', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality'],
                'Value': [data['time'], D0, D1, D2, alpha_width, degree_multifractality]
            })
            params_df.to_csv(os.path.join(output_dir, "multifractal_parameters.csv"), index=False)
        
        # Return results
        return {
            'q_values': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared,
            'D0': D0,
            'D1': D1,
            'D2': D2,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_multifractality,
            'time': data['time']
        }
    
    def analyze_multifractal_evolution(self, vtk_files, output_dir=None, q_values=None):
        """
        Analyze how multifractal properties evolve over time or across resolutions.
        
        Args:
            vtk_files: Dict mapping either times or resolutions to VTK files
                      e.g. {0.1: 'RT_t0.1.vtk', 0.2: 'RT_t0.2.vtk'} for time series
                      or {100: 'RT100x100.vtk', 200: 'RT200x200.vtk'} for resolutions
            output_dir: Directory to save results
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            
        Returns:
            dict: Multifractal evolution results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Determine type of analysis (time or resolution)
        keys = list(vtk_files.keys())
        is_time_series = all(isinstance(k, float) for k in keys)
        
        if is_time_series:
            print(f"Analyzing multifractal evolution over time series: {sorted(keys)}")
            x_label = 'Time'
            series_name = "time"
        else:
            print(f"Analyzing multifractal evolution across resolutions: {sorted(keys)}")
            x_label = 'Resolution'
            series_name = "resolution"
        
        # Initialize results storage
        results = []
        
        # Process each file
        for key, vtk_file in sorted(vtk_files.items()):
            print(f"\nProcessing {series_name} = {key}, file: {vtk_file}")
            
            try:
                # Read VTK file
                data = self.read_vtk_file(vtk_file)
                
                # Create subdirectory for this point
                if output_dir:
                    point_dir = os.path.join(output_dir, f"{series_name}_{key}")
                    os.makedirs(point_dir, exist_ok=True)
                else:
                    point_dir = None
                
                # Perform multifractal analysis
                mf_results = self.compute_multifractal_spectrum(data, q_values=q_values, output_dir=point_dir)
                
                if mf_results:
                    # Store results with the key (time or resolution)
                    mf_results[series_name] = key
                    results.append(mf_results)
                
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary plots
        if results and output_dir:
            # Extract evolution of key parameters
            x_values = [res[series_name] for res in results]
            D0_values = [res['D0'] for res in results]
            D1_values = [res['D1'] for res in results]
            D2_values = [res['D2'] for res in results]
            alpha_width = [res['alpha_width'] for res in results]
            degree_mf = [res['degree_multifractality'] for res in results]
            
            # Plot generalized dimensions evolution
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, D0_values, 'bo-', label='D(0) - Capacity dimension')
            plt.plot(x_values, D1_values, 'ro-', label='D(1) - Information dimension')
            plt.plot(x_values, D2_values, 'go-', label='D(2) - Correlation dimension')
            plt.xlabel(x_label)
            plt.ylabel('Generalized Dimensions')
            plt.title(f'Evolution of Generalized Dimensions with {x_label}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "dimensions_evolution.png"), dpi=300)
            plt.close()
            
            # Plot multifractal parameters evolution
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, alpha_width, 'ms-', label='α width')
            plt.plot(x_values, degree_mf, 'cd-', label='Degree of multifractality')
            plt.xlabel(x_label)
            plt.ylabel('Parameter Value')
            plt.title(f'Evolution of Multifractal Parameters with {x_label}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, "multifractal_params_evolution.png"), dpi=300)
            plt.close()
            
            # Create 3D surface plot of D(q) evolution if matplotlib supports it
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                # Prepare data for 3D plot
                X, Y = np.meshgrid(x_values, q_values)
                Z = np.zeros((len(q_values), len(x_values)))
                
                for i, result in enumerate(results):
                    for j, q in enumerate(q_values):
                        q_idx = np.where(result['q_values'] == q)[0]
                        if len(q_idx) > 0:
                            Z[j, i] = result['Dq'][q_idx[0]]
                
                # Create 3D plot
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('q')
                ax.set_zlabel('D(q)')
                ax.set_title(f'Evolution of D(q) Spectrum with {x_label}')
                
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='D(q)')
                plt.savefig(os.path.join(output_dir, "Dq_evolution_3D.png"), dpi=300)
                plt.close()
                
            except Exception as e:
                print(f"Error creating 3D plot: {str(e)}")
            
            # Save summary CSV
            import pandas as pd
            summary_df = pd.DataFrame({
                series_name: x_values,
                'D0': D0_values,
                'D1': D1_values,
                'D2': D2_values,
                'alpha_width': alpha_width,
                'degree_multifractality': degree_mf
            })
            summary_df.to_csv(os.path.join(output_dir, "multifractal_evolution_summary.csv"), index=False)
        
        return results
