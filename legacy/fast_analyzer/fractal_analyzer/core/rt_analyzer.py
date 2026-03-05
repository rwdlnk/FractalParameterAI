# rt_analyzer.py - ENHANCED VERSION with Clean Mixing Analysis
# Part 1: Imports, VTK File Handler, and Core Classes

import numpy as np
import pandas as pd
from scipy import stats, ndimage, integrate

# Handle numpy trapz -> trapezoid transition
try:
    from numpy import trapezoid
except ImportError:
    # Fallback for older numpy versions
    trapezoid = np.trapz
import matplotlib.pyplot as plt
import os
import re
import time
import glob
import argparse
from typing import Tuple, List, Dict, Optional
from skimage import measure
try:
    from .conrec_extractor import CONRECExtractor, compare_extraction_methods
    from .plic_extractor import PLICExtractor, AdvancedPLICExtractor, compare_plic_vs_conrec
except ImportError:
    # Fallback for direct script execution
    try:
        from conrec_extractor import CONRECExtractor, compare_extraction_methods
        from plic_extractor import PLICExtractor, AdvancedPLICExtractor, compare_plic_vs_conrec
    except ImportError:
        print("Warning: CONREC/PLIC extractors not available")
        CONRECExtractor = None
        PLICExtractor = None
        AdvancedPLICExtractor = None

from scipy import fft

# =================================================================
# VTK FILE HANDLER - Enhanced for RTnnnxmmmm-tttt.vtk format
# =================================================================

class VTKFileHandler:
    """
    Handles VTK file parsing, sorting, and metadata extraction for RT simulation files.
    Supports format: RTnnnxmmmm-tttt.vtk where:
    - nnn, mmmm = grid resolution (x, y)
    - tttt = simulation time * 1000 (variable digits)
    """
    
    @staticmethod
    def parse_vtk_filename(vtk_file: str) -> Dict:
        """
        Parse VTK filename to extract grid resolution and simulation time.
        
        Args:
            vtk_file: Path to VTK file (RTnnnxmmmm-tttt.vtk format)
            
        Returns:
            dict with nx, ny, time_step, sim_time, and validation info
        """
        basename = os.path.basename(vtk_file)
        
        # Enhanced pattern for RTnnnxmmmm-tttt.vtk
        # Handles variable-length time digits (1-5 digits as mentioned)
        pattern = r'RT(\d+)x(\d+)-(\d+)\.vtk$'
        match = re.match(pattern, basename)
        
        if not match:
            # Fallback patterns for edge cases
            fallback_patterns = [
                r'RT(\d+)x(\d+)_(\d+)\.vtk$',   # Underscore instead of dash
                r'RT(\d+)x(\d+)\.(\d+)\.vtk$',  # Dot separator
            ]
            
            for fallback_pattern in fallback_patterns:
                match = re.match(fallback_pattern, basename)
                if match:
                    break
        
        if match:
            nx = int(match.group(1))
            ny = int(match.group(2)) 
            time_step = int(match.group(3))
            sim_time = time_step / 1000.0  # Convert to actual simulation time
            
            return {
                'nx': nx,
                'ny': ny,
                'time_step': time_step,
                'sim_time': sim_time,
                'filename': basename,
                'full_path': vtk_file,
                'is_square': (nx == ny),
                'grid_type': f"{nx}x{ny}",
                'valid': True
            }
        else:
            # Return invalid entry with some extracted info for debugging
            return {
                'nx': None,
                'ny': None,
                'time_step': None,
                'sim_time': None,
                'filename': basename,
                'full_path': vtk_file,
                'is_square': None,
                'grid_type': 'unknown',
                'valid': False,
                'error': f"Could not parse filename: {basename}"
            }
    
    @staticmethod
    def find_and_sort_vtk_files(pattern_or_dir: str, 
                               grid_filter: Optional[str] = None,
                               time_range: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        Find VTK files and sort them by simulation time for time series analysis.
        
        Args:
            pattern_or_dir: Glob pattern (e.g., "RT*.vtk") or directory path
            grid_filter: Filter by grid type (e.g., "160x200", "400x400")
            time_range: Tuple of (min_time, max_time) to filter by simulation time
            
        Returns:
            List of file info dictionaries, sorted by simulation time
        """
        # Handle both glob patterns and directory paths
        if os.path.isdir(pattern_or_dir):
            search_pattern = os.path.join(pattern_or_dir, "RT*.vtk")
        else:
            search_pattern = pattern_or_dir
        
        # Find all matching files
        vtk_files = glob.glob(search_pattern)
        
        if not vtk_files:
            print(f"⚠️  No VTK files found matching: {search_pattern}")
            return []
        
        print(f"📂 Found {len(vtk_files)} VTK files")
        
        # Parse all filenames
        file_info_list = []
        invalid_files = []
        
        for vtk_file in vtk_files:
            file_info = VTKFileHandler.parse_vtk_filename(vtk_file)
            
            if file_info['valid']:
                file_info_list.append(file_info)
            else:
                invalid_files.append(file_info)
        
        # Report parsing results
        print(f"✅ Successfully parsed: {len(file_info_list)} files")
        if invalid_files:
            print(f"❌ Failed to parse: {len(invalid_files)} files")
            for invalid in invalid_files[:3]:  # Show first 3 invalid files
                print(f"   - {invalid['filename']}: {invalid.get('error', 'Unknown error')}")
            if len(invalid_files) > 3:
                print(f"   ... and {len(invalid_files) - 3} more")
        
        # Apply grid filter if specified
        if grid_filter:
            before_filter = len(file_info_list)
            file_info_list = [f for f in file_info_list if f['grid_type'] == grid_filter]
            print(f"🔍 Grid filter '{grid_filter}': {len(file_info_list)}/{before_filter} files")
        
        # Apply time range filter if specified
        if time_range:
            min_time, max_time = time_range
            before_filter = len(file_info_list)
            file_info_list = [f for f in file_info_list 
                            if min_time <= f['sim_time'] <= max_time]
            print(f"⏰ Time filter [{min_time}, {max_time}]: {len(file_info_list)}/{before_filter} files")
        
        if not file_info_list:
            print("❌ No files remaining after filtering")
            return []
        
        # Sort by simulation time (CRITICAL for time series analysis)
        file_info_list.sort(key=lambda x: x['sim_time'])
        
        # Print sorting summary
        time_range_actual = (file_info_list[0]['sim_time'], file_info_list[-1]['sim_time'])
        unique_grids = set(f['grid_type'] for f in file_info_list)
        
        print(f"🔢 Sorted {len(file_info_list)} files by simulation time:")
        print(f"   Time range: {time_range_actual[0]:.3f} → {time_range_actual[1]:.3f}")
        print(f"   Grid types: {', '.join(sorted(unique_grids))}")
        print(f"   First file: {file_info_list[0]['filename']}")
        print(f"   Last file:  {file_info_list[-1]['filename']}")
        
        return file_info_list
    
    @staticmethod
    def group_by_resolution(file_info_list: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group files by grid resolution for separate analysis.
        
        Args:
            file_info_list: List of file info dictionaries
            
        Returns:
            Dictionary mapping grid_type to list of file info
        """
        groups = {}
        
        for file_info in file_info_list:
            grid_type = file_info['grid_type']
            if grid_type not in groups:
                groups[grid_type] = []
            groups[grid_type].append(file_info)
        
        # Sort each group by time and report
        for grid_type, files in groups.items():
            files.sort(key=lambda x: x['sim_time'])
            time_range = (files[0]['sim_time'], files[-1]['sim_time'])
            print(f"📊 {grid_type}: {len(files)} files, time {time_range[0]:.3f} → {time_range[1]:.3f}")
        
        return groups

# =================================================================
# INTERFACE CACHE SYSTEM - Enhanced for large grids
# =================================================================

class InterfaceCache:
    """
    Enhanced cache for interface extraction results and VTK data.
    Critical for large grids (1280×1600) to avoid expensive re-reads.
    """
    
    def __init__(self, primary_segments=None, contours=None, spatial_index=None, 
                 bounding_box=None, grid_spacing=None, extraction_method='scikit-image',
                 vtk_data=None):
        """Initialize interface cache with extracted data."""
        self.primary_segments = primary_segments or []
        self.contours = contours or {}
        self.spatial_index = spatial_index
        self.bounding_box = bounding_box
        self.grid_spacing = grid_spacing
        self.extraction_method = extraction_method
        self.vtk_data = vtk_data  # NEW: Cache VTK data for mixing analysis
        
        # Metadata for diagnostics and optimization tracking
        self.extraction_time = time.time()
        self.segment_count = len(self.primary_segments)
        self.total_contour_points = self._count_contour_points()
        
    def _count_contour_points(self):
        """Count total points across all contour levels."""
        total = 0
        for level_contours in self.contours.values():
            if level_contours:
                for contour in level_contours:
                    if hasattr(contour, 'shape'):
                        total += contour.shape[0]  # numpy array
                    else:
                        total += len(contour)  # list
        return total
    
    def get_segments_for_fractal(self):
        """Get segments optimized for fractal analysis."""
        return self.primary_segments
    
    def get_vtk_data_for_mixing(self):
        """Get cached VTK data for mixing analysis."""
        return self.vtk_data
    
    def has_vtk_data(self):
        """Check if VTK data is cached."""
        return self.vtk_data is not None
    
    def has_segments(self):
        """Check if primary segments are available for fractal analysis."""
        return len(self.primary_segments) > 0
# rt_analyzer.py - ENHANCED VERSION
# Part 2: RTAnalyzer Class Initialization and VTK Reading

class RTAnalyzer:
    """Enhanced Rayleigh-Taylor simulation analyzer with clean mixing calculations."""

    def __init__(self, output_dir="./rt_analysis", use_grid_optimization=False, no_titles=False,
                 use_conrec=False, use_plic=False, debug=False):
        """Initialize the RT analyzer."""
        self.output_dir = output_dir
        self.use_grid_optimization = use_grid_optimization
        self.no_titles = no_titles
        self.use_conrec = use_conrec
        self.use_plic = use_plic
        self.debug = debug
        os.makedirs(output_dir, exist_ok=True)

        # Add rectangular grid support
        self.grid_shape = None  # Will store (ny, nx) or (n, n) for square
        self.is_rectangular = False

        # Initialize interface cache for optimization
        self.interface_cache = {}
        self.interface_value = 0.5  # Default interface level

        # Create fractal analyzer instance
        try:
            # Add framework root to Python path for imports
            import sys
            current_file = os.path.abspath(__file__)
            framework_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            if framework_root not in sys.path:
                sys.path.insert(0, framework_root)

            from fractal_analyzer.core.wu_enhanced_analyzer import WuEnhancedAnalyzer
            self.fractal_analyzer = WuEnhancedAnalyzer.create_rt_analyzer()
            print(f"Fractal analyzer initialized (grid optimization: {'ENABLED' if use_grid_optimization else 'DISABLED'})")
        except ImportError as e:
            print(f"Warning: fractal_analyzer module not found: {str(e)}")
            print("Make sure fractal_analyzer.py is in the same directory")
            self.fractal_analyzer = None
        
        # Initialize PLIC extractor if requested
        if self.use_plic and AdvancedPLICExtractor is not None:
            try:
                self.plic_extractor = AdvancedPLICExtractor(debug=debug)
                print(f"PLIC extractor initialized - theoretical VOF interface reconstruction enabled")
            except Exception as e:
                print(f"ERROR: Could not initialize PLIC extractor: {e}")
                self.plic_extractor = None
                self.use_plic = False
        else:
            self.plic_extractor = None

        # Initialize CONREC extractor if requested (and PLIC not enabled)
        if self.use_conrec and not self.use_plic and CONRECExtractor is not None:
            try:
                self.conrec_extractor = CONRECExtractor(debug=debug)
                print(f"CONREC extractor initialized - precision interface extraction enabled")
            except Exception as e:
                print(f"ERROR: Could not initialize CONREC extractor: {e}")
                self.conrec_extractor = None
                self.use_conrec = False
        else:
            self.conrec_extractor = None

        # Validation: Don't allow both CONREC and PLIC simultaneously
        if self.use_conrec and self.use_plic:
            print("WARNING: Both CONREC and PLIC enabled. PLIC will take precedence.")
            self.use_conrec = False
            print("CONREC disabled. Using PLIC for interface extraction.")
    
    def enable_multifractal_analysis(self):
        """Lazy loading of multifractal capabilities."""
        if not hasattr(self, 'multifractal_analyzer'):
            try:
                from multifractal_analyzer import MultifractalAnalyzer
                self.multifractal_analyzer = MultifractalAnalyzer(debug=self.debug)
                print("Multifractal analyzer enabled")
            except ImportError:
                print("Warning: MultifractalAnalyzer not available")
                self.multifractal_analyzer = None

    def auto_detect_resolution_from_vtk_filename(self, vtk_file):
        """Enhanced resolution detection using the new filename parser."""
        file_info = VTKFileHandler.parse_vtk_filename(vtk_file)
        
        if file_info['valid']:
            nx, ny = file_info['nx'], file_info['ny']
            print(f"  Auto-detected resolution: {nx}×{ny} from filename")
            
            # Update internal state
            self.grid_shape = (ny, nx)  # Store as (ny, nx) for array indexing
            self.is_rectangular = (nx != ny)
            
            return nx, ny
        else:
            print(f"  ⚠️  {file_info.get('error', 'Unknown filename format')}")
            return None, None

    def extract_simulation_time_from_filename(self, vtk_file):
        """Extract simulation time directly from filename (more reliable than VTK parsing)."""
        file_info = VTKFileHandler.parse_vtk_filename(vtk_file)
        
        if file_info['valid']:
            return file_info['sim_time']
        else:
            # Fallback to trying to extract from VTK file content
            print(f"  ⚠️  Could not extract time from filename, will read from VTK")
            return None

    def read_vtk_file(self, vtk_file):
        """Enhanced VTK reader to extract velocity components AND VOF data."""
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
        
        # Check if this is cell-centered data
        is_cell_data = any("CELL_DATA" in line for line in lines)
        
        # Extract ALL scalar fields (F, u, v, and any others)
        scalar_data = {}
        
        for i, line in enumerate(lines):
            if line.strip().startswith("SCALARS"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    field_name = parts[1]  # F, u, v, etc.
                    
                    # Read the data values
                    data_values = []
                    j = i + 2  # Skip the LOOKUP_TABLE line
                    while j < len(lines) and not lines[j].strip().startswith("SCALARS"):
                        if lines[j].strip():  # Skip empty lines
                            data_values.extend(list(map(float, lines[j].strip().split())))
                        j += 1
                    
                    if data_values:
                        scalar_data[field_name] = np.array(data_values)

        # Handle coordinate system (cell vs node centered)
        if is_cell_data:
            nx_cells, ny_cells = nx-1, ny-1
            x_cell = 0.5 * (x_coords[:-1] + x_coords[1:])
            y_cell = 0.5 * (y_coords[:-1] + y_coords[1:])
            x_grid, y_grid = np.meshgrid(x_cell, y_cell)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx_cells, ny_cells)
        else:
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            x_grid, y_grid = x_grid.T, y_grid.T
            grid_shape = (nx, ny)
        
        # Reshape all scalar fields to 2D grids
        reshaped_data = {}
        for field_name, data in scalar_data.items():
            if len(data) == grid_shape[0] * grid_shape[1]:
                reshaped_data[field_name] = data.reshape(grid_shape[1], grid_shape[0]).T
            else:
                print(f"   WARNING: {field_name} size mismatch: {len(data)} vs expected {grid_shape[0] * grid_shape[1]}")
        
        # Extract simulation time from filename first, fallback to pattern matching
        sim_time = self.extract_simulation_time_from_filename(vtk_file)
        if sim_time is None:
            time_match = re.search(r'(\d+)\.vtk$', os.path.basename(vtk_file))
            sim_time = float(time_match.group(1))/1000.0 if time_match else 0.0
        
        # Create comprehensive output dictionary
        result = {
            'x': x_grid,
            'y': y_grid,
            'dims': (nx, ny, nz),
            'time': sim_time,
            'is_cell_data': is_cell_data
        }
        
        # Add all available fields
        result.update(reshaped_data)
        
        # Ensure we have at least F field for backward compatibility
        if 'F' not in result and 'f' not in result:
            print("   WARNING: No VOF field (F) found in VTK file")
        else:
            # Standardize field name for backward compatibility
            if 'F' in result and 'f' not in result:
                result['f'] = result['F']
        
        return result

    # ===============================================================
    # MIXING METHODS - Replace existing methods in RTAnalyzer class
    # ===============================================================

    def _find_initial_interface_position(self, f_grid, y_grid):
        """Find initial interface position where horizontally-averaged f ≈ 0.5"""
        f_avg_y = np.mean(f_grid, axis=0)  # Average over x
        y_coords = y_grid[0, :]  # y-coordinates
        idx_closest = np.argmin(np.abs(f_avg_y - 0.5))
        y0 = y_coords[idx_closest]
        return y0

    def _find_concentration_crossing(self, f_profile, y_profile, target_concentration):
        """Find y-position where concentration profile crosses target value using interpolation."""
        diff = f_profile - target_concentration
        sign_changes = np.diff(np.sign(diff))
        crossings = []
        
        for i in np.where(sign_changes != 0)[0]:
            y1, y2 = y_profile[i], y_profile[i+1]
            f1, f2 = f_profile[i], f_profile[i+1]
            
            if abs(f2 - f1) > 1e-10:  # Avoid division by zero
                y_cross = y1 + (target_concentration - f1) * (y2 - y1) / (f2 - f1)
                crossings.append(y_cross)
        
        return crossings

    def compute_mixing_heights_mass_conservation(self, vtk_data, y0=None):
        """
        Method 1: Mass-conservation based equivalent thicknesses (YOUR THEORETICAL FRAMEWORK)
        
        Computes h₀₁ and h₁₀ from your introduction theory:
        - h₀₁: equivalent thickness of bottom fluid (ρᵦ) above initial interface
        - h₁₀: equivalent thickness of top fluid (ρₜ) below initial interface
        - Mass conservation: ρᵦ × h₀₁ × L = ρₜ × h₁₀ × L
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x'] 
        y_grid = vtk_data['y']
        
        if y0 is None:
            y0 = self._find_initial_interface_position(f_grid, y_grid)
        
        # Get grid spacing for integration
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        L = x_grid[-1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 1.0
        
        # h₀₁: equivalent thickness of bottom fluid (ρᵦ) above y₀
        above_y0_mask = y_grid >= y0
        if np.any(above_y0_mask):
            h_01 = np.sum((1 - f_grid) * above_y0_mask) * dx * dy / L
        else:
            h_01 = 0.0
        
        # h₁₀: equivalent thickness of top fluid (ρₜ) below y₀  
        below_y0_mask = y_grid <= y0
        if np.any(below_y0_mask):
            h_10 = np.sum(f_grid * below_y0_mask) * dx * dy / L
        else:
            h_10 = 0.0
        
        h_total_mass = h_01 + h_10
        
        print(f"    Mass-conservation method:")
        print(f"      h₀₁ (bottom fluid above y₀): {h_01:.6f}")
        print(f"      h₁₀ (top fluid below y₀): {h_10:.6f}")
        print(f"      h_total (mass-based): {h_total_mass:.6f}")
        
        return {
            'h_01': h_01,                    # Bottom fluid above (your notation)
            'h_10': h_10,                    # Top fluid below (your notation)  
            'h_total': h_total_mass,         # For backward compatibility
            'ht': h_01,                      # Legacy compatibility
            'hb': h_10,                      # Legacy compatibility
            'y0': y0,
            'method': 'mass_conservation'
        }

    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }

    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }
    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }
    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }
    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }
    def compute_mixing_diagnostic(self, vtk_data):
        """
        Method 3: Global mixedness diagnostic (YOUR MIXING QUALITY MEASURE)
        
        Computes the mixing diagnostic ∫∫ f(1-f) dA over entire domain.
        When normalized by domain area, gives average degree of mixing.
        """
        f_grid = vtk_data['f']
        x_grid = vtk_data['x']
        y_grid = vtk_data['y'] 
        
        # Get grid spacing
        dy = y_grid[0, 1] - y_grid[0, 0] if y_grid.shape[1] > 1 else 0.001
        dx = x_grid[1, 0] - x_grid[0, 0] if x_grid.shape[0] > 1 else 0.001
        
        # Calculate domain area H×L
        domain_area = (np.max(x_grid) - np.min(x_grid)) * (np.max(y_grid) - np.min(y_grid))

        # Compute f(1-f) pointwise - mixing density field
        mixing_density = f_grid * (1 - f_grid)
        
        # Integrate over entire domain: ∫∫ f(1-f) dA
        total_mixing_integral = np.sum(mixing_density) * dx * dy
        
        # Normalized mixing ratio: (∫∫ f(1-f) dA) / (H×L)
        mixing_ratio = total_mixing_integral / domain_area
        
        # Mixing efficiency: fraction of theoretical maximum (0.25)
        mixing_efficiency = mixing_ratio / 0.25

        print(f"    Global mixedness diagnostic:")
        print(f"      Total mixing ∫∫f(1-f)dA: {total_mixing_integral:.6f}")
        print(f"      Mixing ratio: {mixing_ratio:.6f}")
        print(f"      Mixing efficiency: {mixing_efficiency:.1%}")
        
        return {
            'total_mixing_integral': total_mixing_integral,
            'domain_area': domain_area,
            'mixing_ratio': mixing_ratio,                   # Your key metric!
            'mixing_efficiency': mixing_efficiency,         # Fraction of max possible
            'mixing_density_field': mixing_density,
            'method': 'global_mixedness'
        }

    def compute_youngs_integral_width(self, vtk_data, y0=None):
        """
        Method 4: Youngs integral width W = ∫ f̄(1-f̄) dy (NEW ADDITION)
        
        Computes Youngs "integral width" - more robust than geometric measures 
        for turbulent flows as it's less sensitive to statistical fluctuations.
        """
        f_grid = vtk_data['f']
        y_grid = vtk_data['y']
        
        if y0 is None:
            y0 = self._find_initial_interface_position(f_grid, y_grid)
        
        # Horizontal average profile f̄(y)
        f_avg = np.mean(f_grid, axis=0)  # Average over x
        y_coords = y_grid[0, :]         # y-coordinates
        
        # Compute integrand: f̄(1-f̄) 
        integrand = f_avg * (1 - f_avg)
        
        # Integrate using trapezoidal rule: W = ∫ f̄(1-f̄) dy
        W = trapezoid(integrand, y_coords)
        
        # Additional diagnostics
        max_integrand = np.max(integrand)
        if W > 1e-10:
            mixing_centroid = trapezoid(y_coords * integrand, y_coords) / W
        else:
            mixing_centroid = y0
        
        print(f"    Youngs integral width method:")
        print(f"      W = ∫f̄(1-f̄)dy: {W:.6f}")
        print(f"      Max f̄(1-f̄): {max_integrand:.6f}")
        print(f"      Mixing centroid: {mixing_centroid:.6f}")
        
        return {
            'W': W,                                    # Youngs integral width
            'integral_width': W,                       # Alias for clarity
            'max_mixing_density_1d': max_integrand,    # Peak mixing density
            'mixing_centroid': mixing_centroid,        # Weighted center
            'y0': y0,
            'integrand': integrand,                    # f̄(1-f̄) profile
            'f_avg_profile': f_avg,                    # f̄(y) profile
            'y_coords': y_coords,                      # y-coordinate array
            'method': 'youngs_integral'
        }


    def find_concentration_crossing(self, f_profile, y_profile, target_concentration):
        """
        Find y-position where concentration profile crosses target value using interpolation.
        """
        # Find indices where profile crosses the target
        diff = f_profile - target_concentration
        sign_changes = np.diff(np.sign(diff))

        crossings = []
        crossing_indices = np.where(sign_changes != 0)[0]

        for i in crossing_indices:
            # Linear interpolation between points i and i+1
            y1, y2 = y_profile[i], y_profile[i+1]
            f1, f2 = f_profile[i], f_profile[i+1]
            
            # Interpolate to find exact crossing point
            if f2 != f1:  # Avoid division by zero
                y_cross = y1 + (target_concentration - f1) * (y2 - y1) / (f2 - f1)
                crossings.append(y_cross)

        return crossings

    def compute_mixing_heights_geometric(self, vtk_data, y0=None, 
                                       lower_threshold=0.05, upper_threshold=0.95):
        """
        Method 2: Geometric penetration distances (DALZIEL THRESHOLD METHOD)
        
        Finds geometric distances where horizontally-averaged f crosses thresholds:
        - h_penetration_up: distance upward from y₀ where f̄ = upper_threshold
        - h_penetration_down: distance downward from y₀ where f̄ = lower_threshold  
        """
        f_grid = vtk_data['f']
        y_grid = vtk_data['y']
        
        if y0 is None:
            y0 = self._find_initial_interface_position(f_grid, y_grid)

        # Horizontal average profile f̄(y)
        f_avg = np.mean(f_grid, axis=0)  # Average over x
        y_values = y_grid[0, :]  # y-coordinates

        # Find threshold crossings
        lower_crossings = self._find_concentration_crossing(f_avg, y_values, lower_threshold)
        upper_crossings = self._find_concentration_crossing(f_avg, y_values, upper_threshold)
        
        # Downward penetration (to lower threshold, below y0)
        h_penetration_down = 0.0
        y_lower_boundary = None
        if lower_crossings:
            valid_lower = [c for c in lower_crossings if c <= y0]
            if valid_lower:
                y_lower_boundary = min(valid_lower)  # Deepest penetration
                h_penetration_down = max(0, y0 - y_lower_boundary)
        
        # Upward penetration (to upper threshold, above y0)
        h_penetration_up = 0.0  
        y_upper_boundary = None
        if upper_crossings:
            valid_upper = [c for c in upper_crossings if c >= y0]
            if valid_upper:
                y_upper_boundary = max(valid_upper)  # Highest penetration
                h_penetration_up = max(0, y_upper_boundary - y0)
        
        # Boundary condition fix: handle cases where mixing reaches domain limits
        if h_penetration_down == 0.0:
            # Check if mixing has reached bottom boundary
            y_min = np.min(y_values)
            f_at_bottom = np.mean(f_grid[:, 0])  # Average f at bottom row
            if f_at_bottom > lower_threshold:  # Mixed beyond threshold at boundary
                h_penetration_down = y0 - y_min  # Full penetration to bottom
        
        if h_penetration_up == 0.0:
            # Check if mixing has reached top boundary  
            y_max = np.max(y_values)
            f_at_top = np.mean(f_grid[:, -1])  # Average f at top row
            if f_at_top < upper_threshold:  # Mixed beyond threshold at boundary
                h_penetration_up = y_max - y0  # Full penetration to top 

        h_total_geometric = h_penetration_up + h_penetration_down
        
        print(f"    Geometric penetration method:")
        print(f"      h_penetration_up (f̄={upper_threshold}): {h_penetration_up:.6f}")
        print(f"      h_penetration_down (f̄={lower_threshold}): {h_penetration_down:.6f}")
        print(f"      h_total (geometric): {h_total_geometric:.6f}")
        
        return {
            'h_penetration_up': h_penetration_up,           # Upward penetration distance
            'h_penetration_down': h_penetration_down,       # Downward penetration distance
            'h_total': h_total_geometric,                   # Total geometric extent
            'ht': h_penetration_up,                         # Legacy compatibility  
            'hb': h_penetration_down,                       # Legacy compatibility
            'y0': y0,
            'y_upper_boundary': y_upper_boundary,
            'y_lower_boundary': y_lower_boundary,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold,
            'method': 'geometric_penetration'
        }

    def compute_mixing_analysis(self, vtk_data, y0=None, method='comprehensive', 
                              lower_threshold=0.05, upper_threshold=0.95):
        """
        UPDATED: Main mixing analysis interface - replaces old methods
        
        Args:
            vtk_data: VTK data dict with 'f', 'x', 'y' fields
            y0: Initial interface position (auto-detected if None)
            method: 'mass_conservation', 'geometric', 'diagnostic', 'youngs', or 'comprehensive'
            lower_threshold: Lower threshold for geometric method
            upper_threshold: Upper threshold for geometric method
            
        Returns:
            dict with requested analysis results
        """
        if y0 is None:
            y0 = self._find_initial_interface_position(vtk_data['f'], vtk_data['y'])
        
        results = {'time': vtk_data.get('time', 0.0), 'y0': y0}
        
        if method == 'mass_conservation':
            return self.compute_mixing_heights_mass_conservation(vtk_data, y0)
        elif method == 'geometric':
            return self.compute_mixing_heights_geometric(vtk_data, y0, lower_threshold, upper_threshold)
        elif method == 'diagnostic':
            return self.compute_mixing_diagnostic(vtk_data)
        elif method == 'youngs':
            return self.compute_youngs_integral_width(vtk_data, y0)
        elif method == 'comprehensive':
            print(f"    Computing comprehensive mixing analysis...")
            
            # All four methods
            mass_results = self.compute_mixing_heights_mass_conservation(vtk_data, y0)
            geometric_results = self.compute_mixing_heights_geometric(vtk_data, y0, lower_threshold, upper_threshold)
            diagnostic_results = self.compute_mixing_diagnostic(vtk_data)
            youngs_results = self.compute_youngs_integral_width(vtk_data, y0)
            
            # Combine results
            results.update({
                # Mass conservation
                'h_01': mass_results['h_01'],
                'h_10': mass_results['h_10'],
                'h_total_mass': mass_results['h_total'],
                
                # Geometric
                'h_penetration_up': geometric_results['h_penetration_up'],
                'h_penetration_down': geometric_results['h_penetration_down'],
                'h_total_geometric': geometric_results['h_total'],
                
                # Diagnostic
                'total_mixing_integral': diagnostic_results['total_mixing_integral'],
                'mixing_ratio': diagnostic_results['mixing_ratio'],
                'mixing_efficiency': diagnostic_results['mixing_efficiency'],
                
                # Youngs
                'W': youngs_results['W'],
                'integral_width': youngs_results['W'],
                'mixing_centroid': youngs_results['mixing_centroid'],
                
                # For backward compatibility
                'ht': mass_results['h_01'],  # Use mass-conservation as default
                'hb': mass_results['h_10'],
                'h_total': mass_results['h_total'],
                'method': 'comprehensive'
            })
            
            # Cross-method comparison
            print(f"    Cross-method comparison:")
            h_mass = results['h_total_mass']
            h_geom = results['h_total_geometric'] 
            W = results['W']
            mixing_ratio = results['mixing_ratio']
            
            print(f"      Mass-based total: {h_mass:.6f}")
            print(f"      Geometric total:  {h_geom:.6f}")
            print(f"      Youngs width W:   {W:.6f}")
            print(f"      Mixing ratio:     {mixing_ratio:.6f}")
            
            if W > 1e-8:
                print(f"      Ratios: h_mass/W={h_mass/W:.2f}, h_geom/W={h_geom/W:.2f}")
            
            return results
        else:
            # Default to comprehensive for backward compatibility
            return self.compute_mixing_analysis(vtk_data, y0, 'comprehensive', lower_threshold, upper_threshold)

    # =================================================================
    # INTERFACE EXTRACTION METHODS - Enhanced with Caching
    # =================================================================

    def extract_interface(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """
        Enhanced interface extraction with PLIC, CONREC, or scikit-image methods.
        """
        if self.use_plic and self.plic_extractor is not None:
            print(f"   Interface extraction method: PLIC")
            return self._extract_interface_plic(f_grid, x_grid, y_grid, level, extract_all_levels)
        elif self.use_conrec and self.conrec_extractor is not None:
            print(f"   Interface extraction method: CONREC")
            return self._extract_interface_conrec(f_grid, x_grid, y_grid, level, extract_all_levels)
        else:
            print(f"   Interface extraction method: scikit-image")
            return self._extract_interface_skimage(f_grid, x_grid, y_grid, level, extract_all_levels)

    def _extract_interface_plic(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using PLIC algorithm."""
        print(f"     PLIC: Starting interface extraction...")
        if self.debug:
            print(f"     DEBUG: F-field stats - min={np.min(f_grid):.3f}, max={np.max(f_grid):.3f}")
            print(f"     DEBUG: Grid shape: {f_grid.shape}")

        try:
            if extract_all_levels:
                mixing_levels = {
                    'lower_boundary': 0.05,
                    'interface': 0.5,
                    'upper_boundary': 0.95
                }

                all_contours = {}
                total_segments = 0

                for level_name, level_value in mixing_levels.items():
                    try:
                        print(f"     PLIC: Extracting {level_name} (F={level_value:.2f})")

                        if level_value == 0.5:
                            f_for_plic = f_grid
                        else:
                            f_centered = f_grid - 0.5 + level_value
                            f_for_plic = np.clip(f_centered, 0.0, 1.0)

                        segments = self.plic_extractor.extract_interface_plic(f_for_plic, x_grid, y_grid)
                        contour_paths = self._segments_to_contour_paths(segments)
                        all_contours[level_name] = contour_paths

                        segment_count = len(segments)
                        total_segments += segment_count
                        print(f"     {level_name}: {segment_count} segments → {len(contour_paths)} paths")

                    except Exception as level_error:
                        print(f"     PLIC ERROR for {level_name}: {level_error}")
                        all_contours[level_name] = []

                print(f"     PLIC total: {total_segments} segments across all levels")
                return all_contours

            else:
                print(f"     PLIC: Extracting single level F={level:.3f}")
                segments = self.plic_extractor.extract_interface_plic(f_grid, x_grid, y_grid)
                contour_paths = self._segments_to_contour_paths(segments)
                print(f"     PLIC: {len(segments)} segments → {len(contour_paths)} paths")
                return contour_paths

        except Exception as outer_error:
            print(f"     OUTER PLIC ERROR: {outer_error}")
            print(f"     Falling back to scikit-image method...")
            return self._extract_interface_skimage(f_grid, x_grid, y_grid, level if not extract_all_levels else 0.5, extract_all_levels)

    def _extract_interface_conrec(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using CONREC algorithm."""
        f_for_contour = f_grid

        if extract_all_levels:
            mixing_levels = [0.05, 0.5, 0.95]
            print(f"     CONREC: Extracting multiple levels: {mixing_levels}")
            level_results = self.conrec_extractor.extract_multiple_levels(
                f_for_contour, x_grid, y_grid, mixing_levels
            )

            all_contours = {}
            total_segments = 0

            for level_name, segments in level_results.items():
                contour_paths = self._segments_to_contour_paths(segments)
                all_contours[level_name] = contour_paths

                segment_count = len(segments)
                total_segments += segment_count
                print(f"     {level_name}: {segment_count} segments → {len(contour_paths)} paths")

            print(f"     CONREC total: {total_segments} segments across all levels")
            return all_contours

        else:
            print(f"     CONREC: Extracting single level F={level:.3f}")
            segments = self.conrec_extractor.extract_interface_conrec(
                f_for_contour, x_grid, y_grid, level
            )
            contour_paths = self._segments_to_contour_paths(segments)
            print(f"     CONREC: {len(segments)} segments → {len(contour_paths)} paths")
            return contour_paths

    def _extract_interface_skimage(self, f_grid, x_grid, y_grid, level=0.5, extract_all_levels=True):
        """Extract interface using scikit-image (original method)."""
        # Check for binary data
        unique_vals = np.unique(f_grid)
        is_binary = len(unique_vals) <= 10

        if is_binary:
            print(f"     Binary VOF detected: {unique_vals}")
            print(f"     Applying smoothing for contour interpolation...")
            f_smoothed = ndimage.gaussian_filter(f_grid.astype(float), sigma=0.8)
            print(f"     After smoothing: min={np.min(f_smoothed):.3f}, max={np.max(f_smoothed):.3f}")
            f_for_contour = f_smoothed
        else:
            print(f"     Continuous F-field detected, using direct contouring")
            f_for_contour = f_grid

        if extract_all_levels:
            mixing_levels = {
                'lower_boundary': 0.05,
                'interface': 0.5,
                'upper_boundary': 0.95
            }

            all_contours = {}
            total_segments = 0

            for level_name, level_value in mixing_levels.items():
                try:
                    contours = measure.find_contours(f_for_contour.T, level_value)

                    # Convert to physical coordinates
                    physical_contours = []
                    for contour in contours:
                        if len(contour) > 1:
                            x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                            y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                            physical_contours.append(np.column_stack([x_physical, y_physical]))

                    all_contours[level_name] = physical_contours

                    level_segments = sum(len(contour) - 1 for contour in physical_contours if len(contour) > 1)
                    total_segments += level_segments

                    print(f"     F={level_value:.2f} ({level_name}): {len(physical_contours)} paths, {level_segments} segments")

                except Exception as e:
                    print(f"     F={level_value:.2f} ({level_name}): ERROR - {e}")
                    all_contours[level_name] = []

            print(f"     Total segments across all levels: {total_segments}")

            # If primary interface (F=0.5) failed, try alternative approaches
            if len(all_contours.get('interface', [])) == 0:
                print(f"     Primary interface (F=0.5) failed, trying adaptive approach...")
                all_contours['interface'] = self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)

            return all_contours

        else:
            try:
                contours = measure.find_contours(f_for_contour.T, level)
                print(f"     Found {len(contours)} contour paths for F={level:.2f}")

                physical_contours = []
                total_segments = 0

                for i, contour in enumerate(contours):
                    if len(contour) > 1:
                        x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                        y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                        physical_contours.append(np.column_stack([x_physical, y_physical]))

                        segments_in_path = len(contour) - 1
                        total_segments += segments_in_path

                print(f"     Total segments: {total_segments}")

                if total_segments < 10:
                    print(f"     Too few segments ({total_segments}), trying adaptive approach...")
                    adaptive_contours = self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)
                    if adaptive_contours:
                        return adaptive_contours

                return physical_contours

            except Exception as e:
                print(f"     ERROR in find_contours: {e}")
                print(f"     Falling back to adaptive method...")
                return self._extract_interface_adaptive(f_for_contour, x_grid, y_grid)

    def _extract_interface_adaptive(self, f_grid, x_grid, y_grid):
        """Adaptive interface extraction when standard contouring fails."""
        print(f"       Trying adaptive interface extraction...")

        test_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_contours = []
        best_count = 0
        best_level = 0.5

        for test_level in test_levels:
            try:
                contours = measure.find_contours(f_grid.T, test_level)

                physical_contours = []
                for contour in contours:
                    if len(contour) > 1:
                        x_physical = np.interp(contour[:, 1], np.arange(f_grid.shape[0]), x_grid[:, 0])
                        y_physical = np.interp(contour[:, 0], np.arange(f_grid.shape[1]), y_grid[0, :])
                        physical_contours.append(np.column_stack([x_physical, y_physical]))

                total_segments = sum(len(contour) - 1 for contour in physical_contours if len(contour) > 1)

                if total_segments > best_count:
                    best_count = total_segments
                    best_contours = physical_contours
                    best_level = test_level

            except:
                continue

        if best_count > 0:
            print(f"       Best adaptive level: F={best_level:.1f} with {best_count} segments")
            return best_contours

        print(f"       All adaptive methods failed")
        return []

    def _segments_to_contour_paths(self, segments):
        """Convert line segments back to contour paths for compatibility."""
        if not segments:
            return []

        contour_paths = []
        for (x1, y1), (x2, y2) in segments:
            contour_path = np.array([[x1, y1], [x2, y2]])
            contour_paths.append(contour_path)

        return contour_paths

    def convert_contours_to_segments(self, contours_input):
        """
        Convert contours to line segments, handles both single-level and multi-level input.
        """
        if isinstance(contours_input, dict):
            # Multi-level input - use primary interface (F=0.5)
            if 'interface' in contours_input and contours_input['interface']:
                contours = contours_input['interface']
                print(f"   Using primary interface contours: {len(contours)} paths")
            else:
                # Fallback to any available level
                for level_name, level_contours in contours_input.items():
                    if level_contours:
                        contours = level_contours
                        print(f"   Using {level_name} contours as fallback: {len(contours)} paths")
                        break
                else:
                    print(f"   No contours available in any level")
                    return []
        else:
            # Single-level input (backward compatibility)
            contours = contours_input

        # Convert contours to segments
        segments = []
        for contour in contours:
            for i in range(len(contour) - 1):
                x1, y1 = contour[i]
                x2, y2 = contour[i+1]
                segments.append(((x1, y1), (x2, y2)))

        print(f"   Converted to {len(segments)} line segments")
        return segments

    # rt_analyzer.py - ENHANCED VERSION
    # Part 4: Cache Management and Comprehensive Analysis

    # =================================================================
    # CACHE MANAGEMENT - Enhanced for Large Grids
    # =================================================================

    def _generate_cache_key(self, vtk_file_path):
        """Generate a unique cache key for a VTK file."""
        import hashlib
        try:
            stat = os.stat(vtk_file_path)
            file_info = f"{vtk_file_path}_{stat.st_mtime}_{stat.st_size}_{self.interface_value}"
            cache_key = hashlib.md5(file_info.encode()).hexdigest()
            return cache_key
        except OSError:
            return hashlib.md5(vtk_file_path.encode()).hexdigest()

    def _load_vtk_data(self, vtk_file_path):
        """Load VTK data from file."""
        try:
            return self.read_vtk_file(vtk_file_path)
        except Exception as e:
            print(f"Error loading VTK file {vtk_file_path}: {str(e)}")
            return None

    def _extract_interface_contour(self, vtk_data, interface_value):
        """Extract interface contour using existing method."""
        try:
            contours = self.extract_interface(
                vtk_data['f'], vtk_data['x'], vtk_data['y'],
                level=interface_value, extract_all_levels=False
            )
            interface_points = []
            for contour in contours:
                for point in contour:
                    interface_points.append((float(point[0]), float(point[1])))
            return interface_points
        except Exception as e:
            print(f"Error extracting interface: {str(e)}")
            return []

    def _extract_interface_comprehensive(self, vtk_file_path):
        """
        Enhanced comprehensive interface extraction that ALSO caches VTK data.
        Critical for large grids (1280×1600) to avoid expensive re-reads.
        """
        import time
        start_time = time.time()

        # Check if we already have cached data for this file
        cache_key = self._generate_cache_key(vtk_file_path)
        if cache_key in self.interface_cache:
            cached_data = self.interface_cache[cache_key]
            if 'vtk_data' in cached_data:
                print(f"  ✅ Using cached VTK data (avoiding re-read of large grid)")
                vtk_data = cached_data['vtk_data']
            else:
                vtk_data = self._load_vtk_data(vtk_file_path)
        else:
            # Load VTK data (expensive for 1280×1600!)
            print(f"  📂 Loading VTK data ({os.path.basename(vtk_file_path)})")
            vtk_data = self._load_vtk_data(vtk_file_path)
            
        if vtk_data is None:
            return None

        # Extract interface for fractal analysis (if needed)
        base_interface = self._extract_interface_contour(vtk_data, self.interface_value)
        if not base_interface:
            print(f"Warning: No interface found in {vtk_file_path}")
            # Don't return None - we can still do mixing analysis without interface contours
            base_interface = []

        # Calculate bounds for metadata (if we have interface)
        if base_interface:
            x_coords = [pt[0] for pt in base_interface]
            y_coords = [pt[1] for pt in base_interface]
            bounds = (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
        else:
            # Use grid bounds as fallback
            bounds = (
                np.min(vtk_data['x']), np.max(vtk_data['x']),
                np.min(vtk_data['y']), np.max(vtk_data['y'])
            )

        extraction_time = time.time() - start_time

        # Package comprehensive results INCLUDING VTK data for mixing analysis
        comprehensive_data = {
            'base_interface': base_interface,
            'vtk_data': vtk_data,  # 🔑 CACHE THE EXPENSIVE VTK DATA!
            'metadata': {
                'file_path': vtk_file_path,
                'extraction_time': extraction_time,
                'point_count': len(base_interface),
                'bounds': bounds,
                'grid_shape': vtk_data['f'].shape if 'f' in vtk_data else None
            }
        }

        # Cache the comprehensive data
        self.interface_cache[cache_key] = comprehensive_data
        print(f"  💾 Cached interface + VTK data for future reuse")

        return comprehensive_data

    def get_cache_stats(self):
        """Get statistics about cache usage - useful for large grid processing."""
        total_cached = len(self.interface_cache)
        cached_with_vtk = sum(1 for data in self.interface_cache.values() if 'vtk_data' in data)
        
        return {
            'total_cached_files': total_cached,
            'cached_with_vtk_data': cached_with_vtk,
            'cache_efficiency': cached_with_vtk / total_cached if total_cached > 0 else 0
        }

    # =================================================================
    # FRACTAL DIMENSION ANALYSIS - Preserved Legacy Methods
    # =================================================================

    def compute_fractal_dimension(self, data, min_box_size=None):
        """Compute fractal dimension of the interface using basic box counting."""
        if self.fractal_analyzer is None:
            print("Fractal analyzer not available. Skipping fractal dimension calculation.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

        # Extract contours
        contours = self.extract_interface(data['f'], data['x'], data['y'], extract_all_levels=False)
        segments = self.convert_contours_to_segments(contours)

        if not segments:
            print("No interface segments found.")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

        print(f"Found {len(segments)} interface segments")

        # Auto-estimate min_box_size if not provided
        if min_box_size is None:
            min_box_size = self.fractal_analyzer.estimate_min_box_size(segments)
            print(f"Auto-estimated min_box_size: {min_box_size:.8f}")
        else:
            print(f"Using provided min_box_size: {min_box_size:.8f}")

        try:
            # Use Wu-enhanced analyzer for automatic method selection
            from fractal_analyzer.core.data_types import SegmentArray
            segment_array = SegmentArray.from_list(segments)
            result = self.fractal_analyzer.analyze_mathematical_fractal(segment_array)

            if result is None:
                raise ValueError("Wu-enhanced analysis failed")

            optimal_dimension = result.dimension
            optimal_window = 1  # Default window for compatibility
            box_sizes = result.box_sizes
            box_counts = result.box_counts
            optimal_idx = 0
            error = 0.0  # Wu analyzer doesn't return error estimates in this form
            r_squared = result.r_squared

            print(f"Fractal dimension: {optimal_dimension:.6f} ± {error:.6f}, R² = {r_squared:.6f}")

            return {
                'dimension': optimal_dimension,
                'error': error,
                'r_squared': r_squared,
                'window_size': optimal_window,
                'box_sizes': box_sizes,
                'box_counts': box_counts,
                'segments': segments
            }

        except Exception as e:
            print(f"Error in fractal dimension calculation: {str(e)}")
            return {
                'dimension': np.nan,
                'error': np.nan,
                'r_squared': np.nan
            }

    def _compute_box_counting_dimension(self, interface_points):
        """Compute box counting fractal dimension from interface points."""
        try:
            if not interface_points:
                return np.nan, np.nan, np.nan

            # Convert to segments format for fractal analyzer
            segments = []
            for i in range(len(interface_points) - 1):
                p1 = interface_points[i]
                p2 = interface_points[i + 1]
                segments.append((p1, p2))

            if not segments:
                return np.nan, np.nan, np.nan

            # Use Wu-enhanced fractal analyzer if available
            if self.fractal_analyzer is not None:
                from fractal_analyzer.core.data_types import SegmentArray
                segment_array = SegmentArray.from_list(segments)
                result = self.fractal_analyzer.analyze_mathematical_fractal(segment_array)

                if result is not None:
                    optimal_dimension = result.dimension
                    optimal_error = 0.0  # Wu analyzer doesn't return error estimates in this form
                    optimal_r2 = result.r_squared
                    return optimal_dimension, optimal_error, optimal_r2
                else:
                    return np.nan, np.nan, np.nan
            else:
                return np.nan, np.nan, np.nan
        except Exception as e:
            return np.nan, np.nan, np.nan

    # =================================================================
    # COMPREHENSIVE ANALYSIS METHOD - Enhanced with Caching
    # =================================================================

    def analyze_vtk_file(self, vtk_file_path, analysis_types=None, y0=0.5, mixing_method='integral',
                        min_box_size=None, enable_multifractal=False, q_values=None, mf_output_dir=None):
        """
        Enhanced optimized analysis that can do BOTH fractal and mixing analysis
        using the same cached VTK data.
        """
        import time

        if analysis_types is None:
            analysis_types = ['fractal_dim', 'mixing']

        print(f"\n🚀 OPTIMIZED Analysis: {os.path.basename(vtk_file_path)}")
        analysis_start = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(vtk_file_path)
        cached_data = self.interface_cache.get(cache_key)

        if cached_data and 'vtk_data' in cached_data:
            print(f"✅ Using cached interface + VTK data")
            interface_data = cached_data
            vtk_data = cached_data['vtk_data']
        else:
            print(f"🔄 Extracting comprehensive interface + VTK data...")
            interface_data = self._extract_interface_comprehensive(vtk_file_path)
            if interface_data is None:
                return None
            vtk_data = interface_data['vtk_data']

        # Initialize results with metadata
        results = {
            'file_path': vtk_file_path,
            'time': vtk_data.get('time', 0.0),
            'interface_extraction_time': interface_data['metadata']['extraction_time'],
            'interface_point_count': interface_data['metadata']['point_count'],
            'interface_bounds': interface_data['metadata']['bounds'],
            'grid_shape': interface_data['metadata']['grid_shape'],
            'analysis_types_performed': analysis_types
        }

        # MIXING ANALYSIS (using cached VTK data - no re-reading!)
        if 'mixing' in analysis_types:
            print("  🧪 Computing mixing analysis...")
            mixing_start = time.time()
    
            # Use the comprehensive mixing analysis method
            mixing_results = self.compute_mixing_analysis(vtk_data, y0, method=mixing_method)
    
            # Update results
            results.update(mixing_results)
    
            # Display results based on what's available
            if 'h_01' in mixing_results:
                print(f"    Mass-conservation: h₀₁={mixing_results['h_01']:.6f}, h₁₀={mixing_results['h_10']:.6f}")
    
            if 'h_penetration_up' in mixing_results:
                print(f"    Geometric: up={mixing_results['h_penetration_up']:.6f}, down={mixing_results['h_penetration_down']:.6f}")
    
            if 'W' in mixing_results:
                print(f"    Youngs: W={mixing_results['W']:.6f}")
    
            if 'total_mixing_integral' in mixing_results:
                print(f"    Total mixing integral: {mixing_results['total_mixing_integral']:.6f}")
                print(f"    Mixing ratio: {mixing_results['mixing_ratio']:.6f}")
    
            results['mixing_computation_time'] = time.time() - mixing_start   

        # FRACTAL ANALYSIS (using cached interface segments)
        if 'fractal_dim' in analysis_types:
            print("  📐 Computing fractal dimension...")
            fractal_start = time.time()

            base_interface = interface_data['base_interface']
            if base_interface:
                fractal_dim, fractal_error, fractal_r2 = self._compute_box_counting_dimension(base_interface)
                results['fractal_dimension'] = fractal_dim
                results['fractal_error'] = fractal_error
                results['fractal_r_squared'] = fractal_r2
            else:
                print("    Warning: No interface segments for fractal analysis")
                results['fractal_dimension'] = np.nan
                results['fractal_error'] = np.nan
                results['fractal_r_squared'] = np.nan
                
            results['fractal_computation_time'] = time.time() - fractal_start

        total_time = time.time() - analysis_start
        results['total_analysis_time'] = total_time

        print(f"✅ OPTIMIZED Analysis complete: {total_time:.2f}s")
        if 'grid_shape' in results and results['grid_shape']:
            print(f"   Grid: {results['grid_shape'][0]}×{results['grid_shape'][1]} cells")
        print(f"   Interface extraction: {interface_data['metadata']['extraction_time']:.2f}s")

        # Multifractal analysis if requested
        if enable_multifractal and interface_data and interface_data.get("base_interface"):
            print(f"\n🔬 MULTIFRACTAL ANALYSIS")
            self.enable_multifractal_analysis()
            
            if mf_output_dir is None:
                mf_output_dir = os.path.join(self.output_dir, "multifractal")
            os.makedirs(mf_output_dir, exist_ok=True)
            
            # Convert interface data to segments for multifractal analysis
            segments = []
            base_interface = interface_data["base_interface"]
            for i in range(len(base_interface) - 1):
                p1 = base_interface[i]
                p2 = base_interface[i + 1]
                segments.append((p1, p2))
            
            try:
                if self.multifractal_analyzer:
                    mf_results = self.multifractal_analyzer.compute_multifractal_spectrum(
                        segments,
                        min_box_size=min_box_size,
                        q_values=q_values,
                        output_dir=mf_output_dir,
                        time_value=results.get('time')
                    )
                    
                    if mf_results:
                        results["multifractal"] = mf_results
                        self.multifractal_analyzer.print_multifractal_summary(mf_results)
                    else:
                        results["multifractal"] = None
                else:
                    print("❌ Multifractal analyzer not available")
                    results["multifractal"] = None
                    
            except Exception as e:
                print(f"❌ Multifractal analysis error: {str(e)}")
                results["multifractal"] = None

        return results

# rt_analyzer.py - ENHANCED VERSION
# Part 5: Series Processing and Main Function

    # =================================================================
    # ENHANCED SERIES PROCESSING - Time-Sequential with Caching
    # =================================================================

    def process_vtk_series_enhanced(self, pattern_or_dir: str, 
                                  analysis_types=None,
                                  grid_filter: Optional[str] = None,
                                  time_range: Optional[Tuple[float, float]] = None,
                                  y0=None,
                                  mixing_method='integral',
                                  output_csv=None,
                                  group_by_resolution=True):
        """
        Enhanced VTK series processing with proper filename parsing and time sorting.
        Perfect for large grids (1280×1600) with caching optimization.
        """
        if analysis_types is None:
            analysis_types = ['mixing', 'fractal_dim']
        
        print(f"🚀 Enhanced VTK Series Processing")
        print(f"   Pattern: {pattern_or_dir}")
        print(f"   Analysis: {analysis_types}")
        if grid_filter:
            print(f"   Grid filter: {grid_filter}")
        if time_range:
            print(f"   Time range: {time_range[0]} → {time_range[1]}")
        
        # Find and sort VTK files
        file_info_list = VTKFileHandler.find_and_sort_vtk_files(
            pattern_or_dir, grid_filter=grid_filter, time_range=time_range
        )
        
        if not file_info_list:
            return None
        
        if group_by_resolution and not grid_filter:
            # Process each resolution separately
            groups = VTKFileHandler.group_by_resolution(file_info_list)
            results_by_resolution = {}
            
            for grid_type, files in groups.items():
                print(f"\n🔬 Processing {grid_type} resolution ({len(files)} files)")
                
                results = []
                cache_hits = 0
                
                for i, file_info in enumerate(files):
                    vtk_file = file_info['full_path']
                    print(f"  File {i+1}/{len(files)}: {file_info['filename']} (t={file_info['sim_time']:.3f})")
                    
                    # Check cache status
                    cache_key = self._generate_cache_key(vtk_file)
                    if cache_key in self.interface_cache:
                        cache_hits += 1
                    
                    try:
                        result = self.analyze_vtk_file(
                            vtk_file,
                            analysis_types=analysis_types,
                            y0=y0,
                            mixing_method=mixing_method
                        )
                        if result:
                            # Add filename metadata
                            result.update({
                                'nx': file_info['nx'],
                                'ny': file_info['ny'],
                                'grid_type': file_info['grid_type'],
                                'time_step': file_info['time_step'],
                                'filename': file_info['filename']
                            })
                            results.append(result)
                            
                    except Exception as e:
                        print(f"    ❌ Error: {e}")
                        continue
                
                print(f"  ✅ {grid_type} complete: {len(results)} successful, {cache_hits} cache hits")
                
                if results:
                    df = pd.DataFrame(results)
                    df = df.sort_values('time')  # Final sort by actual simulation time
                    results_by_resolution[grid_type] = df

                    # CORRECTED: Always save CSV if output_csv provided
                    if output_csv:
                        base_name = output_csv.replace('.csv', '')
                        resolution_csv = f"{base_name}_{grid_type}.csv"
                        df.to_csv(resolution_csv, index=False)
                        print(f"  💾 CSV saved to: {resolution_csv}")
            
            return results_by_resolution
        
        else:
            # Process all files as single series
            print(f"\n🔬 Processing all files as single series ({len(file_info_list)} files)")
            
            results = []
            cache_hits = 0
            
            for i, file_info in enumerate(file_info_list):
                vtk_file = file_info['full_path']
                print(f"  File {i+1}/{len(file_info_list)}: {file_info['filename']} (t={file_info['sim_time']:.3f})")
                
                # Check cache status
                cache_key = self._generate_cache_key(vtk_file)
                if cache_key in self.interface_cache:
                    cache_hits += 1
                
                try:
                    result = self.analyze_vtk_file(
                        vtk_file,
                        analysis_types=analysis_types,
                        y0=y0,
                        mixing_method=mixing_method
                    )
                    if result:
                        # Add filename metadata
                        result.update({
                            'nx': file_info['nx'],
                            'ny': file_info['ny'],
                            'grid_type': file_info['grid_type'],
                            'time_step': file_info['time_step'],
                            'filename': file_info['filename']
                        })
                        results.append(result)
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    continue
            
            print(f"✅ Series complete: {len(results)} successful, {cache_hits} cache hits")
            
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values('time')  # Final sort by actual simulation time
               
                # CORRECTED: Always save CSV if output_csv provided 
                if output_csv:
                    df.to_csv(output_csv, index=False)
                    print(f"💾 CSV saved to: {output_csv}")
                
                return df
            else:
                return None

    # =================================================================
    # CLEAN INTERFACE METHODS - Simple Analysis Functions
    # =================================================================

    def analyze_mixing_only(self, vtk_file, y0=None):
        """Simple interface for mixing analysis only."""
        return self.analyze_vtk_file(vtk_file, analysis_types=['mixing'], y0=y0)

    def analyze_fractal_only(self, vtk_file, min_box_size=None):
        """Simple interface for fractal analysis only."""
        return self.analyze_vtk_file(vtk_file, analysis_types=['fractal_dim'], min_box_size=min_box_size)

    # =================================================================
    # LEGACY COMPATIBILITY METHODS - Preserved for Backward Compatibility
    # =================================================================

    def find_initial_interface(self, data):
        """Legacy method: Find the initial interface position."""
        return self._find_initial_interface_position(data['f'], data['y'])

    def process_vtk_series(self, vtk_pattern, resolution=None, mixing_method='integral'):
        """Legacy method: Process a series of VTK files."""
        print("⚠️  Using legacy process_vtk_series method. Consider using process_vtk_series_enhanced.")
        
        # Convert to new method call
        if resolution:
            grid_filter = f"{resolution}x{resolution}" if isinstance(resolution, int) else None
        else:
            grid_filter = None
            
        return self.process_vtk_series_enhanced(
            vtk_pattern,
            analysis_types=['mixing', 'fractal_dim'],
            grid_filter=grid_filter,
            mixing_method=mixing_method,
            group_by_resolution=False
        )

    def create_summary_plots(self, df, output_dir, method_subset=None):
        """
        UPDATED: Create comprehensive mixing plots with proper notation
        
        Args:
            df: DataFrame with time series results  
            output_dir: Output directory for plots
            method_subset: Optional list ['mass', 'geometric', 'diagnostic', 'youngs']
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which methods are available in the data
        available_methods = []
        if 'h_01' in df.columns or 'h_total_mass' in df.columns:
            available_methods.append('mass')
        if 'h_penetration_up' in df.columns or 'h_total_geometric' in df.columns:
            available_methods.append('geometric')  
        if 'mixing_ratio' in df.columns or 'total_mixing_integral' in df.columns:
            available_methods.append('diagnostic')
        if 'W' in df.columns or 'integral_width' in df.columns:
            available_methods.append('youngs')
        
        # Filter methods if subset specified
        if method_subset:
            available_methods = [m for m in available_methods if m in method_subset]
        
        print(f"📊 Creating plots for methods: {available_methods}")
        
        # Plot 1: Mass-Conservation Evolution (if available)
        if 'mass' in available_methods:
            self._plot_mass_conservation_evolution(df, output_dir)
        
        # Plot 2: Geometric Penetration Evolution (if available)  
        if 'geometric' in available_methods:
            self._plot_geometric_evolution(df, output_dir)
        
        # Plot 3: Global Mixedness Evolution (if available)
        if 'diagnostic' in available_methods:
            self._plot_mixing_diagnostic_evolution(df, output_dir)
        
        # Plot 4: Youngs Width Evolution (if available)
        if 'youngs' in available_methods:
            self._plot_youngs_evolution(df, output_dir)

        # Plot 5: Four-Method Comparison (if multiple methods available)
        if len(available_methods) >= 2:
            self._plot_method_comparison(df, output_dir, available_methods)
        
        # Plot 6: Fractal dimension evolution (if available)
        if 'fractal_dimension' in df.columns:
            self._plot_fractal_evolution(df, output_dir)
        
        print(f"✅ All plots saved to: {output_dir}")

    def _plot_mass_conservation_evolution(self, df, output_dir):
        """Plot mass-conservation thickness evolution with proper h₀₁, h₁₀ notation"""
        
        h_01_col = 'h_01' if 'h_01' in df.columns else None
        h_10_col = 'h_10' if 'h_10' in df.columns else None
        h_total_col = 'h_total_mass' if 'h_total_mass' in df.columns else 'h_total'
        
        if not h_01_col or not h_10_col:
            print("   ⚠️ Mass conservation data not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(df['time'], df[h_total_col], 'b-', linewidth=2.5, label='h₀₁ + h₁₀ (total)')
        plt.plot(df['time'], df[h_01_col], 'r--', linewidth=2, label='h₀₁ (bottom fluid above y₀)')
        plt.plot(df['time'], df[h_10_col], 'g--', linewidth=2, label='h₁₀ (top fluid below y₀)')
        
        plt.xlabel('Time')
        plt.ylabel('Equivalent Thickness')
        if not self.no_titles:
            plt.title('Mass-Conservation Analysis: Equivalent Fluid Thicknesses')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'mass_conservation_evolution.png'), dpi=300)
        plt.close()
        print(f"   ✓ Mass-conservation evolution plot")

    def _plot_geometric_evolution(self, df, output_dir):
        """Plot geometric penetration evolution"""
        
        h_up_col = 'h_penetration_up' if 'h_penetration_up' in df.columns else None
        h_down_col = 'h_penetration_down' if 'h_penetration_down' in df.columns else None
        h_total_col = 'h_total_geometric' if 'h_total_geometric' in df.columns else None
        
        if not h_up_col or not h_down_col:
            print("   ⚠️ Geometric penetration data not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        if h_total_col and h_total_col in df.columns:
            plt.plot(df['time'], df[h_total_col], 'purple', linewidth=2.5, label='Total mixing zone')
        
        plt.plot(df['time'], df[h_up_col], 'orange', linestyle='--', linewidth=2, 
                 label='Upward penetration')
        plt.plot(df['time'], df[h_down_col], 'brown', linestyle='--', linewidth=2, 
                 label='Downward penetration')
        
        plt.xlabel('Time')
        plt.ylabel('Penetration Distance')
        if not self.no_titles:
            plt.title('Geometric Analysis: Mixing Zone Penetration Distances')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'geometric_penetration_evolution.png'), dpi=300)
        plt.close()
        print(f"   ✓ Geometric penetration evolution plot")

    def _plot_mixing_diagnostic_evolution(self, df, output_dir):
        """Plot global mixedness diagnostic evolution"""
        
        mixing_ratio_col = 'mixing_ratio' if 'mixing_ratio' in df.columns else None
        mixing_efficiency_col = 'mixing_efficiency' if 'mixing_efficiency' in df.columns else None
        total_mixing_col = 'total_mixing_integral' if 'total_mixing_integral' in df.columns else None
        
        if not mixing_ratio_col and not total_mixing_col:
            print("   ⚠️ Mixing diagnostic data not found")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Subplot 1: Mixing ratio (normalized)
        if mixing_ratio_col:
            ax1.plot(df['time'], df[mixing_ratio_col], 'darkgreen', linewidth=2.5, marker='o', markersize=4)
            ax1.axhline(y=0.25, color='red', linestyle=':', alpha=0.7, label='Theoretical maximum (0.25)')
            ax1.set_ylabel('Mixing Ratio')
            ax1.set_title('Global Mixedness: ∫∫f(1-f)dA / (H×L)' if not self.no_titles else '')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Mixing efficiency or total integral
        if mixing_efficiency_col:
            ax2.plot(df['time'], df[mixing_efficiency_col] * 100, 'darkblue', 
                    linewidth=2.5, marker='s', markersize=4)
            ax2.set_ylabel('Mixing Efficiency (%)')
            ax2.set_title('Mixing Efficiency (% of theoretical maximum)' if not self.no_titles else '')
            ax2.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='Perfect mixing (100%)')
            ax2.legend()
        elif total_mixing_col:
            ax2.plot(df['time'], df[total_mixing_col], 'darkblue', linewidth=2.5, marker='s', markersize=4)
            ax2.set_ylabel('Total Mixing Integral')
            ax2.set_title('Total Mixing: ∫∫f(1-f)dA' if not self.no_titles else '')
        
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mixing_diagnostic_evolution.png'), dpi=300)
        plt.close()
        print(f"   ✓ Mixing diagnostic evolution plot")

    def _plot_youngs_evolution(self, df, output_dir):
        """Plot Youngs integral width evolution"""
        
        W_col = 'W' if 'W' in df.columns else 'integral_width'
        centroid_col = 'mixing_centroid' if 'mixing_centroid' in df.columns else None
        
        if W_col not in df.columns:
            print("   ⚠️ Youngs integral width data not found")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Subplot 1: Integral width W
        ax1.plot(df['time'], df[W_col], 'magenta', linewidth=2.5, marker='d', markersize=4)
        ax1.set_ylabel('Youngs Integral Width W')
        ax1.set_title('Youngs Method: W = ∫f̄(1-f̄)dy' if not self.no_titles else '')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Mixing centroid if available
        if centroid_col and centroid_col in df.columns:
            ax2.plot(df['time'], df[centroid_col], 'darkviolet', linewidth=2, 
                    marker='o', markersize=3, label='Mixing centroid')
            ax2.set_ylabel('Mixing Centroid Position')
            ax2.set_title('Mixing Zone Center Position' if not self.no_titles else '')
            ax2.legend()
        else:
            # Show ratios to other methods if available
            ax2.set_ylabel('Youngs Ratios')
            ax2.set_title('Youngs Width Ratios' if not self.no_titles else '')
            
            if 'h_total_geometric' in df.columns:
                ratios = df['h_total_geometric'] / df[W_col]
                ax2.plot(df['time'], ratios, 'red', linewidth=2, marker='s', markersize=3, 
                        label='h_geometric / W')
            
            if 'h_total_mass' in df.columns:
                ratios = df['h_total_mass'] / df[W_col]
                ax2.plot(df['time'], ratios, 'blue', linewidth=2, marker='^', markersize=3, 
                        label='h_mass / W')
            
            ax2.legend()
        
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'youngs_evolution.png'), dpi=300)
        plt.close()
        print(f"   ✓ Youngs integral width evolution plot")

    def _plot_method_comparison(self, df, output_dir, available_methods):
        """Plot comparison of total thicknesses from different methods"""
        
        plt.figure(figsize=(12, 8))
        
        colors = {'mass': 'blue', 'geometric': 'red', 'youngs': 'purple', 'diagnostic': 'green'}
        linestyles = {'mass': '-', 'geometric': '--', 'youngs': '-.', 'diagnostic': ':'}
        
        # Plot each available method's total measure
        for method in available_methods:
            if method == 'mass':
                col = 'h_total_mass' if 'h_total_mass' in df.columns else 'h_total'
                if col in df.columns:
                    plt.plot(df['time'], df[col], color=colors[method], linestyle=linestyles[method], 
                            linewidth=2.5, label='Mass-conservation total (h₀₁ + h₁₀)')
            
            elif method == 'geometric':
                col = 'h_total_geometric'
                if col in df.columns:
                    plt.plot(df['time'], df[col], color=colors[method], linestyle=linestyles[method], 
                            linewidth=2.5, label='Geometric total (penetration distances)')
            
            elif method == 'youngs':
                col = 'W' if 'W' in df.columns else 'integral_width'
                if col in df.columns:
                    plt.plot(df['time'], df[col], color=colors[method], linestyle=linestyles[method], 
                            linewidth=2.5, label='Youngs integral width (W)')
            
            elif method == 'diagnostic':
                col = 'mixing_ratio'
                if col in df.columns:
                    # Scale mixing ratio for comparison
                    scale = 10  # Simple scale factor
                    plt.plot(df['time'], df[col] * scale, color=colors[method], linestyle=linestyles[method], 
                            linewidth=2.5, label=f'Mixing ratio × {scale} (scaled)')
        
        plt.xlabel('Time')
        plt.ylabel('Thickness / Width')
        if not self.no_titles:
            plt.title('Four-Method Comparison: Total Mixing Measures')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'four_method_comparison.png'), dpi=300)
        plt.close()
        print(f"   ✓ Four-method comparison plot")

    def _plot_fractal_evolution(self, df, output_dir):
        """Plot fractal dimension evolution over time."""

        if 'fractal_dimension' not in df.columns:
            return

        # Filter out NaN values
        valid_data = df.dropna(subset=['fractal_dimension'])

        if len(valid_data) == 0:
            print("   ⚠️ No valid fractal dimension data found")
            return

        plt.figure(figsize=(10, 6))

        # Convert time from filename units to physical seconds (divide by 1000)
        time_seconds = valid_data['time'] / 1000.0

        # Plot fractal dimension with error bars if available
        if 'fractal_error' in df.columns:
            plt.errorbar(time_seconds, valid_data['fractal_dimension'],
                        yerr=valid_data['fractal_error'],
                        fmt='o-', linewidth=2, markersize=4, capsize=3,
                        label='Fractal dimension ± error')
        else:
            plt.plot(time_seconds, valid_data['fractal_dimension'],
                    'o-', linewidth=2, markersize=4, label='Fractal dimension')

        # Set dynamic Y-axis limits based on data range
        y_min = valid_data['fractal_dimension'].min()
        y_max = valid_data['fractal_dimension'].max()
        y_range = y_max - y_min
        y_margin = max(0.05, y_range * 0.1)  # At least 0.05 margin, or 10% of range
        plt.ylim(y_min - y_margin, y_max + y_margin)

        # Add reference lines
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Smooth line (D=1)')
        plt.axhline(y=2.0, color='red', linestyle=':', alpha=0.7, label='Space-filling (D=2)')

        plt.xlabel('Time (s)')
        plt.ylabel('Fractal Dimension')
        if not self.no_titles:
            plt.title('Interface Fractal Dimension Evolution')

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'fractal_dimension_evolution.png'), dpi=300)
        plt.close()
        print(f"   ✓ Fractal dimension evolution plot")

# =================================================================
# MAIN FUNCTION - Enhanced Command Line Interface
# =================================================================

def main():
    """Enhanced main function with new capabilities."""
    parser = argparse.ArgumentParser(
        description='Enhanced Rayleigh-Taylor Simulation Analyzer with Clean Mixing Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Examples:
  # Analyze single file with new mixing method
  python rt_analyzer.py --file RT160x200-9000.vtk --mixing-only

  # Process time series with resolution filtering
  python rt_analyzer.py --pattern "RT*.vtk" --grid-filter "400x400" --time-range 1.0 10.0

  # Process large grid series with caching optimization
  python rt_analyzer.py --pattern "RT1280x1600-*.vtk" --analysis mixing

  # Group analysis by resolution
  python rt_analyzer.py --directory ./vtk_files --group-by-resolution

  # Legacy compatibility
  python rt_analyzer.py --file RT_0009000.vtk --use-conrec

  # Fast integral mixing analysis (good for large grids)
  python rt_analyzer.py --file RT160x200-9000.vtk --mixing_method integral

  # Standard Dalziel mixing analysis (literature comparable)
  python rt_analyzer.py --file RT160x200-9000.vtv --mixing_method dalziel

  # Compare both methods
  python rt_analyzer.py --file RT160x200-9000.vtk --mixing_method both

  # Process time series with Dalziel method
  python rt_analyzer.py --pattern "RT*.vtk" --mixing_method dalziel --output-csv dalziel_results.csv

  # Process large grids with fast integral method
  python rt_analyzer.py --pattern "RT1280x1600-*.vtk" --mixing_method integral --mixing-only

  # Group analysis by resolution
  python rt_analyzer.py --directory ./vtk_files --group-by-resolution --mixing_method both
""")

    parser.add_argument('--file', help='Single VTK file to analyze')
    parser.add_argument('--pattern', help='Pattern for VTK files (e.g., "RT*.vtk")')
    parser.add_argument('--directory', help='Directory containing VTK files')
    parser.add_argument('--output_dir', default='./rt_analysis',
                       help='Output directory for results (default: ./rt_analysis)')
    
    # Analysis type options
    parser.add_argument('--analysis', choices=['mixing', 'fractal', 'both'], default='both',
                       help='Type of analysis to perform')
    parser.add_argument('--mixing-only', action='store_true',
                       help='Perform only mixing analysis (fast)')
    parser.add_argument('--fractal-only', action='store_true',
                       help='Perform only fractal analysis')
    
    # Filtering options
    parser.add_argument('--grid-filter', help='Filter by grid resolution (e.g., "400x400")')
    parser.add_argument('--time-range', nargs=2, type=float, metavar=('MIN', 'MAX'),
                       help='Filter by time range (e.g., --time-range 1.0 10.0)')
    
    # Processing options
    parser.add_argument('--group-by-resolution', action='store_true',
                       help='Group analysis by grid resolution')
    parser.add_argument('--output-csv', help='Custom CSV filename (auto-generated if not specified)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable automatic plot generation (plots enabled by default)')
    # Physics parameters
    parser.add_argument('--h0', type=float, help='Initial interface position (auto-detected if not provided)')
    parser.add_argument('--mixing_method', choices=['mass_conservation', 'geometric', 'diagnostic', 'youngs', 'comprehensive'], 
                       default='comprehensive',
                       help='Method for mixing analysis: mass_conservation (your theory), geometric (Dalziel), diagnostic (global mixedness), youngs (robust width), or comprehensive (all methods)')

    parser.add_argument('--min_box_size', type=float,
                       help='Minimum box size for fractal analysis (auto-estimated if not provided)')
    
    # Technical options
    parser.add_argument('--use_grid_optimization', action='store_true',
                       help='Use grid optimization for fractal dimension calculation')
    parser.add_argument('--no_titles', action='store_true',
                       help='Disable plot titles for journal submissions')
    parser.add_argument('--use-conrec', action='store_true',
                       help='Use CONREC algorithm for precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                       help='Use PLIC algorithm for theoretical VOF interface reconstruction')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    # Multifractal options
    parser.add_argument('--multifractal', action='store_true',
                       help='Enable multifractal analysis')
    parser.add_argument('--q-values', nargs='+', type=float, default=None,
                       help='Q values for multifractal analysis')
    parser.add_argument('--mf-output-dir', default=None,
                       help='Output directory for multifractal results')

    args = parser.parse_args()

    # Validate arguments
    if not any([args.file, args.pattern, args.directory]):
        print("Error: Must specify --file, --pattern, or --directory")
        parser.print_help()
        return

    # Determine analysis types
    if args.mixing_only:
        analysis_types = ['mixing']
    elif args.fractal_only:
        analysis_types = ['fractal_dim']
    elif args.analysis == 'mixing':
        analysis_types = ['mixing']
    elif args.analysis == 'fractal':
        analysis_types = ['fractal_dim']
    else:  # both or default
        analysis_types = ['mixing', 'fractal_dim']

    # Create analyzer instance
    analyzer = RTAnalyzer(
        output_dir=args.output_dir,
        use_grid_optimization=args.use_grid_optimization,
        no_titles=args.no_titles,
        use_conrec=getattr(args, 'use_conrec', False),
        use_plic=getattr(args, 'use_plic', False),
        debug=args.debug
    )

    print(f"🚀 Enhanced RT Analyzer initialized")
    # Determine active extraction method for display
    if getattr(args, 'use_plic', False):
        extraction_method = "PLIC (theoretical reconstruction)"
    elif getattr(args, 'use_conrec', False):
        extraction_method = "CONREC (precision)"
    else:
        extraction_method = "scikit-image (standard)"

    print(f"   Interface extraction: {extraction_method}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Analysis types: {analysis_types}")
    print(f"   Grid optimization: {'ENABLED' if args.use_grid_optimization else 'DISABLED'}")

    try:
        if args.file:
            # Single file analysis
            print(f"\n📄 Analyzing single file: {args.file}")
            result = analyzer.analyze_vtk_file(
                args.file,
                analysis_types=analysis_types,
                y0=args.h0,
                mixing_method=args.mixing_method,
                min_box_size=args.min_box_size,
                enable_multifractal=args.multifractal,
                q_values=args.q_values,
                mf_output_dir=args.mf_output_dir
            )

            # Display single file results
            if result:
                print(f"\n✅ Results for {args.file}:")
                print(f"   Time: {result['time']:.6f}")
                if 'h_01' in result:
                    print(f"   Mass-conservation: h₀₁={result['h_01']:.6f}, h₁₀={result['h_10']:.6f}")
                    # Use the right total field name depending on what's available  
                    total_mass = result.get('h_total_mass', result.get('h_total', 0))
                    print(f"   Total mixing thickness: {total_mass:.6f}")
                if 'h_penetration_up' in result:
                    print(f"   Geometric penetration: up={result['h_penetration_up']:.6f}, down={result['h_penetration_down']:.6f}")
                    # Also show geometric total if available
                    if 'h_total_geometric' in result:
                        print(f"   Total geometric thickness: {result['h_total_geometric']:.6f}")
                if 'W' in result:
                    print(f"   Youngs integral width: W={result['W']:.6f}")
                if 'total_mixing_integral' in result:
                    print(f"   Mixing integral: {result['total_mixing_integral']:.6f}")
                    print(f"   Mixing ratio: {result['mixing_ratio']:.6f}")
                if 'fractal_dimension' in result and not np.isnan(result['fractal_dimension']):
                    print(f"   Fractal dimension: {result['fractal_dimension']:.6f} ± {result.get('fractal_error', 0):.6f}")
        else:
            # Series analysis - ALWAYS create CSV and plots by default
            if args.directory:
                pattern_or_dir = args.directory
            else:
                pattern_or_dir = args.pattern

            print(f"\n📊 Processing time series: {pattern_or_dir}")

            # Generate automatic CSV filename if not provided
            if args.output_csv:
                output_csv = args.output_csv
            else:
                # Auto-generate CSV filename based on input
                base_name = os.path.basename(pattern_or_dir).replace('*', 'series').replace('.vtk', '')
                if not base_name or base_name == '':
                    base_name = 'rt_analysis'
                output_csv = os.path.join(args.output_dir, f"{base_name}_results.csv")
            
            print(f"📄 CSV output: {output_csv}")
            
            results = analyzer.process_vtk_series_enhanced(
                pattern_or_dir,
                analysis_types=analysis_types,
                grid_filter=args.grid_filter,
                time_range=tuple(args.time_range) if args.time_range else None,
                y0=args.h0,
                mixing_method=args.mixing_method,
                output_csv=output_csv,  # ALWAYS provide CSV filename
                group_by_resolution=args.group_by_resolution
            )
        
            # Display series results summary
            if results is not None:
                if isinstance(results, dict):
                    # Grouped by resolution
                    print(f"\n✅ Time series analysis complete (grouped by resolution):")
                    for grid_type, df in results.items():
                        print(f"   {grid_type}: {len(df)} files processed")
                        if len(df) > 0:
                            print(f"     Time range: {df['time'].min():.3f} → {df['time'].max():.3f}")
                else:
                    # Single series
                    print(f"\n✅ Time series analysis complete:")
                    print(f"   Processed {len(results)} files")
                    print(f"   Time range: {results['time'].min():.3f} → {results['time'].max():.3f}")
            # Create plots by default (unless --no-plots specified)
            if not args.no_plots and results is not None:
                print(f"\n📊 Creating summary plots...")
                if isinstance(results, dict):
                    # Multiple resolutions
                    for grid_type, df in results.items():
                        plot_dir = os.path.join(args.output_dir, f"plots_{grid_type}")
                        os.makedirs(plot_dir, exist_ok=True)
                        analyzer.create_summary_plots(df, plot_dir)
                        print(f"   📈 Plots for {grid_type} saved to: {plot_dir}")
                else:
                    # Single series
                    plot_dir = os.path.join(args.output_dir, "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    analyzer.create_summary_plots(results, plot_dir)
                    print(f"   📈 Plots saved to: {plot_dir}")
            elif args.no_plots:
                print(f"📊 Plot generation disabled (--no-plots)")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Display cache statistics
    cache_stats = analyzer.get_cache_stats()
    print(f"\n📈 Cache Statistics:")
    print(f"   Total cached files: {cache_stats['total_cached_files']}")
    print(f"   Files with VTK data: {cache_stats['cached_with_vtk_data']}")
    print(f"   Cache efficiency: {cache_stats['cache_efficiency']:.1%}")

    print(f"\n🎉 Analysis complete. Results saved to: {args.output_dir}")
    return 0

if __name__ == "__main__":
    exit(main())


