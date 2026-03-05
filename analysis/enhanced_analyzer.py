#!/usr/bin/env python3
"""
ENHANCED Hybrid Parallel Resolution Analyzer with MULTIFRACTAL ANALYSIS: 
Comprehensive tool for temporal evolution, convergence analysis, and multifractal spectrum analysis.

MAJOR ENHANCEMENTS:
- UPDATED: Compatible with new rt_analyzer.py four-method framework
- ADDED: Dimensionless physics plots (h_t/H, h_b/H vs. τ = √(Ag/H)×t)
- ADDED: Auto-detection of domain dimensions (H, L) from VTK data
- ADDED: Physics parameters with smart defaults (A, g)
- FIXED: All field name compatibility with new rt_analyzer structure
- FIXED: Mixing method compatibility (comprehensive, mass_conservation, geometric, etc.)

Block 1: Imports, utilities, and core parameter handling
"""

import os
import time
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# Import the updated RTAnalyzer
try:
    from rt_analyzer import RTAnalyzer
except ImportError:
    try:
        from fractal_analyzer.core.rt_analyzer import RTAnalyzer
    except ImportError:
        print("❌ Could not import RTAnalyzer. Please ensure rt_analyzer.py is available.")
        exit(1)

def parse_grid_resolution(resolution_str):
    """
    Parse grid resolution from string, supporting both square and rectangular formats.
    
    Args:
        resolution_str: String like "200", "200x200", "160x200", etc.
        
    Returns:
        tuple: (nx, ny) for the grid dimensions
    """
    if 'x' in resolution_str:
        # Rectangular format: "160x200"
        parts = resolution_str.split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution format: {resolution_str}")
        nx, ny = int(parts[0]), int(parts[1])
        return nx, ny
    else:
        # Square format: "200"
        res = int(resolution_str)
        return res, res

def format_grid_resolution(nx, ny):
    """Format grid resolution for display and file naming."""
    return f"{nx}x{ny}"

def validate_grid_resolution_input(resolution_input):
    """
    Validate and standardize resolution input.
    
    Args:
        resolution_input: Can be int, string, or mixed list
        
    Returns:
        list: List of standardized resolution strings
    """
    if isinstance(resolution_input, (int, str)):
        # Single resolution
        return [str(resolution_input)]
    elif isinstance(resolution_input, list):
        # Multiple resolutions - convert all to strings
        return [str(res) for res in resolution_input]
    else:
        raise ValueError(f"Invalid resolution input type: {type(resolution_input)}")

def auto_detect_domain_dimensions(data_dirs, resolutions):
    """
    NEW: Auto-detect domain dimensions H and L from first available VTK file.
    
    Args:
        data_dirs: List of data directories
        resolutions: List of resolution strings
        
    Returns:
        tuple: (H, L) domain height and width in meters
    """
    print(f"🔍 Auto-detecting domain dimensions from VTK data...")
    
    # Try each data directory until we find a VTK file
    for data_dir, resolution_str in zip(data_dirs, resolutions):
        try:
            # Find any VTK file in this directory
            vtk_pattern = os.path.join(data_dir, "*.vtk")
            vtk_files = glob.glob(vtk_pattern)
            
            if not vtk_files:
                continue
                
            # Use first VTK file to extract domain dimensions
            first_vtk = vtk_files[0]
            print(f"   Reading domain from: {os.path.basename(first_vtk)}")
            
            # Create temporary analyzer to read VTK
            temp_analyzer = RTAnalyzer("./temp", use_grid_optimization=True, no_titles=True)
            vtk_data = temp_analyzer.read_vtk_file(first_vtk)
            
            # Extract domain dimensions
            if not vtk_data.get('is_cell_data', False):
                # For cell-centered data, use cell centers
                H = np.max(vtk_data['y']) - np.min(vtk_data['y'])
                L = np.max(vtk_data['x']) - np.min(vtk_data['x'])
            else:
                # For node-centered data, adjust for grid spacing
                dy = vtk_data['y'][0, 1] - vtk_data['y'][0, 0] if vtk_data['y'].shape[1] > 1 else 0
                dx = vtk_data['x'][1, 0] - vtk_data['x'][0, 0] if vtk_data['x'].shape[0] > 1 else 0
                H = np.max(vtk_data['y']) - np.min(vtk_data['y']) + dy  # Add one cell
                L = np.max(vtk_data['x']) - np.min(vtk_data['x']) + dx  # Add one cell

            print(f"   ✅ Auto-detected: H = {H:.4f} m, L = {L:.4f} m")
            return H, L
            
        except Exception as e:
            print(f"   ⚠️  Failed to read {data_dir}: {str(e)}")
            continue
    
    # If we get here, couldn't auto-detect
    raise RuntimeError("Could not auto-detect domain dimensions from any VTK file. Check data directories.")

def calculate_dimensionless_time_factor(A, g, H):
    """
    Calculate the dimensionless time factor √(Ag/H).

    Thin wrapper around RTPhysics for backward compatibility.

    Args:
        A: Atwood number
        g: Gravitational acceleration (m/s²)
        H: Domain height (m)

    Returns:
        float: √(Ag/H) factor for dimensionless time τ
    """
    from core.rt_physics import RTPhysics
    # Use RTPhysics internally (L is not needed for time factor, use dummy)
    rt = RTPhysics(A=A, g=g, H=H, L=1.0)
    factor = rt.tau_factor
    print(f"📊 Dimensionless time factor √(Ag/H) = √({A:.3e} × {g:.2f} / {H:.4f}) = {factor:.4f} s⁻¹")
    return factor

def convert_to_dimensionless_time(times, tau_factor):
    """
    Convert simulation times to dimensionless time τ.

    Args:
        times: Array of simulation times (seconds)
        tau_factor: √(Ag/H) factor

    Returns:
        array: Dimensionless times τ = √(Ag/H) × t
    """
    return np.array(times) * tau_factor

def find_timestep_files_for_resolution(data_dir, resolution_str, target_times=None, time_tolerance=0.5):
    """
    Find VTK files for a given resolution, supporting rectangular grids.
    
    Args:
        data_dir: Directory containing VTK files
        resolution_str: Resolution string (e.g., "200", "160x200")
        target_times: List of target times (None = find all files)
        time_tolerance: Maximum time difference allowed
        
    Returns:
        dict: {target_time: (vtk_file_path, actual_time)} or {actual_time: vtk_file_path}
    """
    # Parse resolution
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    is_rectangular = (nx != ny)
    
    # Try multiple filename patterns
    patterns = [
        os.path.join(data_dir, f"RT{grid_resolution_str}-*.vtk"),  # RT160x200-*.vtk
        os.path.join(data_dir, f"RT{nx}x{ny}-*.vtk"),              # Alternative format
        os.path.join(data_dir, f"{grid_resolution_str}-*.vtk"),    # 160x200-*.vtk
        os.path.join(data_dir, f"{nx}x{ny}-*.vtk"),                # Alternative format
    ]
    
    # For backward compatibility with square grids
    if not is_rectangular:
        patterns.extend([
            os.path.join(data_dir, f"RT{nx}-*.vtk"),  # Legacy: RT200-*.vtk
            os.path.join(data_dir, f"{nx}-*.vtk")     # Legacy: 200-*.vtk
        ])
    
    # Find files using first matching pattern
    vtk_files = []
    for pattern in patterns:
        found_files = glob.glob(pattern)
        if found_files:
            vtk_files = found_files
            break
    
    if not vtk_files:
        return {}
    
    # Extract times from filenames
    file_time_map = {}
    for vtk_file in vtk_files:
        try:
            basename = os.path.basename(vtk_file)
            if '-' in basename:
                time_str = basename.split('-')[1].split('.')[0]
            else:
                # Fallback: extract number from filename
                import re
                match = re.search(r'(\d+)\.vtk$', basename)
                time_str = match.group(1) if match else "0"
            
            file_time = float(time_str) / 1000.0
            file_time_map[file_time] = vtk_file
        except:
            continue
    
    if target_times is None:
        # Return all files found
        return file_time_map
    
    # Find best matches for target times
    result = {}
    for target_time in target_times:
        best_file = None
        best_diff = float('inf')
        best_time = None
        
        for file_time, vtk_file in file_time_map.items():
            diff = abs(file_time - target_time)
            if diff < best_diff:
                best_diff = diff
                best_file = vtk_file
                best_time = file_time
        
        if best_file and best_diff <= time_tolerance:
            result[target_time] = (best_file, best_time)
    
    return result

def determine_analysis_mode(resolutions, target_times):
    """
    Determine the optimal analysis mode based on inputs.
    
    Args:
        resolutions: List of resolution strings
        target_times: List of target times
        
    Returns:
        str: 'temporal_evolution', 'convergence_study', 'multi_time_convergence', or 'matrix_analysis'
    """
    num_resolutions = len(resolutions)
    num_times = len(target_times)
    
    print(f"🔍 Enhanced mode detection: {num_resolutions} resolutions × {num_times} times")

    if num_resolutions == 1 and num_times > 1:
        print(f"   → Detected: Temporal Evolution (single resolution, multiple times)")
        return 'temporal_evolution'  # ✅ FIXED!
    elif num_resolutions > 1 and num_times == 1:
        print(f"   → Detected: Convergence Study (multiple resolutions, single time)")
        return 'convergence_study'
    elif num_resolutions > 1 and num_times > 1:
        print(f"   → Detected: Multi-Time Convergence (convergence study at each time)")
        return 'multi_time_convergence'
    else:
        print(f"   → Detected: Matrix Analysis (fallback for edge cases)")
        return 'matrix_analysis'

def get_method_info(analysis_params):
    """Get extraction method information for naming and display."""
    if analysis_params.get('use_plic', False):
        return 'PLIC', '_plic', 'PLIC (theoretical reconstruction)'
    elif analysis_params.get('use_conrec', False):
        return 'CONREC', '_conrec', 'CONREC (precision)'
    else:
        return 'scikit-image', '_skimage', 'scikit-image (standard)'

def create_output_directory_name(mode, resolutions, target_times, method_suffix, multifractal_enabled=False, physics_suffix=""):
    """UPDATED: Create descriptive output directory name with optional physics suffix."""
    # Add multifractal suffix if enabled
    mf_suffix = "_mf" if multifractal_enabled else ""
    
    if mode == 'temporal_evolution':
        if len(resolutions) == 1:
            res_str = resolutions[0]
            nx, ny = parse_grid_resolution(res_str)
            grid_str = format_grid_resolution(nx, ny)
            time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
            return f"temporal_evolution_{grid_str}_{time_range}{method_suffix}{mf_suffix}{physics_suffix}"
        else:
            time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
            return f"temporal_evolution_multi_res_{time_range}{method_suffix}{mf_suffix}{physics_suffix}"
    
    elif mode == 'convergence_study':
        time_str = f"t{target_times[0]:.1f}"
        return f"convergence_study_{time_str}{method_suffix}{mf_suffix}{physics_suffix}"
    
    elif mode == 'multi_time_convergence':
        time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}"
        return f"multi_time_convergence_{time_range}{method_suffix}{mf_suffix}{physics_suffix}"
    
    elif mode == 'matrix_analysis':
        time_range = f"t{min(target_times):.1f}-{max(target_times):.1f}"
        res_range = f"{len(resolutions)}res"
        return f"matrix_analysis_{res_range}_{time_range}{method_suffix}{mf_suffix}{physics_suffix}"
    
    else:
        return f"hybrid_analysis{method_suffix}{mf_suffix}{physics_suffix}"

def validate_inputs(data_dirs, resolutions, target_times):
    """
    Validate input parameters for consistency.
    
    Args:
        data_dirs: List of data directories
        resolutions: List of resolution strings
        target_times: List of target times
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check data directories
    if len(data_dirs) != len(resolutions):
        return False, f"Number of data directories ({len(data_dirs)}) must match number of resolutions ({len(resolutions)})"
    
    for i, data_dir in enumerate(data_dirs):
        if not os.path.exists(data_dir):
            return False, f"Data directory not found: {data_dir}"
    
    # Validate resolution formats
    try:
        for res_str in resolutions:
            nx, ny = parse_grid_resolution(res_str)
            if nx <= 0 or ny <= 0:
                return False, f"Invalid resolution: {res_str}"
    except ValueError as e:
        return False, str(e)
    
    # Validate target times
    if not target_times:
        return False, "At least one target time must be specified"
    
    for t in target_times:
        if not isinstance(t, (int, float)) or t < 0:
            return False, f"Invalid target time: {t}"
    
    return True, None

def analyze_grid_types(resolutions):
    """
    Analyze the types of grids in the resolution list.
    
    Returns:
        dict: Information about grid types
    """
    square_count = 0
    rectangular_count = 0
    aspect_ratios = []
    
    for res_str in resolutions:
        nx, ny = parse_grid_resolution(res_str)
        if nx == ny:
            square_count += 1
        else:
            rectangular_count += 1
            aspect_ratios.append(max(nx, ny) / min(nx, ny))
    
    return {
        'total': len(resolutions),
        'square_count': square_count,
        'rectangular_count': rectangular_count,
        'aspect_ratios': aspect_ratios,
        'max_aspect_ratio': max(aspect_ratios) if aspect_ratios else 1.0,
        'has_mixed_types': square_count > 0 and rectangular_count > 0
    }

def print_analysis_header(mode, resolutions, target_times, data_dirs, analysis_params, num_processes, H, L, tau_factor):
    """UPDATED: Print comprehensive analysis header with physics parameters."""
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    grid_info = analyze_grid_types(resolutions)
    
    print(f"🚀 ENHANCED HYBRID PARALLEL RESOLUTION ANALYZER")
    if analysis_params.get('enable_multifractal', False):
        print(f"🔬 WITH MULTIFRACTAL ANALYSIS")
    print(f"🧮 WITH DIMENSIONLESS PHYSICS PLOTS")
    print(f"=" * 70)
    print(f"Analysis mode: {mode.replace('_', ' ').title()}")
    print(f"Interface extraction: {method_description}")
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    print(f"Mixing method: {mixing_method}")
    print(f"Parallel processes: {num_processes}")
    if analysis_params.get('enable_multifractal', False):
        q_values = analysis_params.get('q_values', 'default (-5 to 5)')
        print(f"Multifractal q-values: {q_values}")
    
    print(f"\n📊 ANALYSIS SCOPE:")
    print(f"Resolutions ({len(resolutions)}): {resolutions}")
    print(f"Target times ({len(target_times)}): {target_times}")
    print(f"Data directories: {len(data_dirs)}")
    
    print(f"\n📐 DOMAIN & PHYSICS:")
    print(f"Domain height (H): {H:.4f} m (auto-detected)")
    print(f"Domain width (L): {L:.4f} m (auto-detected)")
    print(f"Atwood number (A): {analysis_params.get('atwood_number'):.3e}")
    print(f"Gravity (g): {analysis_params.get('gravity'):.2f} m/s²")
    print(f"Dimensionless factor √(Ag/H): {tau_factor:.4f} s⁻¹")
    
    print(f"\n📈 GRID ANALYSIS:")
    if grid_info['has_mixed_types']:
        print(f"Mixed grid types: {grid_info['square_count']} square, {grid_info['rectangular_count']} rectangular")
    elif grid_info['rectangular_count'] > 0:
        print(f"All rectangular grids (max aspect ratio: {grid_info['max_aspect_ratio']:.2f})")
    else:
        print(f"All square grids")
    
    total_analyses = len(resolutions) * len(target_times)
    print(f"\n⚡ PARALLEL EXECUTION:")
    print(f"Total analyses: {total_analyses} ({len(resolutions)} resolutions × {len(target_times)} times)")
    
    if mode == 'temporal_evolution':
        print(f"Strategy: Smart batching (each worker processes all times for one resolution)")
    elif mode == 'convergence_study':
        print(f"Strategy: Resolution parallel (each worker processes one resolution)")
    elif mode == 'multi_time_convergence':
        print(f"Strategy: Multi-time convergence (convergence analysis at each time point)")
        print(f"Expected outputs: {len(target_times)} convergence plots + 1 evolution summary + dimensionless physics plots")
    else:
        print(f"Strategy: Matrix parallel (adaptive based on workload)")

# Enhanced Analyzer Block 2: Result Processing & Analysis Functions

def create_base_result_dict(resolution_str, target_time, extraction_method, worker_pid):
    """Create base result dictionary with common fields."""
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)
    
    return {
        'resolution_str': resolution_str,
        'nx': nx,
        'ny': ny,
        'grid_resolution': grid_resolution_str,
        'is_rectangular': (nx != ny),
        'aspect_ratio': max(nx, ny) / min(nx, ny),
        'effective_resolution': max(nx, ny),
        'total_cells': nx * ny,
        'target_time': target_time,
        'extraction_method': extraction_method,
        'worker_pid': worker_pid
    }

def create_success_result(base_result, vtk_analysis_result, vtk_file, actual_time, 
                         processing_time, segments_count, h0, analysis_params):
    """
    UPDATED: Create successful analysis result dictionary compatible with new four-method framework.
    Now handles the updated field names from rt_analyzer.py
    """
    result = base_result.copy()

    if analysis_params.get('debug', False):
        print(f"DEBUG: All vtk_analysis_result fields: {sorted(vtk_analysis_result.keys())}")

    # Extract mixing heights using new field names
    # Mass-conservation method (primary in comprehensive)
    h_01 = vtk_analysis_result.get('h_01', np.nan)  # Bottom fluid above y₀
    h_10 = vtk_analysis_result.get('h_10', np.nan)  # Top fluid below y₀
    h_total_mass = vtk_analysis_result.get('h_total_mass', np.nan)
    
    # Geometric penetration method 
    h_penetration_up = vtk_analysis_result.get('h_penetration_up', np.nan)
    h_penetration_down = vtk_analysis_result.get('h_penetration_down', np.nan)
    h_total_geometric = vtk_analysis_result.get('h_total_geometric', np.nan)
    
    # Youngs integral width method
    W = vtk_analysis_result.get('W', vtk_analysis_result.get('integral_width', np.nan))
    mixing_centroid = vtk_analysis_result.get('mixing_centroid', np.nan)
    
    # Global mixedness diagnostic
    mixing_ratio = vtk_analysis_result.get('mixing_ratio', np.nan)
    mixing_efficiency = vtk_analysis_result.get('mixing_efficiency', np.nan)
    total_mixing_integral = vtk_analysis_result.get('total_mixing_integral', np.nan)

    # Legacy compatibility - use mass-conservation as primary for backward compatibility
    ht = h_01 if not np.isnan(h_01) else vtk_analysis_result.get('ht', np.nan)
    hb = h_10 if not np.isnan(h_10) else vtk_analysis_result.get('hb', np.nan)
    h_total = h_total_mass if not np.isnan(h_total_mass) else vtk_analysis_result.get('h_total', ht + hb if not (np.isnan(ht) or np.isnan(hb)) else np.nan)

    result.update({
        'actual_time': actual_time,
        'time_error': abs(actual_time - base_result['target_time']),
        'fractal_dim': vtk_analysis_result.get('fractal_dimension', np.nan),
        'fd_error': vtk_analysis_result.get('fractal_error', np.nan),
        'fd_r_squared': vtk_analysis_result.get('fractal_r_squared', np.nan),
        
        # Legacy fields (for backward compatibility)
        'ht': ht,
        'hb': hb,
        'h_total': h_total,
        'h0': h0,
        
        # NEW: Mass-conservation method results (your theoretical framework)
        'h_01': h_01,                                                   # Bottom fluid above y₀
        'h_10': h_10,                                                   # Top fluid below y₀
        'h_total_mass': h_total_mass,                                   # Total mass-based thickness
        
        # NEW: Geometric penetration results (Dalziel threshold method)
        'h_penetration_up': h_penetration_up,
        'h_penetration_down': h_penetration_down,
        'h_total_geometric': h_total_geometric,
        'y_upper_boundary': vtk_analysis_result.get('y_upper_boundary'),
        'y_lower_boundary': vtk_analysis_result.get('y_lower_boundary'),
        'upper_threshold': vtk_analysis_result.get('upper_threshold', 0.95),
        'lower_threshold': vtk_analysis_result.get('lower_threshold', 0.05),
        
        # NEW: Youngs integral width results (robust method)
        'W': W,                                                         # Youngs integral width
        'integral_width': W,                                            # Alias
        'mixing_centroid': mixing_centroid,
        'max_mixing_density_1d': vtk_analysis_result.get('max_mixing_density_1d', np.nan),
        
        # NEW: Global mixedness diagnostic (your mixing quality measure)
        'mixing_ratio': mixing_ratio,                                   # Your key metric
        'mixing_efficiency': mixing_efficiency,
        'total_mixing_integral': total_mixing_integral,
        'domain_area': vtk_analysis_result.get('domain_area', np.nan),
        
        # Analysis metadata
        'segments': segments_count,
        'processing_time': processing_time,
        'vtk_file': os.path.basename(vtk_file),
        'analysis_quality': vtk_analysis_result.get('analysis_quality', 'unknown'),
        'status': 'success',
        'y0': vtk_analysis_result.get('y0', h0)                         # Initial interface position
    })

    # Handle mixing zone center - use new field names with fallbacks
    if not np.isnan(mixing_centroid):
        # Prefer Youngs mixing centroid (most robust)
        result['y_center'] = mixing_centroid
        if analysis_params.get('debug', False):
            print(f"DEBUG: Using Youngs mixing centroid: {result['y_center']:.6f}")
    elif 'y_upper_boundary' in vtk_analysis_result and 'y_lower_boundary' in vtk_analysis_result:
        # Use geometric boundaries from threshold method
        y_upper = vtk_analysis_result['y_upper_boundary']
        y_lower = vtk_analysis_result['y_lower_boundary']
        if y_upper is not None and y_lower is not None:
            result['y_center'] = (y_upper + y_lower) / 2
            if analysis_params.get('debug', False):
                print(f"DEBUG: Calculated y_center from geometric boundaries: {result['y_center']:.6f}")
        else:
            result['y_center'] = vtk_analysis_result.get('y0', h0)  # Fallback to initial interface
    else:
        # Final fallback calculation using available heights
        y0_val = vtk_analysis_result.get('y0', h0)
        ht_val = result.get('ht', 0.0)
        hb_val = result.get('hb', 0.0)
        result['y_center'] = y0_val + (ht_val - hb_val) / 2
        if analysis_params.get('debug', False):
            print(f"WARNING: y_center calculated using fallback method: {result['y_center']:.6f}")

    # UPDATED: Physics validation using new geometric method fields
    if 'y_upper_boundary' in vtk_analysis_result and 'y_lower_boundary' in vtk_analysis_result:
        y_upper = vtk_analysis_result.get('y_upper_boundary')
        y_lower = vtk_analysis_result.get('y_lower_boundary')
        
        if y_upper is not None and y_lower is not None:
            mixing_zone_width = y_upper - y_lower
            result['mixing_zone_width'] = mixing_zone_width
            result['physics_ratio'] = mixing_zone_width  # Store width as physics measure
            
            if mixing_zone_width > 0:
                result['physics_validation'] = 'expected'
                if analysis_params.get('debug', False):
                    print(f"DEBUG: ✅ Physics as expected: mixing zone width = {mixing_zone_width:.6f}")
            else:
                result['physics_validation'] = 'unexpected'
                if analysis_params.get('debug', False):
                    print(f"DEBUG: ⚠️ Physics unexpected: mixing zone width = {mixing_zone_width:.6f}")
        else:
            result['physics_ratio'] = np.nan
            result['physics_validation'] = 'unavailable'
    else:
        # Fallback physics validation using penetration distances or basic h_total
        if not np.isnan(h_penetration_up) and not np.isnan(h_penetration_down):
            total_penetration = h_penetration_up + h_penetration_down
            result['physics_ratio'] = total_penetration
            result['physics_validation'] = 'expected' if total_penetration > 0 else 'unavailable'
        elif not np.isnan(h_total):
            result['physics_ratio'] = h_total
            result['physics_validation'] = 'expected' if h_total > 0 else 'unavailable'
        else:
            result['physics_ratio'] = np.nan
            result['physics_validation'] = 'unavailable'

    # Handle mixing fraction field (from mixing_efficiency)
    if not np.isnan(mixing_efficiency):
        result['mixing_fraction'] = mixing_efficiency
    else:
        result['mixing_fraction'] = 0.0  # Default fallback

    # Add multifractal results if available
    if 'multifractal' in vtk_analysis_result and vtk_analysis_result['multifractal']:
        mf_results = vtk_analysis_result['multifractal']
        result.update({
            'mf_D0': mf_results.get('D0', np.nan),
            'mf_D1': mf_results.get('D1', np.nan),
            'mf_D2': mf_results.get('D2', np.nan),
            'mf_alpha_width': mf_results.get('alpha_width', np.nan),
            'mf_degree_multifractality': mf_results.get('degree_multifractality', np.nan),
            'mf_status': 'success'
        })
        if analysis_params.get('debug', False):
            print(f"DEBUG: Added multifractal results: D0={result['mf_D0']:.4f}, D1={result['mf_D1']:.4f}, D2={result['mf_D2']:.4f}")
    else:
        # Add NaN placeholders for multifractal fields
        result.update({
            'mf_D0': np.nan,
            'mf_D1': np.nan,
            'mf_D2': np.nan,
            'mf_alpha_width': np.nan,
            'mf_degree_multifractality': np.nan,
            'mf_status': 'not_enabled'
        })

    # Add method-specific ratios for cross-method analysis
    if not np.isnan(W) and W > 0:
        if not np.isnan(h_total_mass):
            result['h_mass_over_W'] = h_total_mass / W
        if not np.isnan(h_total_geometric):
            result['h_geometric_over_W'] = h_total_geometric / W
        if not np.isnan(h_penetration_up):
            result['h_up_over_W'] = h_penetration_up / W
        if not np.isnan(h_penetration_down):
            result['h_down_over_W'] = h_penetration_down / W

    return result

def create_failure_result(base_result, error_message, vtk_file=None, actual_time=None, processing_time=None, h0=0.5):
    """
    UPDATED: Create failed analysis result dictionary with new field placeholders.
    """
    result = base_result.copy()
    result.update({
        'actual_time': actual_time if actual_time is not None else np.nan,
        'time_error': np.nan,
        'fractal_dim': np.nan,
        'fd_error': np.nan,
        'fd_r_squared': np.nan,
        
        # Legacy fields
        'h_total': np.nan,
        'ht': np.nan,
        'hb': np.nan,
        'h0': h0,
        
        # NEW: Four-method field placeholders
        'h_01': np.nan,
        'h_10': np.nan,
        'h_total_mass': np.nan,
        'h_penetration_up': np.nan,
        'h_penetration_down': np.nan,
        'h_total_geometric': np.nan,
        'W': np.nan,
        'integral_width': np.nan,
        'mixing_centroid': np.nan,
        'mixing_ratio': np.nan,
        'mixing_efficiency': np.nan,
        'total_mixing_integral': np.nan,
        
        # Analysis metadata
        'segments': np.nan,
        'processing_time': processing_time if processing_time is not None else np.nan,
        'vtk_file': os.path.basename(vtk_file) if vtk_file else 'not_found',
        'analysis_quality': 'failed',
        'status': 'failed',
        'error': error_message,
        'physics_ratio': np.nan,
        'physics_validation': 'failed',
        'y_center': np.nan,
        'y0': h0,
        
        # Multifractal field placeholders
        'mf_D0': np.nan,
        'mf_D1': np.nan,
        'mf_D2': np.nan,
        'mf_alpha_width': np.nan,
        'mf_degree_multifractality': np.nan,
        'mf_status': 'failed'
    })
    return result

def analyze_single_file(vtk_file, analyzer, analysis_params):
    """
    Analyze a single VTK file using the analyzer.
    UPDATED: Now uses correct mixing method names compatible with rt_analyzer.py
    """
    start_time = time.time()

    try:
        # Determine analysis types
        analysis_types = ['fractal_dim', 'mixing']
        enable_multifractal = analysis_params.get('enable_multifractal', False)
        
        # Get the correct mixing method for rt_analyzer
        mixing_method = analysis_params.get('mixing_method', 'comprehensive')
        
        print(f"🎯 Using initial interface height h0 = {analysis_params.get('h0', 0.5)}")
        if enable_multifractal:
            print(f"🔬 Multifractal analysis enabled")
        
        result = analyzer.analyze_vtk_file(
            vtk_file,
            analysis_types=analysis_types,
            y0=analysis_params.get('h0', 0.5),
            mixing_method=mixing_method,  # Now uses correct method name
            min_box_size=analysis_params.get('min_box_size', None),
            enable_multifractal=enable_multifractal,
            q_values=analysis_params.get('q_values', None),
            mf_output_dir=analysis_params.get('mf_output_dir', None)
        )
        
        # Debug output for field checking
        if analysis_params.get('debug', False):
            print(f"DEBUG: RT analyzer returned fields: {list(result.keys())}")

        # Check for multifractal results
        if enable_multifractal:
            if analysis_params.get('debug', False):
                print(f"DEBUG: Checking for multifractal results:")
                print(f"  'multifractal' in result: {'multifractal' in result}")
                if 'multifractal' in result and result['multifractal']:
                    mf = result['multifractal']
                    print(f"  Multifractal D0: {mf.get('D0', 'N/A')}")
                    print(f"  Multifractal D1: {mf.get('D1', 'N/A')}")
                    print(f"  Multifractal D2: {mf.get('D2', 'N/A')}")

        # Extract interface for segment count
        data = analyzer.read_vtk_file(vtk_file)
        contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
        segments = analyzer.convert_contours_to_segments(contours)

        processing_time = time.time() - start_time
        return result, len(segments), processing_time

    except Exception as e:
        processing_time = time.time() - start_time
        raise Exception(f"Analysis failed: {str(e)}")

def analyze_temporal_evolution_batch(args):
    """
    Analyze temporal evolution for one resolution across multiple times.
    UPDATED: Now compatible with new rt_analyzer mixing methods.
    """
    data_dir, resolution_str, target_times, base_output_dir, analysis_params = args

    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)

    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"({method_name}, {mixing_method})"
    
    mf_str = " with multifractal" if enable_multifractal else ""
    print(f"🔍 Worker {os.getpid()}: Temporal evolution {grid_resolution_str} for {len(target_times)} times {method_info}{mf_str}")

    # Create analyzer for this resolution (reuse for efficiency)
    res_output_dir = os.path.join(base_output_dir, f"temporal_evolution_{grid_resolution_str}")
    analyzer = RTAnalyzer(
        res_output_dir,
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )

    # Find all available files for this resolution
    file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, target_times)

    batch_results = []
    worker_start = time.time()

    for i, target_time in enumerate(target_times):
        print(f"   [{i+1}/{len(target_times)}] {grid_resolution_str} at t={target_time}")

        base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())

        if target_time not in file_time_map:
            print(f"   No file found for t={target_time}")
            failure_result = create_failure_result(base_result, "No file found for target time", h0=analysis_params.get('h0', 0.5))
            batch_results.append(failure_result)
            continue

        vtk_file, actual_time = file_time_map[target_time]

        try:
            vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)

            success_result = create_success_result(
                base_result, vtk_result, vtk_file, actual_time, processing_time, 
                segments_count, analysis_params.get('h0', 0.5), analysis_params
            )

            # Enhanced output with physics validation as informational
            physics_info = ""
            if 'physics_validation' in success_result:
                if success_result['physics_validation'] == 'expected':
                    physics_info = f", ✅ Physics as expected"
                elif success_result['physics_validation'] == 'unexpected':
                    physics_info = f", ⚠️ Physics unexpected"

            if enable_multifractal and success_result['mf_status'] == 'success':
                print(f"   ✅ t={target_time:.1f}: D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
                      f"MF D0={success_result['mf_D0']:.4f}, Segments={segments_count}, Time={processing_time:.1f}s{physics_info}")
            else:
                print(f"   ✅ t={target_time:.1f}: D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
                      f"Segments={segments_count}, Time={processing_time:.1f}s{physics_info}")

            batch_results.append(success_result)

        except Exception as e:
            print(f"   ❌ t={target_time}: {str(e)}")
            failure_result = create_failure_result(base_result, str(e), vtk_file, actual_time, h0=analysis_params.get('h0', 0.5))
            batch_results.append(failure_result)

    worker_time = time.time() - worker_start
    successful_count = sum(1 for r in batch_results if r['status'] == 'success')

    print(f"✅ Worker {os.getpid()}: {grid_resolution_str} temporal evolution complete - "
          f"{successful_count}/{len(target_times)} successful in {worker_time:.1f}s {method_info}{mf_str}")

    return batch_results

def analyze_convergence_single_resolution(args):
    """
    Analyze single resolution for convergence study.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    data_dir, resolution_str, target_time, base_output_dir, analysis_params = args

    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)

    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"({method_name}, {mixing_method})"
        
    mf_str = " with multifractal" if enable_multifractal else ""
    print(f"🔍 Worker {os.getpid()}: Convergence analysis {grid_resolution_str} at t={target_time} {method_info}{mf_str}")

    base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())

    # Create analyzer for this resolution
    res_output_dir = os.path.join(base_output_dir, f"convergence_{grid_resolution_str}")
    analyzer = RTAnalyzer(
        res_output_dir,
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )

    try:
        # Find file for target time
        file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, [target_time])

        if target_time not in file_time_map:
            print(f"   No file found for t={target_time}")
            return create_failure_result(base_result, "No file found for target time", h0=analysis_params.get('h0', 0.5))

        vtk_file, actual_time = file_time_map[target_time]

        # Perform analysis
        vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)

        success_result = create_success_result(
            base_result, vtk_result, vtk_file, actual_time, processing_time, 
            segments_count, analysis_params.get('h0', 0.5), analysis_params
        )

        # Enhanced output with physics validation as informational
        physics_info = ""
        if 'physics_validation' in success_result:
            if success_result['physics_validation'] == 'expected':
                physics_info = f", ✅ Physics as expected"
            elif success_result['physics_validation'] == 'unexpected':
                physics_info = f", ⚠️ Physics unexpected"

        if enable_multifractal and success_result['mf_status'] == 'success':
            print(f"✅ Worker {os.getpid()}: {grid_resolution_str} D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
                  f"MF D0={success_result['mf_D0']:.4f}, Segments={segments_count}, Time={processing_time:.1f}s {method_info}{physics_info}")
        else:
            print(f"✅ Worker {os.getpid()}: {grid_resolution_str} D={success_result['fractal_dim']:.4f}±{success_result['fd_error']:.4f}, "
                  f"Segments={segments_count}, Time={processing_time:.1f}s {method_info}{physics_info}")

        return success_result

    except Exception as e:
        print(f"❌ Worker {os.getpid()}: {grid_resolution_str} failed - {str(e)}")
        return create_failure_result(base_result, str(e), h0=analysis_params.get('h0', 0.5))

def analyze_matrix_single_point(args):
    """
    Analyze single (resolution, time) point for matrix analysis.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    data_dir, resolution_str, target_time, base_output_dir, analysis_params = args

    method_name, method_suffix, method_description = get_method_info(analysis_params)
    nx, ny = parse_grid_resolution(resolution_str)
    grid_resolution_str = format_grid_resolution(nx, ny)

    base_result = create_base_result_dict(resolution_str, target_time, method_name, os.getpid())

    # Create analyzer for this point
    point_output_dir = os.path.join(base_output_dir, f"matrix_{grid_resolution_str}_t{target_time:.1f}")
    analyzer = RTAnalyzer(
        point_output_dir,
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=analysis_params.get('use_conrec', False),
        use_plic=analysis_params.get('use_plic', False),
        debug=analysis_params.get('debug', False)
    )

    try:
        # Find file for target time
        file_time_map = find_timestep_files_for_resolution(data_dir, resolution_str, [target_time])

        if target_time not in file_time_map:
            return create_failure_result(base_result, "No file found for target time", h0=analysis_params.get('h0', 0.5))

        vtk_file, actual_time = file_time_map[target_time]

        # Perform analysis
        vtk_result, segments_count, processing_time = analyze_single_file(vtk_file, analyzer, analysis_params)

        success_result = create_success_result(
            base_result, vtk_result, vtk_file, actual_time, processing_time, 
            segments_count, analysis_params.get('h0', 0.5), analysis_params
        )

        return success_result

    except Exception as e:
        return create_failure_result(base_result, str(e), h0=analysis_params.get('h0', 0.5))

# Enhanced Analyzer Block 3: Main Analysis Execution Functions

def run_multi_time_convergence_analysis(data_dirs, resolutions, target_times, output_dir,
                                       analysis_params, num_processes):
    """
    Run convergence analysis at multiple time points.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"with {mixing_method}"
    mf_str = " with multifractal" if enable_multifractal else ""
    
    print(f"\n⚡ MULTI-TIME CONVERGENCE ANALYSIS{mf_str.upper()}")
    print(f"Strategy: Convergence study at each of {len(target_times)} time points {method_info}")
    if enable_multifractal:
        print(f"Multifractal: Enabled for all analyses")
    print(f"Will create {len(target_times)} individual convergence plots + evolution summary")

    all_results = []
    total_start = time.time()

    # Analyze convergence at each time point
    for i, target_time in enumerate(target_times):
        print(f"\n📊 [{i+1}/{len(target_times)}] Convergence analysis at t={target_time}")

        # Run convergence analysis for this specific time
        try:
            time_results, time_duration = run_convergence_analysis(
                data_dirs, resolutions, target_time, output_dir, analysis_params, num_processes
            )

            if time_results:
                # Add time point metadata to each result
                for result in time_results:
                    result['convergence_time_point'] = target_time
                    result['convergence_analysis_id'] = i
                    result['analysis_type'] = 'convergence_at_time'

                all_results.extend(time_results)

                # Report success with physics validation
                successful_count = sum(1 for r in time_results if r.get('status') == 'success')
                expected_physics_count = sum(1 for r in time_results if r.get('physics_validation') == 'expected')
                
                if enable_multifractal:
                    mf_successful = sum(1 for r in time_results if r.get('mf_status') == 'success')
                    print(f"✅ t={target_time} convergence complete: {successful_count}/{len(time_results)} successful, "
                          f"{mf_successful} with multifractal, {expected_physics_count} physics as expected in {time_duration:.1f}s")
                else:
                    print(f"✅ t={target_time} convergence complete: {successful_count}/{len(time_results)} successful, "
                          f"{expected_physics_count} physics as expected in {time_duration:.1f}s")
            else:
                print(f"❌ t={target_time} convergence returned no results")

        except Exception as e:
            print(f"❌ t={target_time} convergence failed: {str(e)}")

            # Add failure entries for this time point
            for j, resolution_str in enumerate(resolutions):
                failure_result = create_failure_result(
                    create_base_result_dict(resolution_str, target_time,
                                          get_method_info(analysis_params)[0], os.getpid()),
                    f"Convergence analysis failed: {str(e)}"
                )
                failure_result['convergence_time_point'] = target_time
                failure_result['convergence_analysis_id'] = i
                all_results.append(failure_result)

    total_time = time.time() - total_start

    # Summary with physics validation
    successful_results = [r for r in all_results if r.get('status') == 'success']
    expected_physics_results = [r for r in successful_results if r.get('physics_validation') == 'expected']
    
    print(f"\n📊 MULTI-TIME CONVERGENCE SUMMARY:")
    print(f"   Total time points analyzed: {len(target_times)}")
    print(f"   Total analyses attempted: {len(all_results)}")
    print(f"   Successful analyses: {len(successful_results)}")
    print(f"   Physics as expected: {len(expected_physics_results)}")
    if enable_multifractal:
        mf_successful = sum(1 for r in all_results if r.get('mf_status') == 'success')
        print(f"   Successful multifractal analyses: {mf_successful}")
    print(f"   Total processing time: {total_time:.1f}s")
    print(f"   Expected plots: {len(target_times)} convergence + 1 evolution summary + dimensionless physics plots")

    return all_results, total_time

def run_temporal_evolution_analysis(data_dirs, resolutions, target_times, output_dir,
                                  analysis_params, num_processes):
    """
    Run temporal evolution analysis using smart batching.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"with {mixing_method}"
    mf_str = " with multifractal" if enable_multifractal else ""
    
    print(f"\n⚡ TEMPORAL EVOLUTION ANALYSIS{mf_str.upper()}")
    print(f"Strategy: Smart batching (reuse analyzer per resolution) {method_info}")
    if enable_multifractal:
        print(f"Multifractal: Enabled for all analyses")

    # Prepare arguments for parallel processing
    process_args = [(data_dir, resolution_str, target_times, output_dir, analysis_params)
                   for data_dir, resolution_str in zip(data_dirs, resolutions)]

    total_start = time.time()

    try:
        with Pool(processes=num_processes) as pool:
            batch_results = pool.map(analyze_temporal_evolution_batch, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None, 0
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None, 0

    total_time = time.time() - total_start

    # Flatten results
    all_results = []
    for batch in batch_results:
        all_results.extend(batch)

    return all_results, total_time

def run_convergence_analysis(data_dirs, resolutions, target_time, output_dir,
                            analysis_params, num_processes):
    """
    Run convergence analysis with one worker per resolution.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"with {mixing_method}"
    mf_str = " with multifractal" if enable_multifractal else ""
    
    print(f"\n⚡ CONVERGENCE ANALYSIS{mf_str.upper()}")
    print(f"Strategy: Resolution parallel (one worker per resolution) {method_info}")
    if enable_multifractal:
        print(f"Multifractal: Enabled for all analyses")

    # Prepare arguments for parallel processing
    process_args = [(data_dir, resolution_str, target_time, output_dir, analysis_params)
                   for data_dir, resolution_str in zip(data_dirs, resolutions)]

    total_start = time.time()

    try:
        with Pool(processes=num_processes) as pool:
            results = pool.map(analyze_convergence_single_resolution, process_args)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None, 0
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None, 0

    total_time = time.time() - total_start

    return results, total_time

def run_matrix_analysis(data_dirs, resolutions, target_times, output_dir,
                       analysis_params, num_processes):
    """
    Run matrix analysis with adaptive parallelization.
    UPDATED: Compatible with new rt_analyzer mixing methods.
    """
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    method_info = f"with {mixing_method}"
    mf_str = " with multifractal" if enable_multifractal else ""
    
    print(f"\n⚡ MATRIX ANALYSIS{mf_str.upper()}")
    print(f"Strategy: Matrix parallel (distribute all combinations) {method_info}")
    if enable_multifractal:
        print(f"Multifractal: Enabled for all analyses")

    # Create all (resolution, time) combinations
    process_args = []
    for data_dir, resolution_str in zip(data_dirs, resolutions):
        for target_time in target_times:
            process_args.append((data_dir, resolution_str, target_time, output_dir, analysis_params))

    print(f"Total matrix points: {len(process_args)} ({len(resolutions)} × {len(target_times)})")

    total_start = time.time()

    try:
        with Pool(processes=num_processes) as pool:
            # Process all tasks in parallel
            results = pool.map(analyze_matrix_single_point, process_args)

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return None, 0
    except Exception as e:
        print(f"\n❌ Parallel execution failed: {str(e)}")
        return None, 0

    total_time = time.time() - total_start

    return results, total_time

def run_hybrid_analysis(data_dirs, resolutions, target_times, output_dir,
                   analysis_params, num_processes=None):
    """
    UPDATED: Main hybrid analysis function with physics parameters and domain auto-detection.

    Args:
        data_dirs: List of data directories
        resolutions: List of resolution strings
        target_times: List of target times
        output_dir: Output directory
        analysis_params: Analysis parameters dictionary (now includes A, g)
        num_processes: Number of parallel processes

    Returns:
        DataFrame with analysis results
    """
    # Validate inputs
    is_valid, error_msg = validate_inputs(data_dirs, resolutions, target_times)
    if not is_valid:
        print(f"❌ Input validation failed: {error_msg}")
        return None

    # NEW: Auto-detect domain dimensions
    try:
        H, L = auto_detect_domain_dimensions(data_dirs, resolutions)
        analysis_params['domain_height'] = H
        analysis_params['domain_width'] = L
    except Exception as e:
        print(f"❌ Failed to auto-detect domain dimensions: {str(e)}")
        return None

    # NEW: Calculate dimensionless time factor
    A = analysis_params.get('atwood_number', 2.1e-3)
    g = analysis_params.get('gravity', 9.8)
    tau_factor = calculate_dimensionless_time_factor(A, g, H)
    analysis_params['tau_factor'] = tau_factor

    # Determine analysis mode
    mode = determine_analysis_mode(resolutions, target_times)

    # Set optimal number of processes
    if num_processes is None:
        if mode == 'temporal_evolution':
            num_processes = min(len(resolutions), cpu_count())
        elif mode == 'convergence_study':
            num_processes = min(len(resolutions), cpu_count())
        elif mode == 'multi_time_convergence':
            num_processes = min(len(resolutions), cpu_count())
        else:  # matrix_analysis
            num_processes = min(cpu_count(), 8)  # Limit for I/O

    # Create output directory
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    
    # NEW: Add physics info to directory name
    physics_suffix = f"_A{A:.1e}_g{g:.1f}" if A != 2.1e-3 or g != 9.8 else ""
    
    if output_dir is None:
        output_dir = create_output_directory_name(mode, resolutions, target_times, method_suffix, enable_multifractal, physics_suffix)

    os.makedirs(output_dir, exist_ok=True)

    # Print analysis header with physics info
    print_analysis_header(mode, resolutions, target_times, data_dirs, analysis_params, num_processes, H, L, tau_factor)

    # Run appropriate analysis
    if mode == 'temporal_evolution':
        results, total_time = run_temporal_evolution_analysis(
            data_dirs, resolutions, target_times, output_dir, analysis_params, num_processes)
    elif mode == 'convergence_study':
        results, total_time = run_convergence_analysis(
            data_dirs, resolutions, target_times[0], output_dir, analysis_params, num_processes)
    elif mode == 'multi_time_convergence':
        results, total_time = run_multi_time_convergence_analysis(
            data_dirs, resolutions, target_times, output_dir, analysis_params, num_processes)
    else:  # matrix_analysis
        results, total_time = run_matrix_analysis(
            data_dirs, resolutions, target_times, output_dir, analysis_params, num_processes)

    if results is None:
        return None

    # Convert to DataFrame and add metadata
    df = pd.DataFrame(results)
    df['analysis_mode'] = mode

    # NEW: Add dimensionless time columns to DataFrame
    if 'actual_time' in df.columns:
        df['tau'] = df['actual_time'] * tau_factor  # Dimensionless time τ = √(Ag/H) × t
        
        # Add dimensionless mixing thicknesses if mixing data exists
        if 'ht' in df.columns and 'hb' in df.columns:
            df['ht_normalized'] = df['ht'] / H  # h_t/H
            df['hb_normalized'] = df['hb'] / H  # h_b/H
            df['h_total_normalized'] = df['h_total'] / H  # h_total/H
            
        # Add normalized versions for all four methods
        if 'h_01' in df.columns:
            df['h_01_normalized'] = df['h_01'] / H  # h₀₁/H (mass conservation)
        if 'h_10' in df.columns:
            df['h_10_normalized'] = df['h_10'] / H  # h₁₀/H (mass conservation)
        if 'h_total_mass' in df.columns:
            df['h_total_mass_normalized'] = df['h_total_mass'] / H  # Total mass/H
        if 'h_penetration_up' in df.columns:
            df['h_penetration_up_normalized'] = df['h_penetration_up'] / H  # h_up/H (geometric)
        if 'h_penetration_down' in df.columns:
            df['h_penetration_down_normalized'] = df['h_penetration_down'] / H  # h_down/H (geometric)
        if 'h_total_geometric' in df.columns:
            df['h_total_geometric_normalized'] = df['h_total_geometric'] / H  # Total geometric/H
        if 'W' in df.columns:
            df['W_normalized'] = df['W'] / H  # Youngs width/H

    # Save results
    time_range_str = f"t{min(target_times):.1f}-{max(target_times):.1f}" if len(target_times) > 1 else f"t{target_times[0]:.1f}"
    mf_suffix = "_mf" if enable_multifractal else ""
    results_file = os.path.join(output_dir, f'hybrid_analysis_{mode}_{time_range_str}{method_suffix}{mf_suffix}{physics_suffix}.csv')
    df.to_csv(results_file, index=False)

    # Print summary
    print_analysis_summary(df, mode, total_time, method_description, results_file, enable_multifractal, H, L, A, g, tau_factor)

    return df

# Enhanced Analyzer Block 4: Summary Functions & Dimensionless Plotting

def print_analysis_summary(df, mode, total_time, method_description, results_file, enable_multifractal=False, H=None, L=None, A=None, g=None, tau_factor=None):
    """UPDATED: Print comprehensive analysis summary with physics parameters."""
    print(f"\n📊 ENHANCED ANALYSIS SUMMARY")
    if enable_multifractal:
        print(f"🔬 WITH MULTIFRACTAL ANALYSIS")
    print(f"🧮 WITH DIMENSIONLESS PHYSICS PLOTS")
    print(f"=" * 70)
    print(f"Analysis mode: {mode.replace('_', ' ').title()}")
    print(f"Interface extraction: {method_description}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Results saved to: {results_file}")

    # Filter results
    successful_results = df[df['status'] == 'success']
    failed_results = df[df['status'] == 'failed']

    print(f"\nSuccessful analyses: {len(successful_results)}/{len(df)}")
    
    # Physics validation statistics (informational)
    if 'physics_validation' in df.columns:
        expected_physics = len(df[df['physics_validation'] == 'expected'])
        unexpected_physics = len(df[df['physics_validation'] == 'unexpected'])
        unavailable_physics = len(df[df['physics_validation'] == 'unavailable'])
        print(f"Physics validation: {expected_physics} as expected, {unexpected_physics} unexpected, {unavailable_physics} unavailable")
    
    # Multifractal success statistics
    if enable_multifractal and 'mf_status' in df.columns:
        mf_successful = len(df[df['mf_status'] == 'success'])
        print(f"Successful multifractal analyses: {mf_successful}/{len(df)}")
        
        if mf_successful > 0:
            # Multifractal statistics
            mf_data = df[df['mf_status'] == 'success']
            print(f"\n🔬 MULTIFRACTAL STATISTICS:")
            print(f"  D(0) range: {mf_data['mf_D0'].min():.4f} to {mf_data['mf_D0'].max():.4f}")
            print(f"  D(1) range: {mf_data['mf_D1'].min():.4f} to {mf_data['mf_D1'].max():.4f}")
            print(f"  D(2) range: {mf_data['mf_D2'].min():.4f} to {mf_data['mf_D2'].max():.4f}")
            print(f"  α width range: {mf_data['mf_alpha_width'].min():.4f} to {mf_data['mf_alpha_width'].max():.4f}")
            
            # Classify interfaces
            degree_mf = mf_data['mf_degree_multifractality']
            monofractal_count = len(degree_mf[degree_mf.abs() < 0.1])
            multifractal_count = len(degree_mf[degree_mf.abs() >= 0.1])
            print(f"  Interface classification: {monofractal_count} monofractal, {multifractal_count} multifractal")

    # NEW: Physics parameters summary
    if H is not None and L is not None and A is not None and g is not None and tau_factor is not None:
        print(f"\n📐 DOMAIN & PHYSICS PARAMETERS:")
        print(f"   Domain height (H): {H:.4f} m (auto-detected)")
        print(f"   Domain width (L): {L:.4f} m (auto-detected)")
        
        # Determine parameter sources
        A_source = "specified" if A != 2.1e-3 else "default"
        g_source = "specified" if g != 9.8 else "default"
        print(f"   Atwood number (A): {A:.3e} ({A_source})")
        print(f"   Gravity (g): {g:.3f} m/s² ({g_source})")
        print(f"   Dimensionless factor √(Ag/H): {tau_factor:.4f} s⁻¹")
        
        # Dimensionless time range
        if 'tau' in successful_results.columns and len(successful_results) > 0:
            tau_min, tau_max = successful_results['tau'].min(), successful_results['tau'].max()
            print(f"   Dimensionless time range: τ = {tau_min:.3f} to {tau_max:.3f}")
        
        # Dimensionless mixing thickness range
        if 'ht_normalized' in successful_results.columns and len(successful_results) > 0:
            ht_norm_max = successful_results['ht_normalized'].max()
            hb_norm_max = successful_results['hb_normalized'].max()
            h_total_norm_max = successful_results['h_total_normalized'].max()
            print(f"   Max dimensionless thicknesses: h_t/H = {ht_norm_max:.4f}, h_b/H = {hb_norm_max:.4f}, h_total/H = {h_total_norm_max:.4f}")
    
    if len(failed_results) > 0:
        print(f"Failed analyses: {len(failed_results)}")

        # Show failure summary
        failure_summary = failed_results.groupby(['grid_resolution', 'error']).size()
        print("Failure breakdown:")
        for (resolution, error), count in failure_summary.items():
            print(f"  {resolution}: {count} - {error[:50]}...")

    if len(successful_results) > 0:
        # Calculate efficiency metrics
        total_sequential_time = successful_results['processing_time'].sum()
        theoretical_speedup = total_sequential_time / total_time

        # Determine process count based on mode
        if mode == 'temporal_evolution':
            effective_processes = len(successful_results['grid_resolution'].unique())
        elif mode == 'convergence_study':
            effective_processes = len(successful_results['grid_resolution'].unique())
        elif mode == 'multi_time_convergence':
            effective_processes = len(successful_results['grid_resolution'].unique())
        else:  # matrix_analysis
            effective_processes = min(8, len(successful_results))  # Estimate

        parallel_efficiency = total_sequential_time / (total_time * effective_processes) * 100

        print(f"\n🚀 PARALLEL PERFORMANCE:")
        print(f"  Sequential time estimate: {total_sequential_time:.1f}s")
        print(f"  Actual parallel time: {total_time:.1f}s")
        print(f"  Theoretical speedup: {theoretical_speedup:.1f}×")
        print(f"  Parallel efficiency: {parallel_efficiency:.1f}%")
        print(f"  Average time per analysis: {successful_results['processing_time'].mean():.1f}s")

        print(f"\n📈 ANALYSIS RESULTS:")
        print(f"  Time range: {successful_results['actual_time'].min():.3f} to {successful_results['actual_time'].max():.3f}")
        print(f"  Fractal dimension range: {successful_results['fractal_dim'].min():.4f} to {successful_results['fractal_dim'].max():.4f}")
        print(f"  Segment count range: {int(successful_results['segments'].min())} to {int(successful_results['segments'].max())}")

        # Physics validation summary (informational)
        if 'physics_ratio' in successful_results.columns:
            physics_ratios = successful_results['physics_ratio'].dropna()
            if len(physics_ratios) > 0:
                print(f"  Physics ratios: {physics_ratios.min():.2f} to {physics_ratios.max():.2f}")
                expected_physics = len(physics_ratios[physics_ratios > 0])
                print(f"  Physics as expected cases: {expected_physics}/{len(physics_ratios)} ({expected_physics/len(physics_ratios)*100:.1f}%)")

        # Grid type analysis
        grid_info = analyze_grid_types(successful_results['resolution_str'].unique())
        if grid_info['has_mixed_types']:
            print(f"  Grid types: {grid_info['square_count']} square, {grid_info['rectangular_count']} rectangular")
            if grid_info['aspect_ratios']:
                print(f"  Max aspect ratio: {grid_info['max_aspect_ratio']:.2f}")

        # Mode-specific summaries
        if mode == 'temporal_evolution':
            print_temporal_evolution_summary(successful_results)
        elif mode == 'convergence_study':
            print_convergence_summary(successful_results)
        elif mode == 'multi_time_convergence':
            print_multi_time_convergence_summary(successful_results, enable_multifractal)
        else:  # matrix_analysis
            print_matrix_summary(successful_results)

def print_temporal_evolution_summary(df):
    """Print summary for temporal evolution analysis with physics validation."""
    print(f"\n🌊 TEMPORAL EVOLUTION SUMMARY:")

    for resolution_str in sorted(df['resolution_str'].unique()):
        res_data = df[df['resolution_str'] == resolution_str]
        if len(res_data) > 0:
            nx, ny = parse_grid_resolution(resolution_str)
            grid_str = format_grid_resolution(nx, ny)

            print(f"\n  {grid_str} Resolution ({len(res_data)} time points):")
            print(f"    Time range: {res_data['actual_time'].min():.3f} to {res_data['actual_time'].max():.3f}")
            if 'tau' in res_data.columns:
                print(f"    Dimensionless time range: τ = {res_data['tau'].min():.3f} to {res_data['tau'].max():.3f}")
            print(f"    D range: {res_data['fractal_dim'].min():.4f} to {res_data['fractal_dim'].max():.4f}")
            print(f"    Final mixing thickness: {res_data['h_total'].iloc[-1]:.4f}")
            if 'h_total_normalized' in res_data.columns:
                print(f"    Final dimensionless thickness: h_total/H = {res_data['h_total_normalized'].iloc[-1]:.4f}")
            print(f"    Initial interface height h0: {res_data['h0'].iloc[0]}")
            print(f"    Segment range: {int(res_data['segments'].min())} to {int(res_data['segments'].max())}")
            
            # Add physics validation info
            if 'physics_validation' in res_data.columns:
                expected_physics = len(res_data[res_data['physics_validation'] == 'expected'])
                print(f"    Physics as expected: {expected_physics}/{len(res_data)} cases")
            
            # Add multifractal info if available
            if 'mf_D0' in res_data.columns and not res_data['mf_D0'].isna().all():
                mf_data = res_data[res_data['mf_status'] == 'success']
                if len(mf_data) > 0:
                    print(f"    MF D0 range: {mf_data['mf_D0'].min():.4f} to {mf_data['mf_D0'].max():.4f}")

def print_convergence_summary(df):
    """Print summary for convergence analysis with physics validation."""
    print(f"\n📈 CONVERGENCE SUMMARY:")

    df_sorted = df.sort_values('effective_resolution')
    print(f"  Resolution progression:")

    for _, row in df_sorted.iterrows():
        physics_info = ""
        if 'physics_validation' in row and row['physics_validation'] == 'expected':
            physics_info = " ✅"
        elif 'physics_validation' in row and row['physics_validation'] == 'unexpected':
            physics_info = " ⚠️"
        
        mf_info = ""
        if 'mf_D0' in row and not pd.isna(row['mf_D0']):
            mf_info = f", MF D0 = {row['mf_D0']:.4f}"
        print(f"    {row['grid_resolution']}: D = {row['fractal_dim']:.4f} ± {row['fd_error']:.4f}{mf_info}{physics_info}")

    # Check for convergence
    if len(df_sorted) >= 2:
        fd_change = df_sorted['fractal_dim'].iloc[-1] - df_sorted['fractal_dim'].iloc[-2]
        fd_rel_change = abs(fd_change) / df_sorted['fractal_dim'].iloc[-1]

        print(f"\n  Convergence analysis:")
        print(f"    Latest change: {fd_change:.6f}")
        print(f"    Relative change: {fd_rel_change:.3%}")

        if fd_rel_change < 0.01:
            print(f"    ✅ Appears converged (< 1% change)")
        elif fd_rel_change < 0.05:
            print(f"    ⚠️  Near convergence (< 5% change)")
        else:
            print(f"    ❌ Not converged (≥ 5% change)")

def print_multi_time_convergence_summary(df, enable_multifractal=False):
    """Print summary for multi-time convergence analysis with physics validation."""
    print(f"\n🔄 MULTI-TIME CONVERGENCE SUMMARY:")

    # Group by time points
    time_points = sorted(df['target_time'].unique())
    print(f"  Analyzed {len(time_points)} time points:")

    for target_time in time_points:
        time_data = df[abs(df['actual_time'] - target_time) < 0.5]
        if len(time_data) > 0:
            time_data_sorted = time_data.sort_values('effective_resolution')

            print(f"\n  t = {target_time:.1f} ({len(time_data)} resolutions):")
            print(f"    D range: {time_data['fractal_dim'].min():.4f} to {time_data['fractal_dim'].max():.4f}")
            
            # Add physics validation info
            if 'physics_validation' in time_data.columns:
                expected_physics = len(time_data[time_data['physics_validation'] == 'expected'])
                print(f"    Physics as expected: {expected_physics}/{len(time_data)} cases")
            
            # Add multifractal info
            if enable_multifractal and 'mf_D0' in time_data.columns:
                mf_data = time_data[time_data['mf_status'] == 'success']
                if len(mf_data) > 0:
                    print(f"    MF D0 range: {mf_data['mf_D0'].min():.4f} to {mf_data['mf_D0'].max():.4f}")
                    # Classify interfaces at this time
                    degree_mf = mf_data['mf_degree_multifractality']
                    monofractal_count = len(degree_mf[degree_mf.abs() < 0.1])
                    multifractal_count = len(degree_mf[degree_mf.abs() >= 0.1])
                    print(f"    Interface types: {monofractal_count} monofractal, {multifractal_count} multifractal")

            # Check convergence for this time point
            if len(time_data_sorted) >= 2:
                fd_change = time_data_sorted['fractal_dim'].iloc[-1] - time_data_sorted['fractal_dim'].iloc[-2]
                fd_rel_change = abs(fd_change) / time_data_sorted['fractal_dim'].iloc[-1]

                if fd_rel_change < 0.01:
                    convergence_status = "✅ Converged"
                elif fd_rel_change < 0.05:
                    convergence_status = "⚠️ Near convergence"
                else:
                    convergence_status = "❌ Not converged"

                print(f"    Convergence: {convergence_status} (rel. change: {fd_rel_change:.3%})")
                print(f"    Best resolution: {time_data_sorted['grid_resolution'].iloc[-1]}")

def print_matrix_summary(df):
    """Print summary for matrix analysis with physics validation."""
    print(f"\n🔲 MATRIX SUMMARY:")

    resolutions = sorted(df['resolution_str'].unique())
    times = sorted(df['actual_time'].unique())

    print(f"  Matrix dimensions: {len(resolutions)} resolutions × {len(times)} times")
    print(f"  Coverage: {len(df)}/{len(resolutions) * len(times)} points")
    
    # Physics validation summary
    if 'physics_validation' in df.columns:
        expected_physics = len(df[df['physics_validation'] == 'expected'])
        print(f"  Physics as expected: {expected_physics}/{len(df)} cases")

def create_dimensionless_physics_plots(df, output_dir, method_suffix, analysis_params, enable_multifractal=False):
    """
    NEW: Create dimensionless physics plots h_t/H and h_b/H vs. τ.
    UPDATED: Handles new four-method framework from rt_analyzer.py
    """
    print(f"\n🧮 Creating dimensionless physics plots...")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        print("⚠️  No successful results for dimensionless plotting")
        return

    # Check if we have the required dimensionless data
    if 'tau' not in successful_df.columns or 'ht_normalized' not in successful_df.columns:
        print("⚠️  Missing dimensionless data (tau, ht_normalized, hb_normalized) for physics plots")
        return

    mixing_method = analysis_params.get('mixing_method', 'comprehensive')
    
    # Determine subplot layout based on mixing method
    if mixing_method == 'comprehensive':
        # Four-method comparison: 2x2 layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        plot_title_suffix = " (Four-Method Framework)"
    else:
        # Single method: 1x2 layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        plot_title_suffix = f" ({mixing_method.title()} Method)"

    resolutions = sorted(successful_df['resolution_str'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    if mixing_method != 'comprehensive':
        # Single method plots using legacy compatibility fields
        # Plot h_t/H vs τ
        for i, resolution_str in enumerate(resolutions):
            res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
            nx, ny = parse_grid_resolution(resolution_str)
            grid_str = format_grid_resolution(nx, ny)
            
            if len(res_data) > 0:
                # Split by physics validation for visual distinction
                expected_physics = res_data[res_data.get('physics_validation') == 'expected']
                unexpected_physics = res_data[res_data.get('physics_validation') == 'unexpected']
                
                if len(expected_physics) > 0:
                    ax1.plot(expected_physics['tau'], expected_physics['ht_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} ✅', alpha=1.0)
                
                if len(unexpected_physics) > 0:
                    ax1.plot(unexpected_physics['tau'], unexpected_physics['ht_normalized'], 
                            's--', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} ⚠️', alpha=0.6)
        
        ax1.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
        ax1.set_ylabel('h_t/H')
        ax1.set_title(f'Upper Mixing Thickness Evolution{plot_title_suffix}')
        ax1.grid(True, alpha=0.7)
        ax1.legend()
        
        # Plot h_b/H vs τ
        for i, resolution_str in enumerate(resolutions):
            res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
            nx, ny = parse_grid_resolution(resolution_str)
            grid_str = format_grid_resolution(nx, ny)
            
            if len(res_data) > 0:
                # Split by physics validation
                expected_physics = res_data[res_data.get('physics_validation') == 'expected']
                unexpected_physics = res_data[res_data.get('physics_validation') == 'unexpected']
                
                if len(expected_physics) > 0:
                    ax2.plot(expected_physics['tau'], expected_physics['hb_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} ✅', alpha=1.0)
                
                if len(unexpected_physics) > 0:
                    ax2.plot(unexpected_physics['tau'], unexpected_physics['hb_normalized'], 
                            's--', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} ⚠️', alpha=0.6)
        
        ax2.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
        ax2.set_ylabel('h_b/H')
        ax2.set_title(f'Lower Mixing Thickness Evolution{plot_title_suffix}')
        ax2.grid(True, alpha=0.7)
        ax2.legend()

    else:
        # Four-method comparison plots
        
        # Plot 1: Mass-conservation method (h₀₁/H and h₁₀/H vs τ)
        if 'h_01_normalized' in successful_df.columns and 'h_10_normalized' in successful_df.columns:
            for i, resolution_str in enumerate(resolutions):
                res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
                nx, ny = parse_grid_resolution(resolution_str)
                grid_str = format_grid_resolution(nx, ny)
                
                if len(res_data) > 0:
                    ax1.plot(res_data['tau'], res_data['h_01_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} h₀₁')
                    ax1.plot(res_data['tau'], res_data['h_10_normalized'], 
                            's--', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} h₁₀', alpha=0.7)
            
            ax1.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
            ax1.set_ylabel('Normalized Thickness')
            ax1.set_title('Mass-Conservation Method: h₀₁/H and h₁₀/H vs τ')
            ax1.grid(True, alpha=0.7)
            ax1.legend()
        
        # Plot 2: Geometric penetration method
        if 'h_penetration_up_normalized' in successful_df.columns and 'h_penetration_down_normalized' in successful_df.columns:
            for i, resolution_str in enumerate(resolutions):
                res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
                nx, ny = parse_grid_resolution(resolution_str)
                grid_str = format_grid_resolution(nx, ny)
                
                if len(res_data) > 0:
                    ax2.plot(res_data['tau'], res_data['h_penetration_up_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} h_up')
                    ax2.plot(res_data['tau'], res_data['h_penetration_down_normalized'], 
                            's--', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} h_down', alpha=0.7)
            
            ax2.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
            ax2.set_ylabel('Normalized Thickness')
            ax2.set_title('Geometric Method: h_up/H and h_down/H vs τ')
            ax2.grid(True, alpha=0.7)
            ax2.legend()
        
        # Plot 3: Youngs integral width method
        if 'W_normalized' in successful_df.columns:
            for i, resolution_str in enumerate(resolutions):
                res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
                nx, ny = parse_grid_resolution(resolution_str)
                grid_str = format_grid_resolution(nx, ny)
                
                if len(res_data) > 0:
                    ax3.plot(res_data['tau'], res_data['W_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} W')
            
            ax3.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
            ax3.set_ylabel('W/H')
            ax3.set_title('Youngs Method: W/H vs τ')
            ax3.grid(True, alpha=0.7)
            ax3.legend()
        
        # Plot 4: Method comparison for total mixing thickness
        for i, resolution_str in enumerate(resolutions):
            res_data = successful_df[successful_df['resolution_str'] == resolution_str].sort_values('tau')
            nx, ny = parse_grid_resolution(resolution_str)
            grid_str = format_grid_resolution(nx, ny)
            
            if len(res_data) > 0:
                if 'h_total_mass_normalized' in res_data.columns:
                    ax4.plot(res_data['tau'], res_data['h_total_mass_normalized'], 
                            'o-', color=colors[i], linewidth=2, markersize=6, 
                            label=f'{grid_str} Mass')
                if 'h_total_geometric_normalized' in res_data.columns:
                    ax4.plot(res_data['tau'], res_data['h_total_geometric_normalized'], 
                            's--', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} Geometric', alpha=0.7)
                if 'W_normalized' in res_data.columns:
                    ax4.plot(res_data['tau'], res_data['W_normalized'], 
                            '^:', color=colors[i], linewidth=2, markersize=4, 
                            label=f'{grid_str} Youngs', alpha=0.7)
        
        ax4.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
        ax4.set_ylabel('Normalized Mixing Measure')
        ax4.set_title('Four-Method Comparison: Total Mixing Measures')
        ax4.grid(True, alpha=0.7)
        ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    A = analysis_params.get('atwood_number', 2.1e-3)
    g = analysis_params.get('gravity', 9.8)
    physics_suffix = f"_A{A:.1e}_g{g:.1f}" if A != 2.1e-3 or g != 9.8 else ""
    mf_suffix = "_mf" if enable_multifractal else ""
    
    plot_filename = f'dimensionless_physics_{mixing_method}{method_suffix}{mf_suffix}{physics_suffix}.png'
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved dimensionless physics plots: {plot_filename}")
    
    # Create additional summary info
    tau_min, tau_max = successful_df['tau'].min(), successful_df['tau'].max()
    ht_norm_max = successful_df['ht_normalized'].max()
    hb_norm_max = successful_df['hb_normalized'].max()
    
    print(f"   📊 Dimensionless ranges:")
    print(f"      τ range: {tau_min:.3f} to {tau_max:.3f}")
    print(f"      Max h_t/H: {ht_norm_max:.4f}")
    print(f"      Max h_b/H: {hb_norm_max:.4f}")
    
    # Four-method summary if available
    if mixing_method == 'comprehensive':
        if 'h_total_mass_normalized' in successful_df.columns:
            h_mass_max = successful_df['h_total_mass_normalized'].max()
            print(f"      Max h_mass/H: {h_mass_max:.4f}")
        if 'h_total_geometric_normalized' in successful_df.columns:
            h_geom_max = successful_df['h_total_geometric_normalized'].max()
            print(f"      Max h_geometric/H: {h_geom_max:.4f}")
        if 'W_normalized' in successful_df.columns:
            W_max = successful_df['W_normalized'].max()
            print(f"      Max W/H: {W_max:.4f}")

def create_hybrid_plots(df, output_dir, analysis_params):
    """UPDATED: Create appropriate plots based on analysis mode with dimensionless physics plots."""
    method_name, method_suffix, method_description = get_method_info(analysis_params)
    enable_multifractal = analysis_params.get('enable_multifractal', False)
    mode = df['analysis_mode'].iloc[0] if 'analysis_mode' in df else 'unknown'

    print(f"\n📊 Creating plots for {mode} analysis...")
    print(f"   Interface extraction: {method_description}")
    if enable_multifractal:
        print(f"   Multifractal analysis: Enabled")
    print(f"   Dimensionless physics plots: Enabled")

    # Create standard plots based on analysis mode
    if mode == 'multi_time_convergence':
        create_multi_time_convergence_plots(df, output_dir, method_suffix, enable_multifractal)
    elif mode == 'temporal_evolution':
        create_temporal_evolution_plots(df, output_dir, method_suffix, enable_multifractal)
    elif mode == 'convergence_study':
        create_convergence_plots(df, output_dir, method_suffix, enable_multifractal)
    elif mode == 'matrix_analysis':
        create_matrix_plots(df, output_dir, method_suffix, enable_multifractal)
    else:
        print(f"   Standard plots for mode '{mode}' will be created")

    # NEW: Always create dimensionless physics plots for temporal data
    if 'tau' in df.columns and any(mode in ['temporal_evolution', 'multi_time_convergence', 'matrix_analysis'] for mode in [mode]):
        create_dimensionless_physics_plots(df, output_dir, method_suffix, analysis_params, enable_multifractal)
    else:
        print(f"   Dimensionless physics plots not applicable for {mode} mode")

# Placeholder plotting functions (basic implementations)
def create_multi_time_convergence_plots(df, output_dir, method_suffix, enable_multifractal=False):
    """Create basic multi-time convergence plots."""
    print(f"   ✅ Creating multi-time convergence plots")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Basic fractal dimension plot
    resolutions = sorted(successful_df['resolution_str'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
    
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        if len(res_data) > 0:
            ax1.plot(res_data['actual_time'], res_data['fractal_dim'], 
                    'o-', color=colors[i], label=resolution_str)
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # Basic mixing thickness plot
    for i, resolution_str in enumerate(resolutions):
        res_data = successful_df[successful_df['resolution_str'] == resolution_str]
        if len(res_data) > 0:
            ax2.plot(res_data['actual_time'], res_data['h_total'], 
                    'o-', color=colors[i], label=resolution_str)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Mixing Thickness')
    ax2.set_title('Mixing Thickness Evolution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'multi_time_convergence{method_suffix}.png'), dpi=300)
    plt.close()

def create_temporal_evolution_plots(df, output_dir, method_suffix, enable_multifractal=False):
    """Create basic temporal evolution plots."""
    print(f"   ✅ Creating temporal evolution plots")
    create_multi_time_convergence_plots(df, output_dir, method_suffix, enable_multifractal)

def create_convergence_plots(df, output_dir, method_suffix, enable_multifractal=False):
    """Create basic convergence plots."""
    print(f"   ✅ Creating convergence plots")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        return
    
    successful_df = successful_df.sort_values('effective_resolution')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Fractal dimension convergence
    ax1.errorbar(successful_df['effective_resolution'], successful_df['fractal_dim'],
                yerr=successful_df['fd_error'], fmt='bo-', capsize=5)
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Effective Resolution')
    ax1.set_ylabel('Fractal Dimension')
    ax1.set_title('Fractal Dimension Convergence')
    ax1.grid(True)
    
    # Mixing thickness convergence
    ax2.plot(successful_df['effective_resolution'], successful_df['h_total'], 'go-')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Effective Resolution')
    ax2.set_ylabel('Mixing Thickness')
    ax2.set_title('Mixing Thickness Convergence')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'convergence{method_suffix}.png'), dpi=300)
    plt.close()

def create_matrix_plots(df, output_dir, method_suffix, enable_multifractal=False):
    """Create basic matrix plots."""
    print(f"   ✅ Creating matrix plots")
    
    successful_df = df[df['status'] == 'success'].copy()
    if len(successful_df) == 0:
        return
    
    # Create a simple scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    scatter = ax1.scatter(successful_df['actual_time'], successful_df['effective_resolution'], 
                         c=successful_df['fractal_dim'], cmap='viridis')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Effective Resolution')
    ax1.set_title('Fractal Dimension Matrix')
    plt.colorbar(scatter, ax=ax1)
    
    scatter2 = ax2.scatter(successful_df['actual_time'], successful_df['effective_resolution'], 
                          c=successful_df['h_total'], cmap='plasma')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Effective Resolution')
    ax2.set_title('Mixing Thickness Matrix')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'matrix{method_suffix}.png'), dpi=300)
    plt.close()

# Enhanced Analyzer Block 5: Main Function with Physics Parameters

def main():
    """UPDATED: Main function with physics parameters and enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description='ENHANCED Hybrid Parallel Resolution Analyzer with Dimensionless Physics Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 ENHANCED ANALYSIS MODES (automatically detected):
  Temporal Evolution:     Single resolution, multiple times
  Convergence Study:      Multiple resolutions, single time
  Multi-Time Convergence: Multiple resolutions, multiple times
  Matrix Analysis:        Full parameter space exploration

🧮 NEW: DIMENSIONLESS PHYSICS PLOTS:
  h_t/H and h_b/H vs. τ = √(Ag/H)×t for journal-ready figures
  Auto-detection of domain dimensions H and L from VTK data
  Smart defaults for Atwood number A and gravity g
  Four-method framework compatibility (mass_conservation, geometric, youngs, diagnostic)

🔬 MULTIFRACTAL ANALYSIS:
  Complete multifractal spectrum analysis for all modes
  Generalized dimensions D(q) for q ∈ [-5, 5]
  Interface classification (monofractal vs multifractal)

✅ PHYSICS VALIDATION:
  Physics validation now INFORMATIONAL ONLY
  Plots will be generated regardless of physics validation status
  No more failures due to unexpected physics ratios

📐 GRID SUPPORT:
  Square grids:       --resolutions 200 400 800
  Rectangular grids:  --resolutions 160x200 320x400 640x800
  Mixed grids:        --resolutions 200 160x200 400x400 320x400

⚡ INTERFACE EXTRACTION METHODS:
  Standard:           Default scikit-image method
  Precision:          --use-conrec (CONREC algorithm)
  Theoretical:        --use-plic (PLIC reconstruction)

📊 EXAMPLES:

# Multi-time convergence with dimensionless physics plots
python enhanced_analyzer.py \\
  --data-dirs ~/RT/160x200 ~/RT/320x400 ~/RT/640x800 \\
  --resolutions 160x200 320x400 640x800 \\
  --target-times 1.0 2.0 4.0 6.0 8.0 10.0 12.0 14.0 15.0 \\
  --use-conrec --h0 0.5 --mixing-method comprehensive \\
  --enable-multifractal

# Custom physics parameters with four-method framework
python enhanced_analyzer.py \\
  --data-dirs ~/RT/160x200 \\
  --resolutions 160x200 \\
  --target-times 1.0 2.0 3.0 4.0 5.0 \\
  --atwood-number 1.8e-3 --gravity 9.81 \\
  --mixing-method comprehensive

# Mass-conservation method with default physics
python enhanced_analyzer.py \\
  --data-dirs ~/RT/160x200 \\
  --resolutions 160x200 \\
  --target-times 1.0 2.0 3.0 4.0 5.0 \\
  --use-plic --mixing-method mass_conservation
""")

    # Required arguments
    parser.add_argument('--data-dirs', nargs='+', required=True,
                       help='Data directories (one per resolution)')
    parser.add_argument('--resolutions', nargs='+', required=True,
                       help='Grid resolutions (e.g., "200" for square, "160x200" for rectangular)')
    parser.add_argument('--target-times', nargs='+', type=float, required=True,
                       help='Target simulation times for analysis')

    # Optional arguments
    parser.add_argument('--output-dir', default=None,
                       help='Output directory (default: auto-generated based on analysis mode)')
    parser.add_argument('--mixing-method', default='comprehensive',
                       choices=['mass_conservation', 'geometric', 'diagnostic', 'youngs', 'comprehensive'],
                       help='Mixing thickness calculation method: mass_conservation (your theory), geometric (Dalziel), diagnostic (global mixedness), youngs (robust width), comprehensive (all methods) (default: comprehensive)')
    parser.add_argument('--h0', type=float, default=0.5,
                       help='Initial interface position (default: 0.5)')
    parser.add_argument('--min-box-size', type=float, default=None,
                       help='Minimum box size for fractal analysis (default: auto-estimate)')
    parser.add_argument('--time-tolerance', type=float, default=0.5,
                       help='Maximum time difference allowed when finding files (default: 0.5)')

    # NEW: Physics parameters with smart defaults
    parser.add_argument('--atwood-number', type=float, default=2.1e-3,
                       help='Atwood number A for dimensionless time τ = √(Ag/H)t (default: 2.1e-3)')
    parser.add_argument('--gravity', type=float, default=9.8,
                       help='Gravitational acceleration g in m/s² (default: 9.8)')
    
    # Note: Domain height H and width L are auto-detected from VTK data

    # Multifractal analysis arguments
    parser.add_argument('--enable-multifractal', action='store_true',
                       help='Enable multifractal spectrum analysis for all analyses')
    parser.add_argument('--q-values', nargs='+', type=float, default=None,
                       help='Q values for multifractal analysis (default: -5 to 5 in 0.5 steps)')
    parser.add_argument('--mf-output-dir', default=None,
                       help='Output directory for multifractal results (default: subdirectory of main output)')

    # Parallel processing
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto-detect based on mode)')

    # Interface extraction methods
    parser.add_argument('--use-conrec', action='store_true',
                       help='Use CONREC precision interface extraction')
    parser.add_argument('--use-plic', action='store_true',
                       help='Use PLIC theoretical interface reconstruction')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output for extraction methods')

    # Mode control options
    parser.add_argument('--force-matrix-mode', action='store_true',
                       help='Force matrix analysis mode (override auto-detection)')

    # Output control
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation (including dimensionless physics plots)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Validate arguments
    if len(args.data_dirs) != len(args.resolutions):
        print("❌ Number of data directories must match number of resolutions")
        return 1

    # Check data directories
    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return 1

    # Validate extraction method conflicts
    if args.use_conrec and args.use_plic:
        print("⚠️  Both CONREC and PLIC specified. PLIC will take precedence.")
        args.use_conrec = False

    # Show physics parameters info
    print(f"🧮 Physics parameters for dimensionless plots:")
    print(f"   Atwood number (A): {args.atwood_number:.3e}")
    print(f"   Gravity (g): {args.gravity:.2f} m/s²")
    print(f"   Domain dimensions (H, L): Auto-detected from VTK data")
    print(f"   Dimensionless time: τ = √(Ag/H) × t")

    # Show method-specific info
    if args.mixing_method == 'geometric':
        print(f"🎯 Using geometric penetration method (Dalziel-style)")
        print(f"   Threshold method: F̄=0.05 and F̄=0.95 boundaries")
    elif args.mixing_method == 'mass_conservation':
        print(f"🎯 Using mass-conservation method (your theoretical framework)")
        print(f"   Computes h₀₁ and h₁₀ from mass conservation")
    elif args.mixing_method == 'comprehensive':
        print(f"🎯 Using comprehensive analysis (all four methods)")
        print(f"   Mass-conservation + Geometric + Youngs + Global diagnostic")

    # Set analysis parameters with physics parameters
    analysis_params = {
        'mixing_method': args.mixing_method,
        'h0': args.h0,
        'min_box_size': args.min_box_size,
        'time_tolerance': args.time_tolerance,
        'use_conrec': args.use_conrec,
        'use_plic': args.use_plic,
        'debug': args.debug,
        'verbose': args.verbose,
        'force_matrix_mode': args.force_matrix_mode,
        # Multifractal parameters
        'enable_multifractal': args.enable_multifractal,
        'q_values': args.q_values,
        'mf_output_dir': args.mf_output_dir,
        # NEW: Physics parameters
        'atwood_number': args.atwood_number,
        'gravity': args.gravity
    }

    # Run enhanced hybrid analysis with physics parameters
    df = run_hybrid_analysis(
        args.data_dirs,
        args.resolutions,
        args.target_times,
        args.output_dir,
        analysis_params,
        args.processes
    )

    if df is None:
        print("❌ Analysis failed")
        return 1

    # Create plots unless disabled (including NEW dimensionless physics plots)
    if not args.no_plots:
        try:
            # Determine output directory from DataFrame if not explicitly set
            if 'analysis_mode' in df.columns:
                output_dir = args.output_dir
                if output_dir is None:
                    method_name, method_suffix, _ = get_method_info(analysis_params)
                    mode = df['analysis_mode'].iloc[0]
                    # Include physics info in directory name
                    A = args.atwood_number
                    g = args.gravity
                    physics_suffix = f"_A{A:.1e}_g{g:.1f}" if A != 2.1e-3 or g != 9.8 else ""
                    output_dir = create_output_directory_name(mode, args.resolutions, args.target_times, 
                                                            method_suffix, args.enable_multifractal, physics_suffix)

                create_hybrid_plots(df, output_dir, analysis_params)
            else:
                print("⚠️  Could not determine analysis mode for plotting")
        except Exception as e:
            print(f"⚠️  Plot generation failed: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Enhanced final summary with physics parameters
    successful_count = len(df[df['status'] == 'success']) if 'status' in df.columns else len(df)
    expected_physics_count = len(df[df.get('physics_validation') == 'expected']) if 'physics_validation' in df.columns else 0
    mode = df['analysis_mode'].iloc[0] if 'analysis_mode' in df.columns else 'unknown'
    method_name, _, _ = get_method_info(analysis_params)

    print(f"\n🎉 ENHANCED HYBRID ANALYSIS COMPLETE!")
    if args.enable_multifractal:
        print(f"🔬 WITH MULTIFRACTAL ANALYSIS!")
    print(f"🧮 WITH DIMENSIONLESS PHYSICS PLOTS!")
    print(f"✅ WITH FOUR-METHOD FRAMEWORK!")
    
    print(f"Mode: {mode.replace('_', ' ').title()}")
    print(f"Method: {method_name}")
    print(f"Mixing: {args.mixing_method}")
    print(f"Success rate: {successful_count}/{len(df)}")
    print(f"Physics as expected: {expected_physics_count} cases (informational)")
    
    # Multifractal success statistics
    if args.enable_multifractal and 'mf_status' in df.columns:
        mf_successful = len(df[df['mf_status'] == 'success'])
        print(f"Multifractal success rate: {mf_successful}/{len(df)}")

    # NEW: Physics parameters confirmation in final summary
    if 'domain_height' in analysis_params and 'tau_factor' in analysis_params:
        H = analysis_params['domain_height']
        L = analysis_params['domain_width']
        tau_factor = analysis_params['tau_factor']
        
        print(f"\n📐 CONFIRMED PHYSICS PARAMETERS:")
        print(f"   Domain height (H): {H:.4f} m (auto-detected)")
        print(f"   Domain width (L): {L:.4f} m (auto-detected)")
        A_source = "specified" if args.atwood_number != 2.1e-3 else "default"
        g_source = "specified" if args.gravity != 9.8 else "default"
        print(f"   Atwood number (A): {args.atwood_number:.3e} ({A_source})")
        print(f"   Gravity (g): {args.gravity:.3f} m/s² ({g_source})")
        print(f"   Dimensionless factor √(Ag/H): {tau_factor:.4f} s⁻¹")
        
        # Dimensionless ranges achieved
        if 'tau' in df.columns and len(df[df['status'] == 'success']) > 0:
            successful_df = df[df['status'] == 'success']
            tau_min, tau_max = successful_df['tau'].min(), successful_df['tau'].max()
            print(f"   Dimensionless time range achieved: τ = {tau_min:.3f} to {tau_max:.3f}")
        
        if 'h_total_normalized' in df.columns and len(df[df['status'] == 'success']) > 0:
            successful_df = df[df['status'] == 'success']
            h_norm_max = successful_df['h_total_normalized'].max()
            print(f"   Maximum dimensionless mixing thickness: h_total/H = {h_norm_max:.4f}")

    print(f"\n📁 Check the output directory for:")
    print(f"   • Detailed results and standard plots")
    print(f"   • NEW: Dimensionless physics plots (h_t/H, h_b/H vs τ)")
    print(f"   • Four-method framework analysis results")
    print(f"   • Enhanced summaries with physics parameter documentation")
    print(f"✅ UPDATED: Compatible with new rt_analyzer.py four-method framework!")

    return 0

if __name__ == "__main__":
    exit(main())
