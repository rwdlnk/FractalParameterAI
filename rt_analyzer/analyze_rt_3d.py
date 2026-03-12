#!/usr/bin/env python3
"""
3D Rayleigh-Taylor Simulation Analysis Tool

Performs comprehensive analysis of 3D OpenFOAM RT simulation results:
  Phase 1: Mixing thickness (Dalziel Eq. 7) + true 3D fractal dimension (all timesteps)
  Phase 2: 3D multifractal spectrum at selected timesteps

Data sources per timestep:
  - Volume field (alpha.water): mixing thickness via xy-plane-averaged F profile
  - Isosurface mesh (extractInterface VTK): fractal dimension via 3D cube counting

Requires:
  - 3D-FractalParameterAI library (fractal3d)
  - OpenFOAM postProcessing/extractInterface output (interface.vtk files)
  - Decomposed or reconstructed alpha.water fields

Usage:
  python3 analyze_rt_3d.py /path/to/3D/case [options]

Examples:
  python3 analyze_rt_3d.py /media/rod/OpenFOAM/rod-13/run/RT_Dalziel/3D/160x200x80
  python3 analyze_rt_3d.py /path/to/3D/case --mf-times 4,8,12,16 --skip-phase2
"""

import os
import sys
import re
import glob
import argparse
import subprocess
import time as time_mod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# numpy <2.0 compatibility: trapezoid was called trapz
_trapezoid = getattr(np, 'trapezoid', np.trapz)
from matplotlib.cm import get_cmap
from scipy import stats

# ─── Path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# 3D-FractalParameterAI library
_FRACTAL3D_PATH = os.path.join(os.path.dirname(_REPO_ROOT), '3D-FractalParameterAI')
if os.path.isdir(_FRACTAL3D_PATH):
    sys.path.insert(0, _FRACTAL3D_PATH)

from fractal3d import (
    load_mesh, TriangleMesh, BoundingBox3D,
    compute_fractal_dimension_3d, FractalDimensionResult,
    FastCubeCounter, check_numba_available,
    extract_surface_features, suggest_parameters,
    MultifractalAnalyzer3D, MultifractalResult3D,
    RTPhysics3D,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh / Field I/O (adapted from openfoam_3d_analyzer.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeshInfo3D:
    nx: int
    ny: int
    nz: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    dx: float
    dy: float
    dz: float


def read_mesh_info(case_dir) -> Optional[MeshInfo3D]:
    """Read mesh info from blockMeshDict."""
    blockMeshDict = os.path.join(case_dir, 'system', 'blockMeshDict')
    if not os.path.isfile(blockMeshDict):
        return None

    with open(blockMeshDict, 'r') as f:
        content = f.read()

    vertices_match = re.search(r'vertices\s*\(\s*([\s\S]*?)\s*\)\s*;', content)
    blocks_match = re.search(r'hex\s*\([^)]+\)\s*\((\d+)\s+(\d+)\s+(\d+)\)', content)

    if not vertices_match or not blocks_match:
        return None

    nx = int(blocks_match.group(1))
    ny = int(blocks_match.group(2))
    nz = int(blocks_match.group(3))

    verts_str = vertices_match.group(1)
    coords = re.findall(r'\(\s*([0-9.e+-]+)\s+([0-9.e+-]+)\s+([0-9.e+-]+)\s*\)', verts_str)

    x_coords = [float(c[0]) for c in coords]
    y_coords = [float(c[1]) for c in coords]
    z_coords = [float(c[2]) for c in coords]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)

    return MeshInfo3D(
        nx=nx, ny=ny, nz=nz,
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        z_min=z_min, z_max=z_max,
        dx=(x_max - x_min) / nx,
        dy=(y_max - y_min) / ny,
        dz=(z_max - z_min) / nz,
    )


def read_openfoam_alpha_3d(time_dir, mesh: MeshInfo3D) -> Optional[np.ndarray]:
    """Read alpha.water field from OpenFOAM time directory and reshape to 3D.

    Returns:
        ndarray of shape (nz, ny, nx) or None on failure.
    """
    alpha_file = os.path.join(time_dir, 'alpha.water')
    if not os.path.exists(alpha_file):
        return None

    with open(alpha_file, 'r') as f:
        content = f.read()

    match = re.search(
        r'internalField\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(([\s\S]*?)\)\s*;',
        content)
    if not match:
        match = re.search(r'internalField\s+uniform\s+([0-9.e+-]+)', content)
        if match:
            return np.full((mesh.nz, mesh.ny, mesh.nx), float(match.group(1)))
        return None

    values = np.array([float(v) for v in match.group(2).split()])

    # OpenFOAM ordering: x varies fastest, then y, then z
    try:
        alpha_3d = values.reshape(mesh.nz, mesh.ny, mesh.nx)
    except ValueError:
        print(f"  Shape mismatch: got {len(values)}, expected {mesh.nx*mesh.ny*mesh.nz}")
        return None

    return alpha_3d


def reconstruct_timestep(case_dir, t) -> str:
    """Reconstruct a single timestep (alpha.water only) if needed.

    Returns path to the time directory.
    """
    t_str = str(int(t)) if t == int(t) else f"{t}"
    recon_dir = os.path.join(case_dir, t_str)

    if os.path.exists(os.path.join(recon_dir, 'alpha.water')):
        return recon_dir

    print(f"    Reconstructing t={t}...", end=" ", flush=True)
    cmd = (f"cd \"{case_dir}\" && "
           f"reconstructPar -time {t} -fields '(alpha.water)' > /dev/null 2>&1")
    subprocess.run(cmd, shell=True)
    print("done")

    return recon_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Physics extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_physics_3d(case_dir, mesh: MeshInfo3D) -> dict:
    """Extract physics parameters from a 3D OpenFOAM case.

    Returns dict with A, g, H, L, W, H0, rho_heavy, rho_light, etc.
    """
    H = mesh.y_max - mesh.y_min
    L = mesh.x_max - mesh.x_min
    W = mesh.z_max - mesh.z_min
    H0 = (mesh.y_min + mesh.y_max) / 2.0

    # Parse fluid properties: v13 split files
    rho_heavy, rho_light = 998.0, 998.0
    nu_heavy, nu_light = 1.0e-6, 1.0e-6

    pp_water = os.path.join(case_dir, 'constant', 'physicalProperties.water')
    pp_air = os.path.join(case_dir, 'constant', 'physicalProperties.air')
    if os.path.isfile(pp_water) and os.path.isfile(pp_air):
        for pp_file, phase in [(pp_water, 'water'), (pp_air, 'air')]:
            with open(pp_file, 'r') as f:
                content = f.read()
            m_rho = re.search(r'rho\s+([\d.eE+-]+)', content)
            m_nu = re.search(r'nu\s+([\d.eE+-]+)', content)
            if phase == 'water':
                if m_rho: rho_heavy = float(m_rho.group(1))
                if m_nu: nu_heavy = float(m_nu.group(1))
            else:
                if m_rho: rho_light = float(m_rho.group(1))
                if m_nu: nu_light = float(m_nu.group(1))

    # Parse gravity
    g = 9.81
    g_file = os.path.join(case_dir, 'constant', 'g')
    if os.path.isfile(g_file):
        with open(g_file, 'r') as f:
            content = f.read()
        m = re.search(r'value\s*\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)', content)
        if m:
            g = abs(float(m.group(2)))

    A = (rho_heavy - rho_light) / (rho_heavy + rho_light) if (rho_heavy + rho_light) > 0 else 0.0

    return {
        'A': A, 'g': g, 'H': H, 'L': L, 'W': W, 'H0': H0,
        'NX': mesh.nx, 'NY': mesh.ny, 'NZ': mesh.nz,
        'dx': mesh.dx, 'dy': mesh.dy, 'dz': mesh.dz,
        'rho_heavy': rho_heavy, 'rho_light': rho_light,
        'nu_heavy': nu_heavy, 'nu_light': nu_light,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Timestep discovery
# ═══════════════════════════════════════════════════════════════════════════════

def find_interface_timesteps(case_dir) -> List[Tuple[float, str]]:
    """Find all timesteps with extractInterface VTK output.

    Returns sorted list of (time_float, vtk_path) tuples.
    """
    ei_dir = os.path.join(case_dir, 'postProcessing', 'extractInterface')
    if not os.path.isdir(ei_dir):
        return []

    times = []
    for item in os.listdir(ei_dir):
        item_path = os.path.join(ei_dir, item)
        if not os.path.isdir(item_path):
            continue
        try:
            t = float(item)
        except ValueError:
            continue
        vtk_file = os.path.join(item_path, 'interface.vtk')
        if os.path.isfile(vtk_file):
            times.append((t, vtk_file))

    times.sort(key=lambda x: x[0])
    return times


def find_alpha_timesteps(case_dir) -> List[Tuple[float, str]]:
    """Find timesteps with alpha.water (reconstructed or in processor0).

    Returns sorted list of (time_float, source) tuples where source is:
    - The time directory path if reconstructed fields exist
    - 'decomposed' if only processor-level data exists
    """
    # Check for reconstructed time dirs first
    times = []
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if not os.path.isdir(item_path):
            continue
        try:
            t = float(item)
        except ValueError:
            continue
        if os.path.isfile(os.path.join(item_path, 'alpha.water')):
            times.append((t, item_path))

    if times:
        times.sort(key=lambda x: x[0])
        return times

    # Check processor0 for decomposed data
    proc0 = os.path.join(case_dir, 'processor0')
    if os.path.isdir(proc0):
        for item in os.listdir(proc0):
            try:
                t = float(item)
            except ValueError:
                continue
            if os.path.isfile(os.path.join(proc0, item, 'alpha.water')):
                times.append((t, 'decomposed'))
        times.sort(key=lambda x: x[0])

    return times


# ═══════════════════════════════════════════════════════════════════════════════
# Mixing thickness from 3D volume field
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mixing_3d(alpha_3d: np.ndarray, y_centers: np.ndarray,
                      H0: float, H: float) -> dict:
    """Compute all mixing thickness measures from 3D alpha field.

    Averages alpha over x AND z to produce <F>(y), then applies the same
    integral method (Dalziel Eq. 7) as the 2D code.

    Args:
        alpha_3d: VOF field, shape (nz, ny, nx). alpha.water = 1 for heavy fluid.
        y_centers: Cell center y-coordinates, shape (ny,).
        H0: Initial interface height.
        H: Domain height.

    Returns:
        Dict with geometric, statistical, and integral mixing measures.
    """
    # Average over x (axis=2) and z (axis=0) → <F>(y)
    f_avg = np.mean(alpha_3d, axis=(0, 2))

    result = {}

    # --- Geometric method ---
    # Upper boundary: highest y where f_avg significantly differs from boundary value
    # Lower boundary: lowest y where f_avg significantly differs from boundary value
    # Detect orientation: OpenFOAM alpha.water = 1 for heavy fluid
    f_increases = f_avg[-1] > f_avg[0]

    # --- Statistical method ---
    epsilon = 0.01
    if f_increases:
        upper_idx = np.where(f_avg < 1 - epsilon)[0]
        y_upper = y_centers[upper_idx[-1]] if len(upper_idx) > 0 else y_centers[-1]
        lower_idx = np.where(f_avg > epsilon)[0]
        y_lower = y_centers[lower_idx[0]] if len(lower_idx) > 0 else y_centers[0]
    else:
        upper_idx = np.where(f_avg > epsilon)[0]
        y_upper = y_centers[upper_idx[-1]] if len(upper_idx) > 0 else y_centers[-1]
        lower_idx = np.where(f_avg < 1 - epsilon)[0]
        y_lower = y_centers[lower_idx[0]] if len(lower_idx) > 0 else y_centers[0]

    result['ht_stat'] = max(0.0, y_upper - H0)
    result['hb_stat'] = max(0.0, H0 - y_lower)
    result['h_total_stat'] = result['ht_stat'] + result['hb_stat']

    # --- Integral method (Dalziel 1999 Eq. 7) ---
    lower_mask = y_centers <= H0
    upper_mask = y_centers >= H0

    y_lo = y_centers[lower_mask]
    y_up = y_centers[upper_mask]
    f_lo = f_avg[lower_mask]
    f_up = f_avg[upper_mask]

    if f_increases:
        # F increases with y → heavy on top
        h_10 = float(_trapezoid(f_lo, y_lo))
        h_11 = float(_trapezoid(f_up, y_up))
        h_00 = float(_trapezoid(1.0 - f_lo, y_lo))
        h_01 = float(_trapezoid(1.0 - f_up, y_up))
    else:
        # F decreases with y → heavy on bottom
        h_10 = float(_trapezoid(1.0 - f_lo, y_lo))
        h_11 = float(_trapezoid(1.0 - f_up, y_up))
        h_00 = float(_trapezoid(f_lo, y_lo))
        h_01 = float(_trapezoid(f_up, y_up))

    # ht = light penetrating up, hb = heavy penetrating down
    result['ht_int'] = h_01
    result['hb_int'] = h_10
    result['h_total_int'] = h_01 + h_10
    result['h_10'] = h_10
    result['h_11'] = h_11
    result['h_00'] = h_00
    result['h_01'] = h_01

    # --- Youngs' W integral and mixing efficiency ---
    youngs_integrand = f_avg * (1.0 - f_avg)
    youngs_W = float(_trapezoid(youngs_integrand, y_centers))

    mixed_mask = (f_avg > 0.05) & (f_avg < 0.95)
    if np.any(mixed_mask):
        mixed_indices = np.where(mixed_mask)[0]
        mix_width = y_centers[mixed_indices[-1]] - y_centers[mixed_indices[0]]
        perfect_mixing = 0.25 * mix_width if mix_width > 0 else 0
        mixing_efficiency = min(1.0, youngs_W / perfect_mixing) if perfect_mixing > 0 else 0
    else:
        mix_width = 0.0
        mixing_efficiency = 0.0

    result['youngs_W'] = youngs_W
    result['mixing_efficiency'] = mixing_efficiency

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Mixing + Fractal Dimension (all timesteps)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(case_dir, mesh, phys, rt_physics, interface_times,
               analysis_dir) -> pd.DataFrame:
    """Process all timesteps: mixing thickness + 3D fractal dimension.

    Returns DataFrame with temporal results.
    """
    H = phys['H']
    H0 = phys['H0']
    is_decomposed = os.path.isdir(os.path.join(case_dir, 'processor0'))

    y_centers = np.linspace(mesh.y_min + mesh.dy / 2,
                            mesh.y_max - mesh.dy / 2, mesh.ny)

    # Create output dirs
    for subdir in ['fractal', 'summary']:
        os.makedirs(os.path.join(analysis_dir, subdir), exist_ok=True)

    use_numba = check_numba_available()
    print(f"  Numba acceleration: {'available' if use_numba else 'not available (using pure Python)'}")

    results = []

    for i, (t, vtk_path) in enumerate(interface_times):
        t0 = time_mod.time()
        print(f"\n[{i+1}/{len(interface_times)}] t = {t:.2f} s  "
              f"(tau = {float(rt_physics.nondim_time(t)):.4f})")

        tau = float(rt_physics.nondim_time(t))
        row = {'time': t, 'tau': tau}

        # ── Skip t=0 ──
        if t < 0.01:
            row.update({
                'ht_stat': 0.0, 'hb_stat': 0.0, 'h_total_stat': 0.0,
                'ht_int': 0.0, 'hb_int': 0.0, 'h_total_int': 0.0,
                'h_10': 0.0, 'h_11': H / 2.0, 'h_00': H / 2.0, 'h_01': 0.0,
                'youngs_W': 0.0, 'mixing_efficiency': 0.0,
                'fractal_dim': np.nan, 'fd_error': np.nan, 'fd_r_squared': np.nan,
                'n_triangles': 0, 'surface_area': 0.0,
            })
            results.append(row)
            print("  t=0: flat interface, skipping")
            continue

        # ── Mixing thickness from volume field ──
        mix = None
        try:
            if is_decomposed:
                time_dir = reconstruct_timestep(case_dir, t)
            else:
                t_str = str(int(t)) if t == int(t) else f"{t}"
                time_dir = os.path.join(case_dir, t_str)

            alpha_3d = read_openfoam_alpha_3d(time_dir, mesh)
            if alpha_3d is not None:
                mix = compute_mixing_3d(alpha_3d, y_centers, H0, H)
                del alpha_3d  # Free memory
            else:
                print("  WARNING: could not read alpha.water")
        except Exception as e:
            print(f"  WARNING: mixing computation failed: {e}")

        if mix is None:
            mix = {
                'ht_stat': np.nan, 'hb_stat': np.nan, 'h_total_stat': np.nan,
                'ht_int': np.nan, 'hb_int': np.nan, 'h_total_int': np.nan,
                'h_10': np.nan, 'h_11': np.nan, 'h_00': np.nan, 'h_01': np.nan,
                'youngs_W': np.nan, 'mixing_efficiency': np.nan,
            }
        row.update(mix)

        # ── 3D fractal dimension from isosurface ──
        fd_dim, fd_err, fd_r2 = np.nan, np.nan, np.nan
        n_tri, surf_area = 0, 0.0

        try:
            tri_mesh = load_mesh(vtk_path)
            n_tri = tri_mesh.n_triangles
            surf_area = tri_mesh.surface_area

            if n_tri >= 10:
                # Use adaptive parameters from surface features
                features = extract_surface_features(tri_mesh, compute_curvature=False)
                params = suggest_parameters(features)

                fd_result = compute_fractal_dimension_3d(
                    tri_mesh,
                    initial_delta=params['initial_delta'],
                    delta_factor=params['delta_factor'],
                    num_steps=params['num_steps'],
                    min_delta=tri_mesh.bbox.characteristic_length / 500,
                )
                fd_dim = fd_result.dimension
                fd_err = fd_result.std_error
                fd_r2 = fd_result.r_squared

                # Save box-counting data
                bc_df = pd.DataFrame({
                    'cube_size': fd_result.deltas,
                    'cube_count': fd_result.n_boxes,
                    'cube_size_nondim': np.array(fd_result.deltas) / H,
                })
                bc_file = os.path.join(analysis_dir, 'fractal',
                                       f'boxcount_3d_t{t:.2f}.csv')
                bc_df.to_csv(bc_file, index=False)
            else:
                print(f"  WARNING: only {n_tri} triangles, skipping fractal")
        except Exception as e:
            print(f"  WARNING: fractal dimension failed: {e}")

        row['fractal_dim'] = fd_dim
        row['fd_error'] = fd_err
        row['fd_r_squared'] = fd_r2
        row['n_triangles'] = n_tri
        row['surface_area'] = surf_area

        elapsed = time_mod.time() - t0
        print(f"  h_int={row.get('h_total_int', np.nan):.5f}  "
              f"D={fd_dim:.4f}+/-{fd_err:.4f}  R2={fd_r2:.4f}  "
              f"tri={n_tri}  ({elapsed:.1f}s)")

        results.append(row)

    # Build DataFrame and add nondimensional columns
    df = pd.DataFrame(results)
    for col in ['ht_stat', 'hb_stat', 'h_total_stat',
                'ht_int', 'hb_int', 'h_total_int',
                'h_10', 'h_11', 'h_00', 'h_01']:
        if col in df.columns:
            df[col + '_nondim'] = df[col] / H

    csv_file = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nTemporal results saved to {csv_file}")

    _plot_phase1(df, phys, rt_physics,
                 os.path.join(analysis_dir, 'summary'))

    return df


def _plot_phase1(df, phys, rt_physics, plot_dir):
    """Generate Phase 1 plots."""
    print("\nGenerating Phase 1 plots...")
    dfv = df[df['time'] > 0.01].copy()
    H = phys['H']

    if len(dfv) < 2:
        print("  Not enough data points for plots")
        return

    # ── Mixing thickness evolution ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if 'h_total_stat' in dfv.columns:
        ax1.plot(dfv['time'], dfv['h_total_stat'], 'k:', lw=2, label='Statistical')
    if 'h_total_int' in dfv.columns:
        ax1.plot(dfv['time'], dfv['h_total_int'], 'm-', lw=2, label='Integral (Dalziel)')
        ax1.plot(dfv['time'], dfv['ht_int'], 'r--', lw=1.5, label='Spike (int)')
        ax1.plot(dfv['time'], dfv['hb_int'], 'g--', lw=1.5, label='Bubble (int)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mixing thickness (m)')
    ax1.set_title(f'3D Mixing Layer Evolution ({phys["NX"]}x{phys["NY"]}x{phys["NZ"]})')
    ax1.legend(fontsize=9)
    ax1.grid(True)

    if 'h_total_int_nondim' in dfv.columns:
        ax2.plot(dfv['tau'], dfv['h_total_int_nondim'], 'm-', lw=2, label='Integral (total)')
        ax2.plot(dfv['tau'], dfv['ht_int_nondim'], 'r--', lw=1.5, label='Spike')
        ax2.plot(dfv['tau'], dfv['hb_int_nondim'], 'g--', lw=1.5, label='Bubble')
    ax2.set_xlabel(r'$\tau = t\sqrt{Ag/H}$')
    ax2.set_ylabel(r'$h/H$')
    ax2.set_title('Mixing Layer (nondimensional)')
    ax2.legend(fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mixing_evolution_3d.png'), dpi=300)
    plt.close()

    # ── Self-similar scaling: h/H vs tau^2 ──
    fig, ax = plt.subplots(figsize=(8, 6))
    if 'h_total_int_nondim' in dfv.columns:
        ax.plot(dfv['tau']**2, dfv['h_total_int_nondim'], 'm^-', ms=4, lw=1.5,
                label=r'$h/H$ vs $\tau^2$ (integral)')

        # Fit alpha_RT in late regime
        late = dfv[dfv['tau'] > dfv['tau'].max() * 0.3]
        if len(late) > 3 and not late['h_total_int_nondim'].isna().all():
            coeffs = np.polyfit(late['tau']**2, late['h_total_int_nondim'], 1)
            alpha_RT = coeffs[0]
            tau2_fit = np.linspace(late['tau'].min()**2, late['tau'].max()**2, 50)
            ax.plot(tau2_fit, np.polyval(coeffs, tau2_fit), 'm--', lw=1,
                    label=rf'$\alpha_{{RT}} \approx {alpha_RT:.4f}$')

    ax.set_xlabel(r'$\tau^2$')
    ax.set_ylabel(r'$h/H$')
    ax.set_title('Self-Similar Scaling (3D)')
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'selfsimilar_scaling_3d.png'), dpi=300)
    plt.close()

    # ── Fractal dimension evolution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = dfv['fractal_dim'].notna()
    if valid.any():
        ax.errorbar(dfv.loc[valid, 'tau'], dfv.loc[valid, 'fractal_dim'],
                     yerr=dfv.loc[valid, 'fd_error'],
                     fmt='bo-', ms=5, lw=1.5, capsize=3,
                     label='3D fractal dimension')
        ax.axhline(y=2.0, color='gray', ls='--', alpha=0.5, label='D=2 (smooth surface)')
        ax.set_xlabel(r'$\tau = t\sqrt{Ag/H}$')
        ax.set_ylabel('Fractal Dimension D')
        ax.set_title(f'3D Fractal Dimension Evolution ({phys["NX"]}x{phys["NY"]}x{phys["NZ"]})')
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_ylim(1.8, 2.8)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fractal_evolution_3d.png'), dpi=300)
    plt.close()

    # ── Dalziel integral measures (4-panel) ──
    if all(c in dfv.columns for c in ['h_10', 'h_11', 'h_00', 'h_01']):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(dfv['tau'], dfv['h_10_nondim'], 'b-', lw=2, label=r'$h_{1,0}/H$')
        axes[0, 0].plot(dfv['tau'], dfv['h_01_nondim'], 'r-', lw=2, label=r'$h_{0,1}/H$')
        axes[0, 0].set_title('Penetration depths')
        axes[0, 0].set_ylabel(r'$h/H$')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(dfv['tau'], dfv['h_11_nondim'], 'b-', lw=2, label=r'$h_{1,1}/H$')
        axes[0, 1].plot(dfv['tau'], dfv['h_00_nondim'], 'r-', lw=2, label=r'$h_{0,0}/H$')
        axes[0, 1].set_title('Retention in own region')
        axes[0, 1].set_ylabel(r'$h/H$')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Conservation checks
        H_half = H / 2.0
        lower_sum = dfv['h_10'] + dfv['h_00']
        upper_sum = dfv['h_11'] + dfv['h_01']
        axes[1, 0].plot(dfv['tau'], lower_sum / H_half, 'b-', lw=2,
                        label=r'$(h_{1,0}+h_{0,0})/(H/2)$')
        axes[1, 0].plot(dfv['tau'], upper_sum / H_half, 'r-', lw=2,
                        label=r'$(h_{1,1}+h_{0,1})/(H/2)$')
        axes[1, 0].axhline(y=1.0, color='k', ls='--', alpha=0.5)
        axes[1, 0].set_title('Conservation check (should = 1)')
        axes[1, 0].set_ylabel('Ratio')
        axes[1, 0].set_xlabel(r'$\tau$')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Youngs W and efficiency
        axes[1, 1].plot(dfv['tau'], dfv['youngs_W'], 'g-', lw=2, label="Young's W")
        ax_eff = axes[1, 1].twinx()
        ax_eff.plot(dfv['tau'], dfv['mixing_efficiency'], 'k--', lw=1.5,
                    label='Mixing efficiency')
        axes[1, 1].set_xlabel(r'$\tau$')
        axes[1, 1].set_ylabel("Young's W (m)")
        ax_eff.set_ylabel('Mixing efficiency')
        axes[1, 1].set_title("Young's integral width & efficiency")
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax_eff.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2)
        axes[1, 1].grid(True)

        plt.suptitle(f'Dalziel Integral Measures — 3D ({phys["NX"]}x{phys["NY"]}x{phys["NZ"]})',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'dalziel_integrals_3d.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    # ── Surface complexity evolution ──
    if 'n_triangles' in dfv.columns and 'surface_area' in dfv.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(dfv['tau'], dfv['n_triangles'], 'b.-', ms=4)
        ax1.set_xlabel(r'$\tau$')
        ax1.set_ylabel('Number of triangles')
        ax1.set_title('Isosurface triangle count')
        ax1.grid(True)

        ax2.plot(dfv['tau'], dfv['surface_area'], 'r.-', ms=4)
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel('Surface area (m²)')
        ax2.set_title('Isosurface area evolution')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'surface_complexity_3d.png'), dpi=300)
        plt.close()

    print("  Phase 1 plots complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Multifractal Analysis (selected timesteps)
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(case_dir, phys, rt_physics, interface_times, analysis_dir,
               mf_times, df):
    """Run 3D multifractal analysis at selected timesteps."""
    mf_output_dir = os.path.join(analysis_dir, 'multifractal')
    os.makedirs(mf_output_dir, exist_ok=True)

    H = phys['H']
    max_time = df['time'].max()
    mf_times = [t for t in mf_times if t <= max_time + 0.5]
    if not mf_times:
        print("WARNING: No multifractal times within data range. Skipping Phase 2.")
        return

    # Map target times to actual interface VTK files
    time_vtk = {t: path for t, path in interface_times}
    mf_files = {}
    for target_t in mf_times:
        best_t = min(time_vtk.keys(), key=lambda t: abs(t - target_t))
        mf_files[best_t] = time_vtk[best_t]
        print(f"  MF target t={target_t:.1f} -> actual t={best_t:.2f}")

    q_values = np.arange(-5, 5.1, 0.5)
    analyzer = MultifractalAnalyzer3D(measure='area', verbose=True)

    mf_results = []
    for t in sorted(mf_files.keys()):
        vtk_path = mf_files[t]
        tau = float(rt_physics.nondim_time(t))
        print(f"\n  Multifractal analysis at t={t:.2f} (tau={tau:.4f})...")

        try:
            tri_mesh = load_mesh(vtk_path)
            if tri_mesh.n_triangles < 50:
                print(f"  WARNING: only {tri_mesh.n_triangles} triangles, skipping")
                continue

            result = analyzer.compute_multifractal_spectrum(
                tri_mesh,
                q_values=q_values,
                rt_physics=rt_physics,
            )

            # Save per-timestep results
            label = f't{t:.2f}_'
            analyzer.save_results(result, mf_output_dir, label=label)

            # Store for plotting
            mf_entry = {
                'time': t, 'tau': tau,
                'q_values': result.q_values,
                'tau_q': result.tau,
                'Dq': result.Dq,
                'alpha': result.alpha,
                'f_alpha': result.f_alpha,
                'r_squared': result.r_squared,
                'D0': result.D0,
                'D1': result.D1,
                'D2': result.D2,
                'alpha_width': result.alpha_width,
                'degree_multifractality': result.degree_multifractality,
            }
            mf_results.append(mf_entry)

            print(f"  D0={result.D0:.4f}  D1={result.D1:.4f}  D2={result.D2:.4f}  "
                  f"Δα={result.alpha_width:.4f}")

        except Exception as e:
            print(f"  WARNING: multifractal failed at t={t:.2f}: {e}")
            import traceback
            traceback.print_exc()

    if mf_results:
        _plot_phase2(mf_results, phys, rt_physics, df, mf_output_dir)

        # Summary CSV
        summary_rows = []
        for r in mf_results:
            summary_rows.append({
                'time': r['time'], 'tau': r['tau'],
                'D0': r['D0'], 'D1': r['D1'], 'D2': r['D2'],
                'alpha_width': r['alpha_width'],
                'degree_multifractality': r['degree_multifractality'],
            })
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(mf_output_dir, 'multifractal_summary_3d.csv'), index=False)
        print(f"\n  Multifractal summary saved to {mf_output_dir}")


def _plot_phase2(mf_results, phys, rt_physics, df, mf_output_dir):
    """Generate Phase 2 plots."""
    print("\nGenerating Phase 2 plots...")
    cmap = get_cmap('viridis')
    times_mf = sorted([r['time'] for r in mf_results])
    t_min, t_max = min(times_mf), max(times_mf)

    # ── f(alpha) spectra ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for res in sorted(mf_results, key=lambda x: x['time']):
        t = res['time']
        tau_val = res['tau']
        color = cmap((t - t_min) / (t_max - t_min)) if t_max > t_min else cmap(0.5)
        valid = ~np.isnan(res['alpha']) & ~np.isnan(res['f_alpha'])
        if np.sum(valid) < 2:
            continue
        a = res['alpha'][valid]
        fa = res['f_alpha'][valid]
        sort_idx = np.argsort(a)
        ax.plot(a[sort_idx], fa[sort_idx], '-o', color=color, ms=3, lw=1.5,
                label=rf'$\tau$={tau_val:.3f}')
    ax.set_xlabel(r'$\alpha$', fontsize=14)
    ax.set_ylabel(r'$f(\alpha)$', fontsize=14)
    ax.set_title(r'3D Multifractal Spectrum $f(\alpha)$', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'f_alpha_evolution_3d.png'), dpi=300)
    plt.close()

    # ── D(q) curves ──
    fig, ax = plt.subplots(figsize=(10, 7))
    for res in sorted(mf_results, key=lambda x: x['time']):
        t = res['time']
        tau_val = res['tau']
        color = cmap((t - t_min) / (t_max - t_min)) if t_max > t_min else cmap(0.5)
        valid = ~np.isnan(res['Dq'])
        ax.plot(res['q_values'][valid], res['Dq'][valid], '-o', color=color,
                ms=3, lw=1.5, label=rf'$\tau$={tau_val:.3f}')
    ax.set_xlabel('q', fontsize=14)
    ax.set_ylabel('D(q)', fontsize=14)
    ax.set_title('3D Generalized Dimensions D(q)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'Dq_evolution_3d.png'), dpi=300)
    plt.close()

    # ── D0, D1, D2 vs tau ──
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_vals = [r['tau'] for r in mf_results]
    D0 = [r['D0'] for r in mf_results]
    D1 = [r['D1'] for r in mf_results]
    D2 = [r['D2'] for r in mf_results]

    ax.plot(tau_vals, D0, 'bo-', ms=6, lw=2, label=r'$D_0$ (capacity)')
    ax.plot(tau_vals, D1, 'rs-', ms=6, lw=2, label=r'$D_1$ (information)')
    ax.plot(tau_vals, D2, 'gd-', ms=6, lw=2, label=r'$D_2$ (correlation)')
    ax.axhline(y=2.0, color='gray', ls='--', alpha=0.5, label='D=2')
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Dimension', fontsize=14)
    ax.set_title('3D Generalized Dimensions vs. Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'D012_vs_tau_3d.png'), dpi=300)
    plt.close()

    # ── Multifractality measures ──
    fig, ax = plt.subplots(figsize=(10, 6))
    aw = [r['alpha_width'] for r in mf_results]
    dm = [r['degree_multifractality'] for r in mf_results]
    ax.plot(tau_vals, aw, 'ms-', ms=6, lw=2, label=r'$\Delta\alpha$ (spectrum width)')
    ax.plot(tau_vals, dm, 'cd-', ms=6, lw=2,
            label=r'$D_{-5} - D_5$ (multifractality degree)')
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('3D Multifractality Measures vs. Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'multifractality_measures_3d.png'), dpi=300)
    plt.close()

    # ── Phase portrait: D0 vs h/H ──
    if 'h_total_int_nondim' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for res in mf_results:
            # Find closest h/H from temporal data
            t = res['time']
            closest_idx = (df['time'] - t).abs().idxmin()
            h_nondim = df.loc[closest_idx, 'h_total_int_nondim']
            tau_val = res['tau']
            ax.plot(h_nondim, res['D0'], 'bo', ms=8)
            ax.annotate(rf'$\tau$={tau_val:.2f}', (h_nondim, res['D0']),
                        textcoords='offset points', xytext=(5, 5), fontsize=8)
        ax.set_xlabel(r'$h/H$', fontsize=14)
        ax.set_ylabel(r'$D_0$', fontsize=14)
        ax.set_title('Phase portrait: 3D Fractal Dimension vs Mixing Extent', fontsize=14)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(mf_output_dir, 'phase_portrait_D0_h_3d.png'), dpi=300)
        plt.close()

    print("  Phase 2 plots complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='3D Rayleigh-Taylor Simulation Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 analyze_rt_3d.py /path/to/3D/case
  python3 analyze_rt_3d.py /path/to/3D/case --mf-times 4,8,12,16
  python3 analyze_rt_3d.py /path/to/3D/case --skip-phase2
        """)
    parser.add_argument('case_dir', help='OpenFOAM 3D case directory')
    parser.add_argument('--mf-times', type=str, default='2,4,6,8,10,12,14,16,18',
                        help='Comma-separated times for multifractal analysis (default: 2,4,...,18)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 (use existing temporal_results.csv)')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 (multifractal analysis)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: {case}/analysis)')

    args = parser.parse_args()
    case_dir = os.path.abspath(args.case_dir)

    if not os.path.isdir(case_dir):
        print(f"ERROR: case directory not found: {case_dir}")
        sys.exit(1)

    mf_times = [float(t) for t in args.mf_times.split(',')]
    analysis_dir = args.output_dir or os.path.join(case_dir, 'analysis')

    # ── Header ──
    print("=" * 70)
    print("3D RAYLEIGH-TAYLOR ANALYSIS TOOL")
    print("=" * 70)
    print(f"Case: {case_dir}")

    # ── Read mesh ──
    mesh = read_mesh_info(case_dir)
    if mesh is None:
        print("ERROR: Could not read blockMeshDict")
        sys.exit(1)

    print(f"Mesh: {mesh.nx} x {mesh.ny} x {mesh.nz} = "
          f"{mesh.nx * mesh.ny * mesh.nz:,} cells")
    print(f"Domain: [{mesh.x_min}, {mesh.x_max}] x "
          f"[{mesh.y_min}, {mesh.y_max}] x "
          f"[{mesh.z_min}, {mesh.z_max}]")
    print(f"Cell size: dx={mesh.dx*1000:.3f}mm, dy={mesh.dy*1000:.3f}mm, "
          f"dz={mesh.dz*1000:.3f}mm")

    # ── Physics ──
    phys = get_physics_3d(case_dir, mesh)
    print(f"\nPhysics: A={phys['A']:.6f}, g={phys['g']:.2f} m/s²")
    print(f"  rho_heavy={phys['rho_heavy']:.2f}, rho_light={phys['rho_light']:.2f}")
    print(f"  H={phys['H']:.3f}m, L={phys['L']:.3f}m, W={phys['W']:.3f}m")
    print(f"  H0={phys['H0']:.3f}m (initial interface)")

    rt_physics = RTPhysics3D(
        A=phys['A'], g=phys['g'],
        H=phys['H'], L=phys['L'], W=phys['W'],
    )
    print(f"  tau_factor={rt_physics.tau_factor:.6f} (tau = t * {rt_physics.tau_factor:.6f})")
    print(f"  t_max=20s -> tau_max={float(rt_physics.nondim_time(20)):.4f}")

    # ── Find timesteps ──
    interface_times = find_interface_timesteps(case_dir)
    print(f"\nFound {len(interface_times)} timesteps with isosurface VTK")

    if not interface_times:
        print("ERROR: No extractInterface postProcessing found")
        sys.exit(1)

    # ── Phase 1 ──
    df = None
    if not args.skip_phase1:
        print("\n" + "─" * 70)
        print("PHASE 1: Mixing Thickness + 3D Fractal Dimension")
        print("─" * 70)
        df = run_phase1(case_dir, mesh, phys, rt_physics,
                        interface_times, analysis_dir)
    else:
        csv_file = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
        if os.path.isfile(csv_file):
            df = pd.read_csv(csv_file)
            print(f"\nLoaded existing results from {csv_file} ({len(df)} rows)")
        else:
            print(f"ERROR: --skip-phase1 but no {csv_file} found")
            sys.exit(1)

    # ── Phase 2 ──
    if not args.skip_phase2:
        print("\n" + "─" * 70)
        print("PHASE 2: 3D Multifractal Analysis")
        print("─" * 70)
        run_phase2(case_dir, phys, rt_physics, interface_times,
                   analysis_dir, mf_times, df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results in: {analysis_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
