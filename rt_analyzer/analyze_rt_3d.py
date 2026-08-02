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
    result['_f_avg'] = f_avg

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Mixing + Fractal Dimension (all timesteps)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_projected_D(tri_mesh, phys, mesh, analysis_dir, t):
    """Compute fractal dimension of the isosurface projected onto the x-y plane.

    Projects all triangle edges onto the x-y plane (dropping z, the depth
    axis), giving the silhouette seen looking through the tank from the
    front.  This is the "shadow" dimension comparable to Linden, Redondo
    & Youngs (1994), who projected the 3D isosurface onto a viewing plane.

    Gravity is along y; x is horizontal width; z is depth (collapsed).

    Returns (D_projected, R²).
    """
    verts = tri_mesh.vertices                  # (N, 3) — x, y, z
    tris  = tri_mesh.triangles                 # (M, 3) — vertex indices

    # Extract unique edges and project onto x-y (columns 0, 1)
    edge_set = set()
    for tri in tris:
        for i, j in [(0,1), (1,2), (2,0)]:
            e = (min(tri[i], tri[j]), max(tri[i], tri[j]))
            edge_set.add(e)

    # Build projected 2D segments [(x1,y1), (x2,y2)]
    segments = []
    for i, j in edge_set:
        p1 = (float(verts[i, 0]), float(verts[i, 1]))   # (x, y)
        p2 = (float(verts[j, 0]), float(verts[j, 1]))
        segments.append((p1, p2))

    if len(segments) < 10:
        return np.nan, np.nan

    # Domain bounds in the x-y projection
    x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
    z_min, z_max = float(verts[:, 1].min()), float(verts[:, 1].max())
    L_proj = max(x_max - x_min, z_max - z_min)

    # Box sizes: geometric ladder from L_proj/2 down to 2*max(dx,dy)
    min_box = max(mesh.dx, mesh.dy) * 2
    max_box = L_proj / 2
    factor  = 1.5
    box_sizes = []
    bs = max_box
    while bs >= min_box:
        box_sizes.append(bs)
        bs /= factor
    if len(box_sizes) < 4:
        return np.nan, np.nan

    # Count occupied boxes at each scale (single offset for speed)
    counts = []
    for bs in box_sizes:
        occupied = set()
        for (x1, z1), (x2, z2) in segments:
            # Rasterise segment: mark boxes at both endpoints and midpoint
            for frac in [0.0, 0.5, 1.0]:
                px = x1 + frac * (x2 - x1)
                pz = z1 + frac * (z2 - z1)
                bx = int((px - x_min) / bs)
                bz = int((pz - z_min) / bs)
                occupied.add((bx, bz))
        counts.append(len(occupied))

    # Log-log fit
    log_eps = np.log(box_sizes)
    log_N   = np.log(np.array(counts, dtype=float))
    slope, intercept, r_value, _, _ = stats.linregress(log_eps, log_N)
    D_proj = -slope
    R2     = r_value ** 2

    return D_proj, R2


def compute_molecular_mixing_3d(alpha_3d, y_centers):
    """Molecular mixing fraction, Dalziel, Linden & Youngs (1999) §7.2.

    Four quantities, transcribed from the paper (pp. 41-44). C is the
    concentration; here the heavy-fluid volume fraction alpha plays that role.

      Eq (10)  theta(z)    = <<C(1-C)>> / (<<C>> <<1-C>>)
               double overbar = average over a horizontal PLANE, i.e. along and
               across the domain (our x and z). A profile in height, not a scalar.

      Eq (11)  theta_hat(z) = < bar{C(1-C)} / (bar{C} bar{1-C}) >
               along-tank (x) averages only, then averaged across the tank (z).
               Note this averages the RATIO, where (10) is a ratio of averages.

      Eq (12)  Theta      = int bar{C(1-C)} dz / int bar{C} bar{1-C} dz
               global mixing/horizontal homogeneity over the ENTIRE domain
               (-H/2 to H/2, not just the mixing layer), after Linden et al. (1994).

      Eq (13)  Theta_hat  = < int bar{C(1-C)} dz / int bar{C} bar{1-C} dz >
               the same, but averaging the QUOTIENT of the integrals across the
               tank rather than taking a quotient of averaged integrals.

    theta = 1 means concentration uniform across the plane (fully mixed);
    0 means unmixed. Dalziel reports Theta_hat -> 0.8 asymptotically for
    well-resolved self-similar mixing (Linden et al. 1994), and notes that
    simulations underestimate it early on for want of fine-scale resolution.

    NOTE OF SCOPE: those target values are at Dalziel's single Atwood number,
    A = 2.1e-3. For the Atwood study (A = 0.04-0.60) compare against Banerjee,
    Kraft & Andrews (2010) instead.

    alpha_3d is indexed (nz, ny, nx); y_centers indexes axis 1.
    """
    C = alpha_3d
    P = C * (1.0 - C)

    # --- Eq (10): plane averages over both horizontal directions ---
    C_pl = C.mean(axis=(0, 2))                     # (ny,)
    P_pl = P.mean(axis=(0, 2))
    den_pl = C_pl * (1.0 - C_pl)
    theta = np.divide(P_pl, den_pl, out=np.zeros_like(P_pl), where=den_pl > 0)

    # --- Eq (11): along-tank averages, then average the ratio across the tank ---
    C_x = C.mean(axis=2)                           # (nz, ny)
    P_x = P.mean(axis=2)
    den_x = C_x * (1.0 - C_x)
    ratio = np.divide(P_x, den_x, out=np.zeros_like(P_x), where=den_x > 0)
    theta_hat = ratio.mean(axis=0)                 # (ny,)

    # --- Eq (12): global, ratio of integrals over the full domain ---
    num = float(_trapezoid(P_pl, y_centers))
    den = float(_trapezoid(den_pl, y_centers))
    # den = 0 means every plane is pure fluid (C_bar is 0 or 1 everywhere), i.e.
    # a sharp, horizontally flat interface: unmixed, so Theta = 0 rather than 0/0.
    Theta = num / den if den > 0 else 0.0

    # --- Eq (13): quotient of integrals per slice, then averaged across tank ---
    num_s = np.array([float(_trapezoid(P_x[k], y_centers)) for k in range(P_x.shape[0])])
    den_s = np.array([float(_trapezoid(den_x[k], y_centers)) for k in range(den_x.shape[0])])
    q = np.divide(num_s, den_s, out=np.zeros_like(num_s), where=den_s > 0)
    Theta_hat = float(q.mean())   # slices with den = 0 are unmixed, contributing 0

    return {'theta': theta, 'theta_hat': theta_hat,
            'Theta': Theta, 'Theta_hat': Theta_hat}


def read_velocity_fast(time_dir, mesh):
    """Read U into (nz, ny, nx, 3) with a vectorised parse.

    The line-by-line reader used by Phase 3 costs minutes per snapshot at 20M
    cells, which is prohibitive when every timestep needs it. Falls back to that
    reader if the fast path fails.
    """
    U_file = os.path.join(time_dir, 'U')
    if not os.path.exists(U_file):
        return None
    n_cells = mesh.nx * mesh.ny * mesh.nz
    try:
        with open(U_file) as f:
            txt = f.read()
        i = txt.index('(', txt.index('nonuniform'))
        end = txt.index('boundaryField') if 'boundaryField' in txt else len(txt)
        j = txt.rindex(')', 0, end)
        block = txt[i + 1:j].translate({ord('('): ' ', ord(')'): ' '})
        del txt
        arr = np.fromstring(block, sep=' ')
        del block
        if arr.size != 3 * n_cells:
            raise ValueError(f"parsed {arr.size} values, expected {3 * n_cells}")
        return arr.reshape(mesh.nz, mesh.ny, mesh.nx, 3)
    except Exception as e:
        print(f"  (fast U read failed: {e}; falling back)")
        return read_openfoam_velocity_3d(time_dir, mesh)


def compute_energy_budget_3d(alpha_3d, U_3d, y_centers, phys, mesh):
    """Resolved-scale energy budget for Rayleigh-Taylor.

    PE  = sum(rho g y dV)   total potential energy
    BPE = PE of the adiabatically re-sorted minimum-PE state (Winters et al.
          1995): densities sorted heaviest-to-lightest and restacked bottom-up.
          The rise in BPE measures irreversible mixing.
    APE = PE - BPE, the part available to drive motion
    KE  = sum(0.5 rho |u|^2 dV)

    The caller forms the cumulative dissipation from PE(0) - PE(t) - KE(t).
    This budget is dominated by the largest scales, so unlike the enstrophy
    integral (whose spectral density ~ k^2 E(k) peaks at the grid scale) it does
    not require the dissipation range to be resolved.

    alpha_3d is the heavy-fluid volume fraction, indexed (nz, ny, nx).
    """
    g = phys['g']
    rho_h, rho_l = phys['rho_heavy'], phys['rho_light']
    dV = mesh.dx * mesh.dy * mesh.dz

    rho = rho_l + (rho_h - rho_l) * alpha_3d
    y = y_centers[None, :, None]

    PE = float(np.sum(rho * y) * g * dV)

    # Minimum-PE rearrangement: heaviest fluid into the lowest cells. On a
    # uniform mesh this is a sort against cell-centre heights, each repeated
    # nx*nz times.
    rho_sorted = np.sort(rho, axis=None)[::-1]
    y_stack = np.repeat(y_centers, mesh.nx * mesh.nz)
    BPE = float(np.sum(rho_sorted * y_stack) * g * dV)
    del rho_sorted, y_stack

    out = {'PE': PE, 'BPE': BPE, 'APE': PE - BPE}
    out['KE'] = (float(0.5 * np.sum(rho * np.sum(U_3d * U_3d, axis=-1)) * dV)
                 if U_3d is not None else np.nan)
    return out


def run_phase1(case_dir, mesh, phys, rt_physics, interface_times,
               analysis_dir, do_energy=True) -> pd.DataFrame:
    """Process all timesteps: mixing thickness + 3D fractal dimension.

    Returns DataFrame with temporal results.
    """
    H = phys['H']
    H0 = phys['H0']
    is_decomposed = os.path.isdir(os.path.join(case_dir, 'processor0'))

    y_centers = np.linspace(mesh.y_min + mesh.dy / 2,
                            mesh.y_max - mesh.dy / 2, mesh.ny)

    # Create output dirs
    for subdir in ['fractal', 'summary', 'profiles']:
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
                # unmixed: C is 0 or 1 everywhere, so C(1-C) = 0 identically
                'Theta': 0.0, 'Theta_hat': 0.0,
            })
            if do_energy:
                # Reference state: heavy above H0, light below, at rest. Computed
                # analytically rather than read, since the sub-grid perturbation
                # changes PE by O(sigma/H) ~ 1e-4 and this fixes the datum that
                # every later PE_released is measured against.
                dV = mesh.dx * mesh.dy * mesh.dz
                ncol = mesh.nx * mesh.nz
                rho_col = np.where(y_centers >= H0, phys['rho_heavy'], phys['rho_light'])
                PE0 = float(np.sum(rho_col * y_centers) * phys['g'] * dV * ncol)
                # Sorted (minimum-PE) state has the HEAVY fluid at the bottom —
                # the inverse of the unstable initial stacking, so BPE0 << PE0
                # and APE0 = PE0 - BPE0 is the full reservoir driving the flow.
                rho_sorted0 = np.sort(rho_col)[::-1]
                BPE0 = float(np.sum(rho_sorted0 * y_centers) * phys['g'] * dV * ncol)
                row.update({'PE': PE0, 'BPE': BPE0, 'APE': PE0 - BPE0, 'KE': 0.0})

            # Save flat profile at t=0
            f_avg_0 = np.where(y_centers >= H0, 1.0, 0.0)
            prof_df = pd.DataFrame({'y_m': y_centers, 'alpha_mean': f_avg_0})
            prof_df.to_csv(os.path.join(analysis_dir, 'profiles',
                                        'alpha_profile_t00000.csv'), index=False)
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
                try:
                    mm = compute_molecular_mixing_3d(alpha_3d, y_centers)
                    row['Theta'] = mm['Theta']
                    row['Theta_hat'] = mm['Theta_hat']
                    t_ms = int(round(t * 1000))
                    pd.DataFrame({'y_m': y_centers,
                                  'theta': mm['theta'],
                                  'theta_hat': mm['theta_hat']}).to_csv(
                        os.path.join(analysis_dir, 'profiles',
                                     f'theta_profile_t{t_ms:05d}.csv'), index=False)
                except Exception as e:
                    print(f"  WARNING: molecular mixing failed: {e}")
                if do_energy:
                    try:
                        U_3d = read_velocity_fast(time_dir, mesh)
                        energy = compute_energy_budget_3d(alpha_3d, U_3d,
                                                          y_centers, phys, mesh)
                        del U_3d
                        row.update(energy)
                    except Exception as e:
                        print(f"  WARNING: energy budget failed: {e}")
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

        # Save horizontally-averaged profile <F>(y)
        f_avg = mix.pop('_f_avg', None)
        if f_avg is not None:
            t_ms = int(round(t * 1000))
            prof_df = pd.DataFrame({'y_m': y_centers, 'alpha_mean': f_avg})
            prof_file = os.path.join(analysis_dir, 'profiles',
                                     f'alpha_profile_t{t_ms:05d}.csv')
            prof_df.to_csv(prof_file, index=False)

        row.update(mix)

        # ── 3D fractal dimension from isosurface ──
        fd_dim, fd_err, fd_r2 = np.nan, np.nan, np.nan
        n_tri, surf_area = 0, 0.0

        try:
            tri_mesh = load_mesh(vtk_path)
            n_tri = tri_mesh.n_triangles
            surf_area = tri_mesh.surface_area

            if n_tri >= 10:
                # Fixed domain-based scales for temporal consistency.
                # max_delta = min(L,W)/2, min_delta = 2*dx, factor = 1.5.
                domain_char = min(phys['L'], phys['W'])
                fixed_initial = domain_char / 2
                fixed_min = max(mesh.dx, mesh.dy, mesh.dz) * 2
                fd_result = compute_fractal_dimension_3d(
                    tri_mesh,
                    initial_delta=fixed_initial,
                    delta_factor=1.5,
                    num_steps=20,
                    min_delta=fixed_min,
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

        # ── Projected fractal dimension (shadow onto x-z plane) ──
        fd_proj, fd_proj_r2 = np.nan, np.nan
        try:
            if n_tri >= 10:
                fd_proj, fd_proj_r2 = _compute_projected_D(
                    tri_mesh, phys, mesh, analysis_dir, t)
        except Exception as e:
            print(f"  WARNING: projected D failed: {e}")

        row['fractal_dim_proj'] = fd_proj
        row['fd_r_squared_proj'] = fd_proj_r2

        elapsed = time_mod.time() - t0
        print(f"  h_int={row.get('h_total_int', np.nan):.5f}  "
              f"D={fd_dim:.4f}+/-{fd_err:.4f}  Dproj={fd_proj:.4f}  "
              f"tri={n_tri}  ({elapsed:.1f}s)")

        results.append(row)

    # Build DataFrame and add nondimensional columns
    df = pd.DataFrame(results)
    for col in ['ht_stat', 'hb_stat', 'h_total_stat',
                'ht_int', 'hb_int', 'h_total_int',
                'h_10', 'h_11', 'h_00', 'h_01']:
        if col in df.columns:
            df[col + '_nondim'] = df[col] / H

    # ── Energy budget: derived quantities ──
    # PE(0) - PE(t) = KE(t) + E_diss(t). E_diss is cumulative dissipation, and
    # its time derivative is the dissipation rate. This route to dissipation is
    # large-scale dominated, so it converges at resolutions where the enstrophy
    # integral (2*nu*Omega) is still grid-determined.
    if {'PE', 'KE'}.issubset(df.columns) and df['PE'].notna().any():
        PE0 = df['PE'].iloc[0]
        BPE0 = df['BPE'].iloc[0] if 'BPE' in df.columns else np.nan
        df['PE_released'] = PE0 - df['PE']
        df['E_diss'] = df['PE_released'] - df['KE']
        APE0 = df['APE'].iloc[0] if 'APE' in df.columns else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            df['diss_fraction'] = df['E_diss'] / df['PE_released']
            df['ke_fraction'] = df['KE'] / df['PE_released']
            # Two mixing efficiencies, differing only in denominator:
            #   mixing_eff_bpe  ΔBPE / PE_released — share of the energy actually
            #                   released that went into irreversible mixing; the
            #                   natural partner of diss_fraction and ke_fraction,
            #                   with which it closes the budget.
            #   eta_dalziel     ΔBPE / |ΔAPE| — Dalziel et al. (2008) Phys. Fluids
            #                   20, 065106, Eq. (23). Quote THIS one against their
            #                   value. Note their eta = 1/2 is the theoretical
            #                   ceiling (BPE cannot exceed the well-mixed state),
            #                   which their experiment attains — not an empirical
            #                   constant other flows should reproduce.
            df['mixing_eff_bpe'] = (df['BPE'] - BPE0) / df['PE_released']
            df['eta_dalziel'] = (df['BPE'] - BPE0) / (APE0 - df['APE']).abs()
        # dissipation rate, one-sided at the ends
        t_arr = df['time'].values.astype(float)
        e_arr = df['E_diss'].values.astype(float)
        if len(t_arr) > 2 and np.isfinite(e_arr).sum() > 2:
            df['diss_rate'] = np.gradient(e_arr, t_arr)
        else:
            df['diss_rate'] = np.nan
        # normalise: rate / (rho_ref * (A g)^{3/2} H^{1/2} * volume)
        rho_ref = 0.5 * (phys['rho_heavy'] + phys['rho_light'])
        vol = phys['L'] * phys['H'] * phys['W']
        scale = rho_ref * (phys['A'] * phys['g']) ** 1.5 * phys['H'] ** 0.5 * vol
        df['diss_rate_star'] = df['diss_rate'] / scale if scale > 0 else np.nan

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

            # Use exact Phase 1 cube sizes so D_q(0) matches box-counting D
            bc_file = os.path.join(analysis_dir, 'fractal',
                                   f'boxcount_3d_t{t:.2f}.csv')
            phase1_cubes = None
            if os.path.isfile(bc_file):
                bc_df = pd.read_csv(bc_file)
                phase1_cubes = bc_df['cube_size'].values
                print(f"  Using Phase 1 scales ({len(phase1_cubes)} sizes from {bc_file})")

            if phase1_cubes is not None:
                result = analyzer.compute_multifractal_spectrum(
                    tri_mesh,
                    q_values=q_values,
                    cube_sizes=phase1_cubes,
                    rt_physics=rt_physics,
                )
            else:
                # Fallback: regenerate scales via suggest_parameters()
                features = extract_surface_features(tri_mesh, compute_curvature=False)
                params = suggest_parameters(features)
                result = analyzer.compute_multifractal_spectrum(
                    tri_mesh,
                    q_values=q_values,
                    max_delta=params['initial_delta'],
                    delta_factor=params['delta_factor'],
                    num_scales=params['num_steps'],
                    min_delta=tri_mesh.bbox.characteristic_length / 500,
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
# Phase 3: Concentration Spectrum + Velocity Statistics (selected timesteps)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_horizontal_spectrum_3d(alpha_3d, y_centers, x_coords, z_coords,
                                   y0, H, dx, dz):
    """Dalziel-style horizontal concentration spectrum from 3D field.

    1. Define slab: y0 - 0.1*H <= y <= y0
    2. Average F over slab rows → F_bar(x, z)
    3. 2D FFT in (x, z), power |F_hat|^2
    4. Radial average in (kx, kz) → P(k_h)
    5. Fit power law over specified bands

    Returns dict with spectrum arrays and fitted slopes.
    """
    # Slab selection
    slab_lo = y0 - 0.1 * H
    slab_mask = (y_centers >= slab_lo) & (y_centers <= y0)
    n_slab = int(slab_mask.sum())
    if n_slab < 2:
        return None

    # Average over slab rows: alpha_3d is (nz, ny, nx)
    f_slab = np.mean(alpha_3d[:, slab_mask, :], axis=1)  # shape (nz, nx)

    # Subtract mean to get fluctuation
    f_slab = f_slab - np.mean(f_slab)

    # 2D FFT
    nz, nx = f_slab.shape
    f_hat = np.fft.fft2(f_slab)
    power_2d = np.abs(f_hat) ** 2 / (nx * nz)

    # Wavenumber grids
    kx = np.fft.fftfreq(nx, d=dx)
    kz = np.fft.fftfreq(nz, d=dz)
    KX, KZ = np.meshgrid(kx, kz)
    K_h = np.sqrt(KX**2 + KZ**2)

    # Fundamental wavenumber (use x-direction domain length)
    Lx = nx * dx
    k0 = 1.0 / Lx  # cycles per metre (not 2π/L)

    # Radial binning
    k_max = 0.5 / min(dx, dz)  # Nyquist
    n_bins = min(nx, nz) // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    P_radial = np.zeros(n_bins)
    for ib in range(n_bins):
        mask = (K_h >= k_edges[ib]) & (K_h < k_edges[ib + 1])
        if mask.any():
            P_radial[ib] = np.mean(power_2d[mask])

    # Normalise wavenumbers by k0
    k_norm = k_centers / k0

    # Fit power law over bands
    result = {
        'k_over_k0': k_norm,
        'P': P_radial,
        'n_slab_rows': n_slab,
        'k0': k0,
    }

    for band_name, k_lo, k_hi in [('10_25', 10, 25), ('10_50', 10, 50)]:
        band_mask = (k_norm >= k_lo) & (k_norm <= k_hi) & (P_radial > 0)
        if band_mask.sum() >= 3:
            log_k = np.log10(k_norm[band_mask])
            log_P = np.log10(P_radial[band_mask])
            coeffs = np.polyfit(log_k, log_P, 1)
            slope = coeffs[0]
            residuals = log_P - np.polyval(coeffs, log_k)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((log_P - np.mean(log_P))**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            result[f'slope_{band_name}'] = slope
            result[f'r2_{band_name}'] = r2
        else:
            result[f'slope_{band_name}'] = np.nan
            result[f'r2_{band_name}'] = np.nan

    return result


def compute_velocity_statistics_3d(U_3d, y_centers, H0):
    """Compute velocity statistics from 3D velocity field.

    U_3d: shape (nz, ny, nx, 3) — velocity components (ux, uy, uz)
    Returns dict with RMS velocities, anisotropy, centreline values.
    """
    # Horizontally-averaged velocity profiles
    u_mean = np.mean(U_3d, axis=(0, 2))  # shape (ny, 3)

    # Fluctuations
    u_prime = U_3d - u_mean[np.newaxis, :, np.newaxis, :]

    # RMS profiles
    u_rms = np.sqrt(np.mean(u_prime[:, :, :, 0]**2, axis=(0, 2)))
    v_rms = np.sqrt(np.mean(u_prime[:, :, :, 1]**2, axis=(0, 2)))
    w_rms = np.sqrt(np.mean(u_prime[:, :, :, 2]**2, axis=(0, 2)))

    # Centreline index
    j0 = np.argmin(np.abs(y_centers - H0))

    # Anisotropy at centreline
    u_rms_c = float(u_rms[j0])
    v_rms_c = float(v_rms[j0])
    w_rms_c = float(w_rms[j0])
    anisotropy = v_rms_c / u_rms_c if u_rms_c > 1e-12 else np.nan

    # Mass flux <v'F'> would need alpha field too — skip for now

    return {
        'y_centers': y_centers,
        'u_rms': u_rms,
        'v_rms': v_rms,
        'w_rms': w_rms,
        'u_rms_centreline': u_rms_c,
        'v_rms_centreline': v_rms_c,
        'w_rms_centreline': w_rms_c,
        'anisotropy_v_over_u': anisotropy,
    }


def read_openfoam_velocity_3d(time_dir, mesh):
    """Read the OpenFOAM U field and reshape to (nz, ny, nx, 3)."""
    U_file = os.path.join(time_dir, 'U')
    if not os.path.exists(U_file):
        return None
    try:
        n_cells = mesh.nx * mesh.ny * mesh.nz
        U = np.zeros((n_cells, 3))
        in_data = False
        count = 0
        with open(U_file) as f:
            for line in f:
                stripped = line.strip()
                if not in_data:
                    if stripped == '(':
                        in_data = True
                    continue
                if stripped == ')':
                    break
                if stripped.startswith('(') and stripped.endswith(')'):
                    vals = stripped[1:-1].split()
                    if len(vals) == 3 and count < n_cells:
                        U[count] = [float(v) for v in vals]
                        count += 1
        if count != n_cells:
            print(f"  WARNING: U read {count} cells, expected {n_cells}")
        U_3d = U.reshape(mesh.nz, mesh.ny, mesh.nx, 3)
        return U_3d
    except Exception as e:
        print(f"  WARNING: could not read U: {e}")
        return None


def run_phase3(case_dir, mesh, phys, rt_physics, analysis_dir,
               spec_times, df):
    """Run 3D concentration spectrum and velocity statistics at selected times."""
    H = phys['H']
    H0 = phys['H0']
    is_decomposed = os.path.isdir(os.path.join(case_dir, 'processor0'))

    y_centers = np.linspace(mesh.y_min + mesh.dy / 2,
                            mesh.y_max - mesh.dy / 2, mesh.ny)
    x_coords = np.linspace(mesh.x_min + mesh.dx / 2,
                           mesh.x_max - mesh.dx / 2, mesh.nx)
    z_coords = np.linspace(mesh.z_min + mesh.dz / 2,
                           mesh.z_max - mesh.dz / 2, mesh.nz)

    spec_dir = os.path.join(analysis_dir, 'spectra')
    vel_dir = os.path.join(analysis_dir, 'velocity')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(vel_dir, exist_ok=True)

    # Clamp to data range
    if df is not None:
        max_time = df['time'].max()
        spec_times = [t for t in spec_times if t <= max_time + 0.5]

    if not spec_times:
        print("WARNING: No Phase 3 times within data range. Skipping.")
        return

    spec_summary = []
    vel_summary = []

    for target_t in spec_times:
        tau = float(rt_physics.nondim_time(target_t))
        print(f"\n  Phase 3: t={target_t:.2f} s (tau={tau:.4f})")

        try:
            if is_decomposed:
                time_dir = reconstruct_timestep(case_dir, target_t)
            else:
                t_str = str(int(target_t)) if target_t == int(target_t) else f"{target_t}"
                time_dir = os.path.join(case_dir, t_str)

            # Concentration spectrum
            alpha_3d = read_openfoam_alpha_3d(time_dir, mesh)
            if alpha_3d is not None:
                spec = compute_horizontal_spectrum_3d(
                    alpha_3d, y_centers, x_coords, z_coords,
                    H0, H, mesh.dx, mesh.dz)
                if spec is not None:
                    row = {'time': target_t, 'tau': tau,
                           'n_slab_rows': spec['n_slab_rows']}
                    for band in ['10_25', '10_50']:
                        row[f'vof_slope_{band}'] = spec.get(f'slope_{band}', np.nan)
                        row[f'vof_r2_{band}'] = spec.get(f'r2_{band}', np.nan)
                    spec_summary.append(row)
                    # Save full spectrum
                    spec_df = pd.DataFrame({
                        'k_over_k0': spec['k_over_k0'],
                        'P': spec['P'],
                    })
                    spec_df.to_csv(os.path.join(spec_dir,
                                  f'spectrum_3d_t{target_t:.2f}.csv'), index=False)
                    print(f"    spectrum: slope_10_25={row.get('vof_slope_10_25', np.nan):.3f}, "
                          f"slope_10_50={row.get('vof_slope_10_50', np.nan):.3f}")

            # Velocity statistics
            U_3d = read_openfoam_velocity_3d(time_dir, mesh)
            if U_3d is not None:
                vel = compute_velocity_statistics_3d(U_3d, y_centers, H0)
                vel_row = {'time': target_t, 'tau': tau,
                           'u_rms_centreline': vel['u_rms_centreline'],
                           'v_rms_centreline': vel['v_rms_centreline'],
                           'w_rms_centreline': vel['w_rms_centreline'],
                           'anisotropy_v_over_u': vel['anisotropy_v_over_u']}
                vel_summary.append(vel_row)
                # Save profiles
                vel_prof = pd.DataFrame({
                    'y_m': vel['y_centers'],
                    'u_rms': vel['u_rms'],
                    'v_rms': vel['v_rms'],
                    'w_rms': vel['w_rms'],
                })
                vel_prof.to_csv(os.path.join(vel_dir,
                                f'velocity_profile_t{target_t:.2f}.csv'), index=False)
                print(f"    velocity: v'/u'={vel['anisotropy_v_over_u']:.2f}, "
                      f"v_rms={vel['v_rms_centreline']:.4f} m/s")

                del U_3d
            if alpha_3d is not None:
                del alpha_3d

        except Exception as e:
            print(f"  WARNING: Phase 3 failed at t={target_t:.2f}: {e}")
            import traceback; traceback.print_exc()

    # Save summaries
    if spec_summary:
        pd.DataFrame(spec_summary).to_csv(
            os.path.join(spec_dir, 'spectrum_summary_3d.csv'), index=False)
        print(f"\n  Wrote spectrum summary ({len(spec_summary)} timesteps)")
    if vel_summary:
        pd.DataFrame(vel_summary).to_csv(
            os.path.join(vel_dir, 'velocity_summary_3d.csv'), index=False)
        print(f"  Wrote velocity summary ({len(vel_summary)} timesteps)")


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
    parser.add_argument('--skip-phase3', action='store_true',
                        help='Skip Phase 3 (spectra + velocity)')
    parser.add_argument('--skip-energy', action='store_true',
                        help='Skip the Phase 1 energy budget (PE/BPE/KE and the '
                             'dissipation derived from it). Saves reading U at '
                             'every timestep.')
    parser.add_argument('--spec-times', type=str, default='2,4,6,8,10,12,14,16,18,20',
                        help='Comma-separated times for Phase 3 analysis')
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
                        interface_times, analysis_dir,
                        do_energy=not args.skip_energy)
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

    # ── Phase 3 ──
    if not args.skip_phase3:
        print("\n" + "─" * 70)
        print("PHASE 3: Concentration Spectrum + Velocity Statistics")
        print("─" * 70)
        spec_times = [float(t) for t in args.spec_times.split(',')]
        run_phase3(case_dir, mesh, phys, rt_physics,
                   analysis_dir, spec_times, df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results in: {analysis_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
