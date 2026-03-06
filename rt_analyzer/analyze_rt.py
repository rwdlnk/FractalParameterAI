#!/usr/bin/env python3
"""
Generalized Rayleigh-Taylor Simulation Analysis Tool

Performs comprehensive analysis of RT simulation results:
  Phase 1: Interface extraction, mixing thickness, fractal dimension (all timesteps)
  Phase 2: Multifractal spectrum at selected timesteps
  Phase 3: Power spectrum and velocity field analysis

Supports both SOLA-VOF (VTK) and OpenFOAM data formats.
Auto-detects resolution and physics from input files.

Usage:
  python3 analyze_rt.py /path/to/case_dir [options]

Examples:
  python3 analyze_rt.py /data/Threadripper/sola-vof/160x200
  python3 analyze_rt.py /data/Threadripper/OpenFOAM-2D/160x200x1
  python3 analyze_rt.py /data/sola-vof/320x400 --skip-phase3
  python3 analyze_rt.py /data/sola-vof/640x800 --mf-times 2,4,6,8,10,12,14
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# Resolve paths relative to this script's location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)

# Add required paths
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'core'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'integration'))

# Find FractalAnalyzer — look in sibling repos
_fractal_analyzer_path = os.path.join(os.path.dirname(_REPO_ROOT), 'FractalAnalyzer')
if os.path.isdir(_fractal_analyzer_path):
    sys.path.insert(0, _fractal_analyzer_path)

from rt_analyzer import RTAnalyzer
from rt_physics import RTPhysics
from solvof_input import parse_solvof_input, derive_physics, find_input_file, summary


# ═══════════════════════════════════════════════════════════════════════════════
# Case Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_case_format(case_dir):
    """Detect whether a case directory contains SOLA-VOF or OpenFOAM data.

    Returns:
        'vtk' if RT*-*.vtk files found
        'openfoam' if constant/polyMesh/ structure found
        None if neither detected
    """
    # Check for VTK files
    vtk_files = glob.glob(os.path.join(case_dir, 'RT*-*.vtk'))
    if vtk_files:
        return 'vtk'

    # Check for OpenFOAM structure
    polymesh = os.path.join(case_dir, 'constant', 'polyMesh', 'points')
    if os.path.isfile(polymesh):
        return 'openfoam'

    return None


def find_vtk_files(case_dir):
    """Find and sort all VTK data files. Returns (files, nx, ny).

    Excludes Mesh*.vtk files. Extracts resolution from filename pattern.
    """
    pattern = os.path.join(case_dir, 'RT*-*.vtk')
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(re.search(r'-(\d+)\.vtk$', f).group(1)))

    if not files:
        raise FileNotFoundError(f"No RT*-*.vtk files found in {case_dir}")

    # Extract resolution from first filename
    m = re.search(r'RT(\d+)x(\d+)-', os.path.basename(files[0]))
    if not m:
        raise ValueError(f"Cannot parse resolution from: {os.path.basename(files[0])}")

    nx, ny = int(m.group(1)), int(m.group(2))
    return files, nx, ny


def find_openfoam_timesteps(case_dir):
    """Find all numeric time directories in an OpenFOAM case.

    Returns sorted list of (time_float, time_dir_path) tuples.
    """
    timesteps = []
    for entry in os.listdir(case_dir):
        full_path = os.path.join(case_dir, entry)
        if not os.path.isdir(full_path):
            continue
        try:
            t = float(entry)
            # Check that it contains alpha.water
            if os.path.isfile(os.path.join(full_path, 'alpha.water')):
                timesteps.append((t, full_path))
        except ValueError:
            continue
    return sorted(timesteps, key=lambda x: x[0])


def get_physics_vtk(case_dir, input_file=None):
    """Extract physics parameters for a SOLA-VOF case.

    Args:
        case_dir: Path to case directory
        input_file: Override path to int*.in file

    Returns:
        (phys_dict, raw_params) from solvof_input module
    """
    if input_file is None:
        input_file = find_input_file(case_dir)
    if input_file is None:
        raise FileNotFoundError(f"No int*.in file found in {case_dir}")

    params = parse_solvof_input(input_file)
    phys = derive_physics(params)
    return phys, params


def get_physics_openfoam(case_dir):
    """Extract physics parameters from an OpenFOAM case directory.

    Reads constant/transportProperties and constant/g for fluid properties
    and gravity. Falls back to default values if files not found.

    Returns:
        dict with A, g, H, L, H0, NX, NY, RHOF, RHOFC, NU
    """
    from rt_analyzer import _parse_openfoam_points

    # Get domain extents from mesh
    points_file = os.path.join(case_dir, 'constant', 'polyMesh', 'points')
    x_nodes, y_nodes = _parse_openfoam_points(points_file)
    L = float(x_nodes[-1] - x_nodes[0])
    H = float(y_nodes[-1] - y_nodes[0])
    NX = len(x_nodes) - 1
    NY = len(y_nodes) - 1
    dx = L / NX
    dy = H / NY
    H0 = H / 2.0  # Default: mid-domain interface

    # Try to parse transportProperties for densities and viscosities
    rhof, rhofc, nu = 998.0, 998.0, 1.0e-6
    tp_file = os.path.join(case_dir, 'constant', 'transportProperties')
    if os.path.isfile(tp_file):
        with open(tp_file, 'r') as f:
            content = f.read()
        # Look for phase densities and viscosities
        # Format varies but commonly: rho  rho [ ... ] VALUE;
        for phase, var in [('water', 'rho'), ('air', 'rho')]:
            m = re.search(rf'{phase}.*?{var}\s+{var}\s*\[.*?\]\s*([\d.eE+-]+)', content, re.DOTALL)
            if m:
                if phase == 'water':
                    rhof = float(m.group(1))
                else:
                    rhofc = float(m.group(1))
        # nu
        m = re.search(r'nu\s+nu\s*\[.*?\]\s*([\d.eE+-]+)', content)
        if m:
            nu = float(m.group(1))

    # Try to parse g
    g = 9.81  # Default
    g_file = os.path.join(case_dir, 'constant', 'g')
    if os.path.isfile(g_file):
        with open(g_file, 'r') as f:
            content = f.read()
        # OpenFOAM g file: value ( 0 -9.81 0 );
        m = re.search(r'value\s*\(\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*\)', content)
        if m:
            g = abs(float(m.group(2)))  # y-component

    A = (rhof - rhofc) / (rhof + rhofc) if (rhof + rhofc) > 0 else 0.0

    return {
        'A': A, 'g': g, 'H': H, 'L': L, 'H0': H0,
        'NX': NX, 'NY': NY, 'dx': dx, 'dy': dy,
        'RHOF': rhof, 'RHOFC': rhofc, 'NU': nu,
    }


def vtk_time(vtk_file):
    """Extract simulation time from VTK filename (timestep_index / 1000)."""
    m = re.search(r'-(\d+)\.vtk$', os.path.basename(vtk_file))
    return int(m.group(1)) / 1000.0 if m else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Mixing Thickness + Fractal Dimension
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase1(analyzer, physics, vtk_files, analysis_dir, H0):
    """Process all timesteps for mixing thickness and fractal dimension.

    Returns:
        DataFrame with temporal results
    """
    H = physics.H

    # Create output subdirectories
    for subdir in ['interfaces', 'fractal', 'summary']:
        os.makedirs(os.path.join(analysis_dir, subdir), exist_ok=True)

    results = []

    for i, vtk_file in enumerate(vtk_files):
        basename = os.path.basename(vtk_file)
        sim_time = vtk_time(vtk_file)
        tau = float(physics.nondim_time(sim_time))

        print(f"\n[{i+1}/{len(vtk_files)}] {basename}  (t={sim_time:.3f} s, tau={tau:.4f})")

        data = analyzer.read_vtk_file(vtk_file)

        # Skip t=0 -- flat interface, no fractal structure
        if sim_time < 0.01:
            results.append({
                'time': sim_time, 'tau': tau,
                'ht_geo': 0.0, 'hb_geo': 0.0, 'h_total_geo': 0.0,
                'ht_stat': 0.0, 'hb_stat': 0.0, 'h_total_stat': 0.0,
                'fractal_dim': np.nan, 'fd_error': np.nan, 'fd_r_squared': np.nan,
                'n_segments': 0
            })
            print("  t=0: flat interface, skipping")
            continue

        # Mixing thickness -- geometric method
        try:
            mix_geo = analyzer.compute_mixing_thickness(data, H0, method='geometric')
        except Exception as e:
            print(f"  WARNING: geometric mixing failed: {e}")
            mix_geo = {'ht': np.nan, 'hb': np.nan, 'h_total': np.nan}

        # Mixing thickness -- statistical method
        try:
            mix_stat = analyzer.compute_mixing_thickness(data, H0, method='statistical')
        except Exception as e:
            print(f"  WARNING: statistical mixing failed: {e}")
            mix_stat = {'ht': np.nan, 'hb': np.nan, 'h_total': np.nan}

        # Extract interface and save segments
        time_idx = int(re.search(r'-(\d+)\.vtk$', basename).group(1))
        try:
            contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
            segments = analyzer.convert_contours_to_segments(contours)
            n_segments = len(segments)

            interface_file = os.path.join(analysis_dir, 'interfaces',
                                          f'interface_t{time_idx:05d}.dat')
            with open(interface_file, 'w') as fout:
                fout.write(f"# Interface at t={sim_time:.6f} s (tau={tau:.6f})\n")
                fout.write(f"# {n_segments} segments from {len(contours)} contour(s)\n")
                fout.write("# x1,y1 x2,y2\n")
                for seg in segments:
                    fout.write(f"{seg[0][0]:.7f},{seg[0][1]:.7f} "
                              f"{seg[1][0]:.7f},{seg[1][1]:.7f}\n")
        except Exception as e:
            print(f"  WARNING: interface extraction failed: {e}")
            n_segments = 0

        # Fractal dimension
        try:
            fd = analyzer.compute_fractal_dimension(data)
            fd_dim = fd['dimension']
            fd_err = fd['error']
            fd_r2 = fd['r_squared']

            if not np.isnan(fd_dim) and 'box_sizes' in fd:
                bc_file = os.path.join(analysis_dir, 'fractal',
                                       f'boxcount_t{time_idx:05d}.csv')
                bc_df = pd.DataFrame({
                    'box_size': fd['box_sizes'],
                    'box_count': fd['box_counts']
                })
                if 'box_sizes_nondim' in fd:
                    bc_df['box_size_nondim'] = fd['box_sizes_nondim']
                bc_df.to_csv(bc_file, index=False)
        except Exception as e:
            print(f"  WARNING: fractal dimension failed: {e}")
            fd_dim, fd_err, fd_r2 = np.nan, np.nan, np.nan

        print(f"  h_geo={mix_geo['h_total']:.5f}  h_stat={mix_stat['h_total']:.5f}  "
              f"D={fd_dim:.4f}+/-{fd_err:.4f}  R2={fd_r2:.4f}  segs={n_segments}")

        results.append({
            'time': sim_time, 'tau': tau,
            'ht_geo': mix_geo['ht'], 'hb_geo': mix_geo['hb'],
            'h_total_geo': mix_geo['h_total'],
            'ht_stat': mix_stat['ht'], 'hb_stat': mix_stat['hb'],
            'h_total_stat': mix_stat['h_total'],
            'fractal_dim': fd_dim, 'fd_error': fd_err, 'fd_r_squared': fd_r2,
            'n_segments': n_segments
        })

    # Build DataFrame with nondimensional quantities
    df = pd.DataFrame(results)
    for col in ['ht_geo', 'hb_geo', 'h_total_geo', 'ht_stat', 'hb_stat', 'h_total_stat']:
        df[col + '_nondim'] = df[col] / H

    csv_file = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nTemporal results saved to {csv_file}")

    # Generate Phase 1 plots
    _plot_phase1(df, physics, vtk_files, analyzer, H0,
                 os.path.join(analysis_dir, 'summary'))

    return df


def _plot_phase1(df, physics, vtk_files, analyzer, H0, plot_dir):
    """Generate all Phase 1 plots."""
    print("\nGenerating Phase 1 plots...")
    dfv = df[df['time'] > 0.01].copy()
    H = physics.H

    # --- Mixing thickness evolution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(dfv['time'], dfv['h_total_geo'], 'b-', lw=2, label='Geometric (total)')
    ax1.plot(dfv['time'], dfv['ht_geo'], 'r--', lw=1.5, label='Geometric (spike)')
    ax1.plot(dfv['time'], dfv['hb_geo'], 'g--', lw=1.5, label='Geometric (bubble)')
    ax1.plot(dfv['time'], dfv['h_total_stat'], 'k:', lw=2, label='Statistical (total)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mixing thickness (m)')
    ax1.set_title('Mixing Layer Evolution (dimensional)')
    ax1.legend(fontsize=9)
    ax1.grid(True)

    ax2.plot(dfv['tau'], dfv['h_total_geo_nondim'], 'b-', lw=2, label='Geometric (total)')
    ax2.plot(dfv['tau'], dfv['ht_geo_nondim'], 'r--', lw=1.5, label='Spike')
    ax2.plot(dfv['tau'], dfv['hb_geo_nondim'], 'g--', lw=1.5, label='Bubble')
    ax2.set_xlabel(r'$\tau = t\sqrt{Ag/H}$')
    ax2.set_ylabel(r'$h/H$')
    ax2.set_title('Mixing Layer Evolution (nondimensional)')
    ax2.legend(fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mixing_evolution.png'), dpi=300)
    plt.close()

    # --- Self-similar scaling: h/H vs tau^2 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dfv['tau']**2, dfv['h_total_geo_nondim'], 'bo-', ms=4, lw=1.5,
            label=r'$h/H$ vs $\tau^2$ (geometric)')
    ax.plot(dfv['tau']**2, dfv['h_total_stat_nondim'], 'rs-', ms=4, lw=1.5,
            label=r'$h/H$ vs $\tau^2$ (statistical)')

    late = dfv[dfv['tau'] > dfv['tau'].max() * 0.3]
    if len(late) > 3:
        coeffs = np.polyfit(late['tau']**2, late['h_total_geo_nondim'], 1)
        alpha_RT = coeffs[0]
        tau2_fit = np.linspace(late['tau'].min()**2, late['tau'].max()**2, 50)
        ax.plot(tau2_fit, np.polyval(coeffs, tau2_fit), 'b--', lw=1,
                label=rf'$\alpha_{{RT}} \approx {alpha_RT:.4f}$ (geo)')
        coeffs_s = np.polyfit(late['tau']**2, late['h_total_stat_nondim'], 1)
        alpha_RT_s = coeffs_s[0]
        ax.plot(tau2_fit, np.polyval(coeffs_s, tau2_fit), 'r--', lw=1,
                label=rf'$\alpha_{{RT}} \approx {alpha_RT_s:.4f}$ (stat)')

    ax.set_xlabel(r'$\tau^2$')
    ax.set_ylabel(r'$h/H$')
    ax.set_title(r'Self-similar scaling: $h/H = \alpha \tau^2$')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'mixing_self_similar.png'), dpi=300)
    plt.close()

    # --- Fractal dimension evolution ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    valid_fd = dfv.dropna(subset=['fractal_dim'])

    ax1.errorbar(valid_fd['tau'], valid_fd['fractal_dim'], yerr=valid_fd['fd_error'],
                 fmt='ko-', capsize=3, lw=1.5, ms=4, label='Fractal dimension')
    ax1.fill_between(valid_fd['tau'],
                     valid_fd['fractal_dim'] - valid_fd['fd_error'],
                     valid_fd['fractal_dim'] + valid_fd['fd_error'],
                     alpha=0.2, color='gray')
    ax1.set_xlabel(r'$\tau = t\sqrt{Ag/H}$')
    ax1.set_ylabel('Fractal dimension D')
    ax1.set_title('Fractal Dimension Evolution')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(valid_fd['tau'], valid_fd['fd_r_squared'], 'g--', alpha=0.6, label='R2')
    ax2.set_ylabel('R2', color='g')
    ax2.set_ylim(0.9, 1.0)
    ax2.tick_params(axis='y', labelcolor='g')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'fractal_dimension_evolution.png'), dpi=300)
    plt.close()

    # --- Combined: mixing + fractal dimension ---
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(dfv['tau'], dfv['h_total_geo_nondim'], 'b-', lw=2, label='Mixing thickness h/H')
    ax1.set_xlabel(r'$\tau$', fontsize=14)
    ax1.set_ylabel(r'$h/H$', color='b', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.errorbar(valid_fd['tau'], valid_fd['fractal_dim'], yerr=valid_fd['fd_error'],
                 fmt='ro-', capsize=3, ms=4, label='Fractal dimension D')
    ax2.set_ylabel('Fractal dimension D', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Mixing Layer Growth and Fractal Dimension', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'combined_evolution.png'), dpi=300)
    plt.close()

    # --- Interface snapshots ---
    max_time = df['time'].max()
    snapshot_times = [t for t in [2.0, 5.0, 8.0, 10.0, 14.0, 18.0] if t <= max_time]
    if not snapshot_times:
        snapshot_times = np.linspace(df['time'].min(), max_time, 6).tolist()

    n_snap = len(snapshot_times)
    ncols = min(3, n_snap)
    nrows = (n_snap + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if n_snap == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, target_t in enumerate(snapshot_times):
        best_file = min(vtk_files, key=lambda f: abs(vtk_time(f) - target_t))
        data = analyzer.read_vtk_file(best_file)
        actual_t = data['time']
        tau_val = float(physics.nondim_time(actual_t))

        ax = axes[idx]
        cf = ax.contourf(data['x'], data['y'], data['f'], levels=20, cmap='RdBu_r')
        contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
        for c in contours:
            ax.plot(c[:, 0], c[:, 1], 'k-', lw=0.8)
        ax.axhline(y=H0, color='gray', ls='--', alpha=0.5)
        ax.set_title(rf't={actual_t:.2f} s ($\tau$={tau_val:.3f})', fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    # Hide unused axes
    for idx in range(n_snap, len(axes)):
        axes[idx].set_visible(False)

    nx_res = int(re.search(r'RT(\d+)x', os.path.basename(vtk_files[0])).group(1))
    ny_res = int(re.search(r'x(\d+)-', os.path.basename(vtk_files[0])).group(1))
    plt.suptitle(f'RT Interface Evolution -- {nx_res}x{ny_res} SOLA-VOF', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'interface_snapshots.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Phase 1 plots complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Multifractal Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase2(analyzer, physics, vtk_files, analysis_dir, mf_times, df):
    """Run multifractal analysis at selected timesteps."""
    mf_output_dir = os.path.join(analysis_dir, 'multifractal')
    os.makedirs(mf_output_dir, exist_ok=True)

    # Clamp mf_times to actual data range
    max_time = df['time'].max()
    mf_times = [t for t in mf_times if t <= max_time + 0.5]
    if not mf_times:
        print("WARNING: No multifractal times within data range. Skipping Phase 2.")
        return

    # Build TIME_FILES dict
    mf_files = {}
    for target_t in mf_times:
        best_file = min(vtk_files, key=lambda f: abs(vtk_time(f) - target_t))
        actual_t = vtk_time(best_file)
        mf_files[actual_t] = best_file
        print(f"  MF target t={target_t:.1f} -> actual t={actual_t:.3f}: "
              f"{os.path.basename(best_file)}")

    q_values = np.arange(-5, 5.1, 0.5)

    mf_results = analyzer.analyze_multifractal_evolution(
        mf_files,
        output_dir=mf_output_dir,
        q_values=q_values
    )

    if mf_results:
        _plot_phase2(mf_results, physics, df, mf_output_dir)


def _plot_phase2(mf_results, physics, df, mf_output_dir):
    """Generate all Phase 2 plots."""
    print("\nGenerating Phase 2 plots...")
    cmap = get_cmap('viridis')
    times_mf = sorted([r['time'] for r in mf_results])
    t_min, t_max = min(times_mf), max(times_mf)

    # --- Overlaid f(alpha) spectra ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for res in sorted(mf_results, key=lambda x: x['time']):
        t = res['time']
        tau_val = float(physics.nondim_time(t))
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
    ax.set_title(r'Multifractal Spectrum Evolution $f(\alpha)$', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'f_alpha_evolution_overlay.png'), dpi=300)
    plt.close()

    # --- Overlaid D(q) curves ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for res in sorted(mf_results, key=lambda x: x['time']):
        t = res['time']
        tau_val = float(physics.nondim_time(t))
        color = cmap((t - t_min) / (t_max - t_min)) if t_max > t_min else cmap(0.5)
        valid = ~np.isnan(res['Dq'])
        ax.plot(res['q_values'][valid], res['Dq'][valid], '-o', color=color,
                ms=3, lw=1.5, label=rf'$\tau$={tau_val:.3f}')

    ax.set_xlabel('q', fontsize=14)
    ax.set_ylabel('D(q)', fontsize=14)
    ax.set_title('Generalized Dimensions D(q) Evolution', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'Dq_evolution_overlay.png'), dpi=300)
    plt.close()

    # --- D0, D1, D2 vs tau ---
    fig, ax = plt.subplots(figsize=(10, 6))
    tau_vals = [float(physics.nondim_time(r['time'])) for r in mf_results]
    D0 = [r['D0'] for r in mf_results]
    D1 = [r['D1'] for r in mf_results]
    D2 = [r['D2'] for r in mf_results]

    ax.plot(tau_vals, D0, 'bo-', ms=6, lw=2, label=r'$D_0$ (capacity)')
    ax.plot(tau_vals, D1, 'rs-', ms=6, lw=2, label=r'$D_1$ (information)')
    ax.plot(tau_vals, D2, 'gd-', ms=6, lw=2, label=r'$D_2$ (correlation)')
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Dimension', fontsize=14)
    ax.set_title('Generalized Dimensions vs. Nondimensional Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'D012_vs_tau.png'), dpi=300)
    plt.close()

    # --- Multifractality measures vs tau ---
    fig, ax = plt.subplots(figsize=(10, 6))
    aw = [r['alpha_width'] for r in mf_results]
    dm = [r['degree_multifractality'] for r in mf_results]
    ax.plot(tau_vals, aw, 'ms-', ms=6, lw=2, label=r'$\Delta\alpha$ (spectrum width)')
    ax.plot(tau_vals, dm, 'cd-', ms=6, lw=2, label=r'$D_{-5} - D_5$ (multifractality degree)')
    ax.set_xlabel(r'$\tau$', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title('Multifractality Measures vs. Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'multifractality_measures.png'), dpi=300)
    plt.close()

    # --- Phase portrait: D0 vs h/H ---
    fig, ax = plt.subplots(figsize=(10, 7))
    h_at_mf = []
    for res in mf_results:
        row = df.iloc[(df['time'] - res['time']).abs().argsort()[:1]]
        h_at_mf.append(float(row['h_total_geo_nondim'].values[0]))

    ax.plot(h_at_mf, D0, 'bo-', ms=8, lw=2, label=r'$D_0$')
    ax.plot(h_at_mf, D1, 'rs-', ms=8, lw=2, label=r'$D_1$')
    for i, tv in enumerate(tau_vals):
        ax.annotate(rf'$\tau$={tv:.2f}', (h_at_mf[i], D0[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel(r'$h/H$', fontsize=14)
    ax.set_ylabel('Fractal Dimension', fontsize=14)
    ax.set_title('Phase Portrait: Dimension vs. Mixing Thickness', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(mf_output_dir, 'phase_portrait.png'), dpi=300)
    plt.close()

    # Save multifractal summary CSV
    mf_summary = pd.DataFrame({
        'time': [r['time'] for r in mf_results],
        'tau': tau_vals,
        'h_H': h_at_mf,
        'D0': D0, 'D1': D1, 'D2': D2,
        'alpha_width': aw,
        'degree_multifractality': dm
    })
    mf_summary.to_csv(os.path.join(mf_output_dir, 'multifractal_summary.csv'), index=False)

    print("Phase 2 plots complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Power Spectrum + Velocity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_phase3(analyzer, physics, vtk_files, analysis_dir, spec_times, df):
    """Run power spectrum and velocity field analysis at selected timesteps."""
    from rt_power_spectrum import PowerSpectrumAnalyzer
    from rt_velocity_statistics import VelocityAnalyzer

    spectra_dir = os.path.join(analysis_dir, 'spectra')
    velocity_dir = os.path.join(analysis_dir, 'velocity')
    os.makedirs(spectra_dir, exist_ok=True)
    os.makedirs(velocity_dir, exist_ok=True)

    psa = PowerSpectrumAnalyzer()
    va = VelocityAnalyzer()

    # Clamp to data range
    max_time = df['time'].max()
    spec_times = [t for t in spec_times if t <= max_time + 0.5]
    if not spec_times:
        print("WARNING: No spectrum times within data range. Skipping Phase 3.")
        return

    spec_results_list = []
    vel_results_list = []

    for target_t in spec_times:
        best_file = min(vtk_files, key=lambda f: abs(vtk_time(f) - target_t))
        data = analyzer.read_vtk_file(best_file)
        actual_t = data['time']
        tau_val = float(physics.nondim_time(actual_t))
        print(f"\n  t={actual_t:.3f} s (tau={tau_val:.4f}): {os.path.basename(best_file)}")

        # --- Power spectrum ---
        spec_result = {'time': actual_t, 'tau': tau_val}

        vof_spec = psa.analyze_2d_field_spectrum(data['f'], data['x'], data['y'],
                                                  field_name='F', detrend=True)
        spec_result['vof_spectrum'] = vof_spec
        print(f"    VOF spectrum: slope={vof_spec['power_law_slope']:.3f}, "
              f"R2={vof_spec['power_law_r_squared']:.3f}, "
              f"lambda_dom={vof_spec['dominant_wavelength']:.4f}")

        if 'U' in data and 'V' in data:
            u_spec = psa.analyze_2d_field_spectrum(data['U'], data['x'], data['y'],
                                                    field_name='u', detrend=True)
            v_spec = psa.analyze_2d_field_spectrum(data['V'], data['x'], data['y'],
                                                    field_name='v', detrend=True)
            tke_spec = psa.analyze_tke_spectrum(data['U'], data['V'],
                                                data['x'], data['y'], detrend=True)
            spec_result['u_velocity_spectrum'] = u_spec
            spec_result['v_velocity_spectrum'] = v_spec
            spec_result['tke_spectrum'] = tke_spec
            print(f"    TKE spectrum: slope={tke_spec['power_law_slope']:.3f}, "
                  f"R2={tke_spec['power_law_r_squared']:.3f}")

        spec_results_list.append(spec_result)

        # --- Velocity statistics ---
        if 'U' in data and 'V' in data:
            vstats = va.analyze_velocity_field(data['U'], data['V'])
            vel_results_list.append({
                'time': actual_t, 'tau': tau_val,
                'u_mean': vstats.u_mean, 'v_mean': vstats.v_mean,
                'u_rms': vstats.u_rms, 'v_rms': vstats.v_rms,
                'TKE': vstats.turbulent_kinetic_energy,
                'reynolds_stress': vstats.reynolds_stress,
                'turbulence_intensity': vstats.turbulence_intensity,
                'velocity_mag_max': vstats.velocity_magnitude_max,
                'velocity_mag_mean': vstats.velocity_magnitude_mean,
            })
            print(f"    Velocity: u_rms={vstats.u_rms:.4e}, v_rms={vstats.v_rms:.4e}, "
                  f"TKE={vstats.turbulent_kinetic_energy:.4e}, "
                  f"|V|_max={vstats.velocity_magnitude_max:.4e}")

    # Generate Phase 3 plots
    _plot_phase3(spec_results_list, vel_results_list, analyzer, physics, psa,
                 vtk_files, spectra_dir, velocity_dir)


def _plot_phase3(spec_results_list, vel_results_list, analyzer, physics, psa,
                 vtk_files, spectra_dir, velocity_dir):
    """Generate all Phase 3 plots."""
    print("\nGenerating Phase 3 plots...")

    nx_res = int(re.search(r'RT(\d+)x', os.path.basename(vtk_files[0])).group(1))
    ny_res = int(re.search(r'x(\d+)-', os.path.basename(vtk_files[0])).group(1))
    res_label = f'{nx_res}x{ny_res}'

    # --- Velocity statistics evolution ---
    if vel_results_list:
        vel_df = pd.DataFrame(vel_results_list)
        vel_df.to_csv(os.path.join(velocity_dir, 'velocity_statistics.csv'), index=False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(vel_df['tau'], vel_df['u_rms'], 'bo-', lw=2, ms=5, label=r"$u'_{rms}$")
        ax.plot(vel_df['tau'], vel_df['v_rms'], 'rs-', lw=2, ms=5, label=r"$v'_{rms}$")
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('RMS velocity (m/s)')
        ax.set_title('RMS Velocity Fluctuations')
        ax.legend()
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(vel_df['tau'], vel_df['TKE'], 'kd-', lw=2, ms=5)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('TKE (m$^2$/s$^2$)')
        ax.set_title('Turbulent Kinetic Energy')
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(vel_df['tau'], vel_df['reynolds_stress'], 'g^-', lw=2, ms=5)
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r"$\langle u'v' \rangle$ (m$^2$/s$^2$)")
        ax.set_title('Reynolds Stress')
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(vel_df['tau'], vel_df['velocity_mag_max'], 'mp-', lw=2, ms=5, label='Max |V|')
        ax.plot(vel_df['tau'], vel_df['velocity_mag_mean'], 'c+-', lw=2, ms=5, label='Mean |V|')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('Velocity magnitude (m/s)')
        ax.set_title('Velocity Magnitude')
        ax.legend()
        ax.grid(True)

        plt.suptitle(f'Velocity Statistics Evolution -- {res_label} SOLA-VOF', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(velocity_dir, 'velocity_statistics_evolution.png'), dpi=300)
        plt.close()

        # --- Velocity field snapshots ---
        max_time = max(r['time'] for r in vel_results_list)
        snap_times = [t for t in [2.0, 8.0, 14.0, 18.0] if t <= max_time + 0.5]
        if not snap_times:
            snap_times = [vel_results_list[0]['time'], vel_results_list[-1]['time']]

        n_snap = len(snap_times)
        fig, axes = plt.subplots(2, n_snap, figsize=(5*n_snap, 10))
        if n_snap == 1:
            axes = axes.reshape(2, 1)

        for idx, target_t in enumerate(snap_times):
            best_file = min(vtk_files, key=lambda f: abs(vtk_time(f) - target_t))
            data = analyzer.read_vtk_file(best_file)
            actual_t = data['time']
            tau_val = float(physics.nondim_time(actual_t))

            if 'U' not in data or 'V' not in data:
                continue

            vel_mag = np.sqrt(data['U']**2 + data['V']**2)

            ax = axes[0, idx]
            cf = ax.contourf(data['x'], data['y'], vel_mag, levels=20, cmap='hot')
            plt.colorbar(cf, ax=ax, label='|V| (m/s)')
            ax.set_title(rf't={actual_t:.1f} s ($\tau$={tau_val:.3f})')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')

            ax = axes[1, idx]
            dx = data['x'][1, 0] - data['x'][0, 0]
            dy = data['y'][0, 1] - data['y'][0, 0]
            dvdx = np.gradient(data['V'], dx, axis=0)
            dudy = np.gradient(data['U'], dy, axis=1)
            vorticity = dvdx - dudy
            vmax = np.percentile(np.abs(vorticity), 98)
            cf = ax.contourf(data['x'], data['y'], vorticity, levels=20,
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            plt.colorbar(cf, ax=ax, label=r'$\omega_z$ (1/s)')
            ax.set_title(rf'Vorticity t={actual_t:.1f} s')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal')

        plt.suptitle(f'Velocity Field & Vorticity -- {res_label} SOLA-VOF', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(velocity_dir, 'velocity_field_snapshots.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # --- Power spectrum plots ---
    if spec_results_list:
        # Save spectrum summary CSV
        spec_summary = []
        for sr in spec_results_list:
            row = {'time': sr['time'], 'tau': sr['tau']}
            for key in ['vof_spectrum', 'u_velocity_spectrum', 'v_velocity_spectrum', 'tke_spectrum']:
                if key in sr:
                    prefix = key.replace('_spectrum', '')
                    row[f'{prefix}_slope'] = sr[key]['power_law_slope']
                    row[f'{prefix}_R2'] = sr[key]['power_law_r_squared']
                    row[f'{prefix}_dom_wavelength'] = sr[key]['dominant_wavelength']
                    row[f'{prefix}_total_energy'] = sr[key]['total_energy']
            spec_summary.append(row)
        pd.DataFrame(spec_summary).to_csv(
            os.path.join(spectra_dir, 'spectrum_summary.csv'), index=False)

        # 4-panel spectral evolution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        taus = [sr['tau'] for sr in spec_results_list]

        ax = axes[0, 0]
        ax.plot(taus, [sr['vof_spectrum']['power_law_slope'] for sr in spec_results_list],
                'bo-', lw=2, ms=5, label='VOF (F)')
        if 'tke_spectrum' in spec_results_list[0]:
            ax.plot(taus, [sr['tke_spectrum']['power_law_slope'] for sr in spec_results_list],
                    'rs-', lw=2, ms=5, label='TKE')
        ax.axhline(-5/3, color='k', ls='--', alpha=0.5, label=r'$k^{-5/3}$')
        ax.axhline(-3, color='gray', ls=':', alpha=0.5, label=r'$k^{-3}$')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('Spectral slope')
        ax.set_title('Power Law Slope Evolution')
        ax.legend(fontsize=9)
        ax.grid(True)

        ax = axes[0, 1]
        ax.plot(taus, [sr['vof_spectrum']['dominant_wavelength'] for sr in spec_results_list],
                'go-', lw=2, ms=5, label='VOF (F)')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('Dominant wavelength (m)')
        ax.set_title('Dominant Wavelength Evolution')
        ax.legend()
        ax.grid(True)

        ax = axes[1, 0]
        ax.semilogy(taus, [sr['vof_spectrum']['total_energy'] for sr in spec_results_list],
                    'bo-', lw=2, ms=5, label='VOF (F)')
        if 'tke_spectrum' in spec_results_list[0]:
            ax.semilogy(taus, [sr['tke_spectrum']['total_energy'] for sr in spec_results_list],
                        'rs-', lw=2, ms=5, label='TKE')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel('Total spectral energy')
        ax.set_title('Spectral Energy Evolution')
        ax.legend()
        ax.grid(True)

        ax = axes[1, 1]
        ax.plot(taus, [sr['vof_spectrum']['power_law_r_squared'] for sr in spec_results_list],
                'bo-', lw=2, ms=5, label='VOF (F)')
        if 'tke_spectrum' in spec_results_list[0]:
            ax.plot(taus, [sr['tke_spectrum']['power_law_r_squared'] for sr in spec_results_list],
                    'rs-', lw=2, ms=5, label='TKE')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$R^2$')
        ax.set_title('Power Law Fit Quality')
        ax.legend()
        ax.grid(True)

        plt.suptitle(f'Power Spectrum Evolution -- {res_label} SOLA-VOF', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(spectra_dir, 'spectrum_evolution.png'), dpi=300)
        plt.close()

        # Individual spectrum plots at early, mid, late times
        indices = [0, len(spec_results_list)//2, -1]
        for idx in indices:
            sr = spec_results_list[idx]
            t = sr['time']
            tau_val = sr['tau']

            best_file = min(vtk_files, key=lambda f: abs(vtk_time(f) - t))
            data = analyzer.read_vtk_file(best_file)

            field = data['f'] - np.mean(data['f'])
            nx_f, ny_f = data['f'].shape
            dx = np.abs(data['x'][1, 0] - data['x'][0, 0])
            dy = np.abs(data['y'][0, 1] - data['y'][0, 0])
            fft_2d = np.fft.fft2(field)
            power_2d = np.abs(fft_2d)**2 / field.size
            kx = 2 * np.pi * np.fft.fftfreq(nx_f, d=dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny_f, d=dy)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            K = np.sqrt(KX**2 + KY**2)
            wavenumbers, power = psa._radial_average(K, power_2d)

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.loglog(wavenumbers, power, 'b-', lw=2, alpha=0.8, label='E(k) -- VOF')

            slope = sr['vof_spectrum']['power_law_slope']
            n_k = len(wavenumbers)
            k_mid = wavenumbers[n_k//4:3*n_k//4]
            p_mid = power[n_k//4:3*n_k//4]
            valid = p_mid > 0
            if np.sum(valid) > 2:
                log_k = np.log10(k_mid[valid])
                log_p = np.log10(p_mid[valid])
                intercept = np.mean(log_p - slope * log_k)
                p_fit = 10**(slope * np.log10(k_mid) + intercept)
                ax.loglog(k_mid, p_fit, 'r--', lw=2,
                          label=rf'$k^{{{slope:.2f}}}$ '
                                rf'(R$^2$={sr["vof_spectrum"]["power_law_r_squared"]:.3f})')

            k_ref = wavenumbers[n_k//2]
            p_ref = power[n_k//2]
            k_53 = np.logspace(np.log10(k_ref*0.3), np.log10(k_ref*3), 20)
            ax.loglog(k_53, p_ref * (k_53/k_ref)**(-5/3), 'k:', lw=1.5, alpha=0.5,
                      label=r'$k^{-5/3}$')

            ax.set_xlabel('Wavenumber k (rad/m)', fontsize=13)
            ax.set_ylabel('Power Spectral Density', fontsize=13)
            ax.set_title(rf'Power Spectrum at $\tau$={tau_val:.3f} (t={t:.1f} s)', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3, which='both')
            plt.tight_layout()
            plt.savefig(os.path.join(spectra_dir, f'spectrum_t{t:.0f}.png'), dpi=300)
            plt.close()

    print("Phase 3 plots complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Generalized RT simulation analysis tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /data/Threadripper/sola-vof/160x200
  %(prog)s /data/Threadripper/sola-vof/320x400 --skip-phase3
  %(prog)s /data/Threadripper/OpenFOAM-2D/160x200x1
  %(prog)s /data/sola-vof/640x800 --mf-times 2,4,6,8,10,12,14
        """)
    parser.add_argument('case_dir', help='Path to case directory')
    parser.add_argument('--input-file',
                        help='Override input file (default: auto-detect int*.in)')
    parser.add_argument('--mf-times', default='2,4,6,8,10,12,14,16,18',
                        help='Comma-separated multifractal/spectrum analysis times (default: 2,4,...,18)')
    parser.add_argument('--skip-phase1', action='store_true',
                        help='Skip Phase 1 (mixing + fractal)')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 (multifractal)')
    parser.add_argument('--skip-phase3', action='store_true',
                        help='Skip Phase 3 (spectra + velocity)')
    parser.add_argument('--output-dir',
                        help='Override output directory (default: case_dir/analysis)')

    args = parser.parse_args()

    case_dir = os.path.abspath(args.case_dir)
    if not os.path.isdir(case_dir):
        print(f"ERROR: Case directory not found: {case_dir}")
        sys.exit(1)

    # Detect format
    fmt = detect_case_format(case_dir)
    if fmt is None:
        print(f"ERROR: Cannot detect data format in {case_dir}")
        print("  Expected: RT*-*.vtk files (SOLA-VOF) or constant/polyMesh/ (OpenFOAM)")
        sys.exit(1)

    print(f"Case directory: {case_dir}")
    print(f"Data format:    {fmt}")

    # Get physics parameters
    if fmt == 'vtk':
        phys, raw_params = get_physics_vtk(case_dir, args.input_file)
        from solvof_input import summary as phys_summary
        print(f"\n{phys_summary(raw_params, phys)}\n")

        vtk_files, nx_det, ny_det = find_vtk_files(case_dir)

        # Validate resolution
        if nx_det != phys['NX'] or ny_det != phys['NY']:
            print(f"WARNING: Filename resolution ({nx_det}x{ny_det}) differs from "
                  f"input file ({phys['NX']}x{phys['NY']})")

        print(f"Found {len(vtk_files)} VTK files")
        print(f"Time range: {vtk_time(vtk_files[0]):.3f} to {vtk_time(vtk_files[-1]):.3f} s\n")

    elif fmt == 'openfoam':
        phys = get_physics_openfoam(case_dir)
        print(f"\nGrid: {phys['NX']}x{phys['NY']} cells")
        print(f"Domain: L={phys['L']:.4f} m x H={phys['H']:.4f} m")
        print(f"Atwood: A={phys['A']:.6f}, g={phys['g']:.4f} m/s2\n")

        timesteps = find_openfoam_timesteps(case_dir)
        if not timesteps:
            print("ERROR: No timesteps with alpha.water found")
            sys.exit(1)
        print(f"Found {len(timesteps)} timesteps")
        print(f"Time range: {timesteps[0][0]:.3f} to {timesteps[-1][0]:.3f} s\n")

        # For OpenFOAM, we read each timestep and create VTK-like dicts
        # The analysis functions expect vtk_files list -- we'll create wrapper
        vtk_files = None  # Handled differently below

    # Initialize physics and analyzer
    physics = RTPhysics(A=phys['A'], g=phys['g'], H=phys['H'], L=phys['L'])
    print(physics.summary())
    print()

    analysis_dir = args.output_dir or os.path.join(case_dir, 'analysis')
    analyzer = RTAnalyzer(output_dir=analysis_dir, rt_physics=physics)

    # Parse multifractal/spectrum times
    mf_times = [float(t) for t in args.mf_times.split(',')]

    H0 = phys['H0']

    # ─── Run Analysis Phases ──────────────────────────────────────────────
    if fmt == 'vtk':
        # Phase 1
        df = None
        if not args.skip_phase1:
            print("=" * 70)
            print("PHASE 1: Mixing thickness and fractal dimension for all timesteps")
            print("=" * 70)
            df = run_phase1(analyzer, physics, vtk_files, analysis_dir, H0)
        else:
            # Try to load existing results
            csv_path = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                print(f"Loaded existing Phase 1 results from {csv_path}")
            else:
                print("WARNING: Phase 1 skipped and no existing results found.")
                print("Phase 2 and 3 require Phase 1 results. Running Phase 1.")
                print("=" * 70)
                print("PHASE 1: Mixing thickness and fractal dimension for all timesteps")
                print("=" * 70)
                df = run_phase1(analyzer, physics, vtk_files, analysis_dir, H0)

        # Phase 2
        if not args.skip_phase2:
            print("\n" + "=" * 70)
            print("PHASE 2: Multifractal analysis at selected timesteps")
            print("=" * 70)
            run_phase2(analyzer, physics, vtk_files, analysis_dir, mf_times, df)

        # Phase 3
        if not args.skip_phase3:
            print("\n" + "=" * 70)
            print("PHASE 3: Power spectrum and velocity field analysis")
            print("=" * 70)
            run_phase3(analyzer, physics, vtk_files, analysis_dir, mf_times, df)

    elif fmt == 'openfoam':
        # OpenFOAM support -- read each timestep via read_openfoam_field
        # For now, print a message about future support
        print("OpenFOAM analysis: reading timesteps via read_openfoam_field()...")

        # Convert OpenFOAM timesteps to a list that Phase functions can use
        # We create temporary VTK-like files list with time info
        # The analyzer already has read_openfoam_field() method
        of_times = [t for t, _ in timesteps]
        of_dirs = [d for _, d in timesteps]

        # Phase 1 for OpenFOAM
        if not args.skip_phase1:
            print("=" * 70)
            print("PHASE 1: Mixing thickness and fractal dimension for all timesteps")
            print("=" * 70)
            df = _run_phase1_openfoam(analyzer, physics, case_dir,
                                       of_times, analysis_dir, H0)
        else:
            csv_path = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
            else:
                print("WARNING: Phase 1 skipped and no existing results. Running Phase 1.")
                df = _run_phase1_openfoam(analyzer, physics, case_dir,
                                           of_times, analysis_dir, H0)

        if not args.skip_phase2:
            print("\n" + "=" * 70)
            print("PHASE 2: Multifractal analysis at selected timesteps")
            print("=" * 70)
            _run_phase2_openfoam(analyzer, physics, case_dir,
                                 of_times, analysis_dir, mf_times, df)

        if not args.skip_phase3:
            print("\n" + "=" * 70)
            print("PHASE 3: Power spectrum and velocity field analysis")
            print("=" * 70)
            _run_phase3_openfoam(analyzer, physics, case_dir,
                                  of_times, analysis_dir, mf_times, df)

    # ─── Final Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nPhysics: A={phys['A']:.6f}, g={phys['g']}, H={phys['H']}, L={phys['L']}")
    print(f"tau_factor = {physics.tau_factor:.6f} s^-1")
    if df is not None:
        print(f"Time range: 0 to {df['time'].max():.3f} s  "
              f"(tau = 0 to {df['tau'].max():.4f})")
    print(f"\nOutput: {analysis_dir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# OpenFOAM Phase Functions
# ═══════════════════════════════════════════════════════════════════════════════

def _run_phase1_openfoam(analyzer, physics, case_dir, of_times, analysis_dir, H0):
    """Phase 1 for OpenFOAM cases: process all timesteps."""
    H = physics.H

    for subdir in ['interfaces', 'fractal', 'summary']:
        os.makedirs(os.path.join(analysis_dir, subdir), exist_ok=True)

    results = []

    for i, sim_time in enumerate(of_times):
        tau = float(physics.nondim_time(sim_time))
        print(f"\n[{i+1}/{len(of_times)}] t={sim_time:.3f} s (tau={tau:.4f})")

        data = analyzer.read_openfoam_field(case_dir, sim_time)

        if sim_time < 0.01:
            results.append({
                'time': sim_time, 'tau': tau,
                'ht_geo': 0.0, 'hb_geo': 0.0, 'h_total_geo': 0.0,
                'ht_stat': 0.0, 'hb_stat': 0.0, 'h_total_stat': 0.0,
                'fractal_dim': np.nan, 'fd_error': np.nan, 'fd_r_squared': np.nan,
                'n_segments': 0
            })
            continue

        try:
            mix_geo = analyzer.compute_mixing_thickness(data, H0, method='geometric')
        except Exception as e:
            mix_geo = {'ht': np.nan, 'hb': np.nan, 'h_total': np.nan}

        try:
            mix_stat = analyzer.compute_mixing_thickness(data, H0, method='statistical')
        except Exception as e:
            mix_stat = {'ht': np.nan, 'hb': np.nan, 'h_total': np.nan}

        try:
            contours = analyzer.extract_interface(data['f'], data['x'], data['y'])
            segments = analyzer.convert_contours_to_segments(contours)
            n_segments = len(segments)

            time_idx = int(sim_time * 1000)
            interface_file = os.path.join(analysis_dir, 'interfaces',
                                          f'interface_t{time_idx:05d}.dat')
            with open(interface_file, 'w') as fout:
                fout.write(f"# Interface at t={sim_time:.6f} s (tau={tau:.6f})\n")
                fout.write(f"# {n_segments} segments\n")
                fout.write("# x1,y1 x2,y2\n")
                for seg in segments:
                    fout.write(f"{seg[0][0]:.7f},{seg[0][1]:.7f} "
                              f"{seg[1][0]:.7f},{seg[1][1]:.7f}\n")
        except Exception:
            n_segments = 0

        try:
            fd = analyzer.compute_fractal_dimension(data)
            fd_dim, fd_err, fd_r2 = fd['dimension'], fd['error'], fd['r_squared']
        except Exception:
            fd_dim, fd_err, fd_r2 = np.nan, np.nan, np.nan

        print(f"  h_geo={mix_geo['h_total']:.5f}  h_stat={mix_stat['h_total']:.5f}  "
              f"D={fd_dim:.4f}+/-{fd_err:.4f}  segs={n_segments}")

        results.append({
            'time': sim_time, 'tau': tau,
            'ht_geo': mix_geo['ht'], 'hb_geo': mix_geo['hb'],
            'h_total_geo': mix_geo['h_total'],
            'ht_stat': mix_stat['ht'], 'hb_stat': mix_stat['hb'],
            'h_total_stat': mix_stat['h_total'],
            'fractal_dim': fd_dim, 'fd_error': fd_err, 'fd_r_squared': fd_r2,
            'n_segments': n_segments
        })

    df = pd.DataFrame(results)
    for col in ['ht_geo', 'hb_geo', 'h_total_geo', 'ht_stat', 'hb_stat', 'h_total_stat']:
        df[col + '_nondim'] = df[col] / H

    csv_file = os.path.join(analysis_dir, 'summary', 'temporal_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"\nTemporal results saved to {csv_file}")

    # Plots would go here -- for now reuse VTK plotting by generating
    # snapshot data on the fly (to be expanded when OpenFOAM runs are available)
    print("Phase 1 complete (OpenFOAM). Plotting deferred until data available.")

    return df


def _run_phase2_openfoam(analyzer, physics, case_dir, of_times, analysis_dir, mf_times, df):
    """Phase 2 for OpenFOAM cases."""
    mf_output_dir = os.path.join(analysis_dir, 'multifractal')
    os.makedirs(mf_output_dir, exist_ok=True)

    max_time = df['time'].max()
    mf_times = [t for t in mf_times if t <= max_time + 0.5]
    if not mf_times:
        print("WARNING: No multifractal times within data range.")
        return

    # For OpenFOAM, we need to build TIME_FILES differently
    # The multifractal evolution expects {time: vtk_file} dict
    # We'll read data directly and call the underlying methods
    print("Phase 2 (OpenFOAM): Multifractal analysis...")
    # Find closest available times
    mf_actual = {}
    for target_t in mf_times:
        closest = min(of_times, key=lambda t: abs(t - target_t))
        mf_actual[closest] = os.path.join(case_dir, str(closest), 'alpha.water')
        print(f"  MF target t={target_t:.1f} -> actual t={closest:.3f}")

    # Use read_openfoam_field for each time, then call multifractal analysis
    # This requires the analyze_multifractal method to accept data dicts
    # For now, note that this needs the OpenFOAM data to be present
    print("  Phase 2 OpenFOAM: deferred until runs are available.")


def _run_phase3_openfoam(analyzer, physics, case_dir, of_times, analysis_dir, spec_times, df):
    """Phase 3 for OpenFOAM cases."""
    print("Phase 3 (OpenFOAM): deferred until runs are available.")


if __name__ == '__main__':
    main()
