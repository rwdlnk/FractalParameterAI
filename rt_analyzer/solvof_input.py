"""
SOLA-VOF Input File Parser

Parses int*.in input files used by the SOLA-VOF Rayleigh-Taylor simulation code.
Extracts physics parameters (densities, gravity, viscosity), grid specification,
domain extents, boundary conditions, and initial interface position.

File format:
  - KEYWORD(FORMAT_SPEC) = VALUE lines for most parameters
  - Columnar data for grid zone specification (-XC-!, -YC-!, -NKX!)
  - Free-format arrays for domain extents (XL, YL)
  - Free-format disturbance parameters (R, H, ILOW, IHIGH)
"""

import re
import os
import glob
from typing import Dict, Optional

# Regex for KEYWORD(FORMAT) = VALUE lines
_KV_RE = re.compile(
    r'^([A-Za-z][A-Za-z0-9]*)\s*(?:\([^)]*\))?\s*=\s*(.+?)\s*$',
    re.IGNORECASE
)

# Parameters that should be parsed as floats
_FLOAT_KEYS = {
    'FLENG', 'FTIME', 'NU', 'NUC', 'EPSI', 'GX', 'GY', 'UI', 'VI',
    'VELMX', 'OMG', 'ALPHA', 'CSQ', 'AUTOT', 'SIGMA', 'CANGLE',
    'RHOF', 'RHOFC', 'FLHT', 'XPL', 'YPB', 'XPR', 'YPT',
    'VORIG', 'SHVEL', 'DELT', 'TWFIN', 'PRTDT',
    'PLTST1', 'PLTST2', 'PLTDT1', 'PLTDT2'
}

# Parameters that should be parsed as ints
_INT_KEYS = {
    'IPLOT', 'ICYL', 'IMOVY', 'WL', 'WR', 'WT', 'WB',
    'ISYMPLT', 'ISURF10', 'NMAT', 'NPX', 'NPY', 'NOBS', 'NDIS',
    'IDIS', 'ISHVEL', 'IARE'
}


def parse_solvof_input(filepath: str) -> Dict:
    """Parse a SOLA-VOF input file and return all parameters.

    Args:
        filepath: Path to the int*.in file

    Returns:
        Dictionary with all parsed parameters. Keys are uppercase.
        Includes special keys: NKX, NKY, NONU, XC, NXL, NXR, DXMN,
        YC, NYL, NYR, DYMN, XL (list), YL (list), H0 (interface height),
        IFL, IFR, JFB, JFT, DIST_R, DIST_H, DIST_ILOW, DIST_IHIGH
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    params = {}
    params['_filepath'] = filepath
    i = 0

    while i < len(lines):
        line = lines[i].rstrip('\n')
        stripped = line.strip()

        # Skip blank lines and pure comment lines
        if not stripped or stripped.startswith('!'):
            # Check for special comment-based sections
            upper = stripped.upper()

            if 'VALUES FOR XL' in upper:
                # Skip ruler line, read data
                i += 2
                if i < len(lines):
                    params['XL'] = [float(v) for v in lines[i].split()]
                i += 1
                continue

            if 'VALUES FOR YL' in upper:
                i += 2
                if i < len(lines):
                    params['YL'] = [float(v) for v in lines[i].split()]
                i += 1
                continue

            if 'DISTURBANCE PARAMETERS' in upper:
                # Skip ruler line, read R, H, ILOW, IHIGH
                i += 2
                if i < len(lines):
                    vals = lines[i].split()
                    if len(vals) >= 4:
                        params['DIST_R'] = float(vals[0])
                        params['DIST_H'] = float(vals[1])
                        params['DIST_ILOW'] = int(vals[2])
                        params['DIST_IHIGH'] = int(vals[3])
                i += 1
                continue

            i += 1
            continue

        # Grid zone headers (start with -)
        if stripped.startswith('-NKX') or stripped.startswith('-nkx'):
            i += 1
            if i < len(lines):
                vals = lines[i].split()
                params['NKX'] = int(vals[0])
                params['NKY'] = int(vals[1])
                params['NONU'] = int(vals[2])
            i += 1
            continue

        if stripped.startswith('-XC') or stripped.startswith('-xc'):
            i += 1
            if i < len(lines):
                vals = lines[i].split()
                params['XC'] = float(vals[0])
                params['NXL'] = int(vals[1])
                params['NXR'] = int(vals[2])
                params['DXMN'] = float(vals[3])
            i += 1
            continue

        if stripped.startswith('-YC') or stripped.startswith('-yc'):
            i += 1
            if i < len(lines):
                vals = lines[i].split()
                params['YC'] = float(vals[0])
                params['NYL'] = int(vals[1])
                params['NYR'] = int(vals[2])
                params['DYMN'] = float(vals[3])
            i += 1
            continue

        if stripped.startswith('-IFL') or stripped.startswith('-ifl'):
            i += 1
            if i < len(lines):
                vals = lines[i].split()
                params['IFL'] = int(vals[0])
                params['IFR'] = int(vals[1])
                params['JFB'] = int(vals[2])
                params['JFT'] = int(vals[3])
            i += 1
            continue

        # Try KEYWORD(FORMAT) = VALUE pattern
        m = _KV_RE.match(stripped)
        if m:
            key = m.group(1).upper()
            val_str = m.group(2).strip()

            if key in _FLOAT_KEYS:
                try:
                    params[key] = float(val_str)
                except ValueError:
                    params[key] = val_str
            elif key in _INT_KEYS:
                try:
                    params[key] = int(val_str)
                except ValueError:
                    params[key] = val_str
            else:
                # Store as string; caller can cast as needed
                params[key] = val_str

        i += 1

    return params


def derive_physics(params: Dict) -> Dict:
    """Compute derived physics quantities from parsed input parameters.

    Args:
        params: Dictionary from parse_solvof_input()

    Returns:
        Dictionary with derived quantities:
          A     - Atwood number
          g     - Gravitational acceleration (positive, m/s^2)
          H     - Domain height (m)
          L     - Domain width (m)
          H0    - Initial interface height (m)
          NX    - Total cells in x (NXL + NXR)
          NY    - Total cells in y (NYL + NYR)
          dx    - Cell width (m)
          dy    - Cell height (m)
          RHOF  - Heavy fluid density (kg/m^3)
          RHOFC - Light fluid density (kg/m^3)
          NU    - Heavy fluid kinematic viscosity (m^2/s)
    """
    rhof = params['RHOF']
    rhofc = params['RHOFC']
    A = (rhof - rhofc) / (rhof + rhofc)
    g = abs(params['GY'])

    # Domain extents from XL, YL arrays
    xl = params.get('XL', [0.0, 0.4])
    yl = params.get('YL', [0.0, 0.5])
    L = xl[-1] - xl[0]
    H = yl[-1] - yl[0]

    # Initial interface height from disturbance parameters or YC
    H0 = params.get('DIST_H', params.get('YC', H / 2.0))

    # Grid
    NX = params['NXL'] + params['NXR']
    NY = params['NYL'] + params['NYR']
    dx = params['DXMN']
    dy = params['DYMN']

    return {
        'A': A,
        'g': g,
        'H': H,
        'L': L,
        'H0': H0,
        'NX': NX,
        'NY': NY,
        'dx': dx,
        'dy': dy,
        'RHOF': rhof,
        'RHOFC': rhofc,
        'NU': params.get('NU', 1.0e-6),
    }


def find_input_file(case_dir: str) -> Optional[str]:
    """Auto-detect the SOLA-VOF input file in a case directory.

    Looks for int*.in files. Returns the first match, or None.
    """
    matches = sorted(glob.glob(os.path.join(case_dir, 'int*.in')))
    return matches[0] if matches else None


def summary(params: Dict, phys: Dict) -> str:
    """Return a human-readable summary of parsed parameters."""
    lines = [
        f"Input file: {params.get('_filepath', 'unknown')}",
        f"Grid: {phys['NX']}x{phys['NY']} cells, dx={phys['dx']:.6f} m, dy={phys['dy']:.6f} m",
        f"Domain: L={phys['L']:.4f} m x H={phys['H']:.4f} m",
        f"Interface: H0={phys['H0']:.4f} m",
        f"Densities: rho_heavy={phys['RHOF']:.4f}, rho_light={phys['RHOFC']:.4f} kg/m^3",
        f"Atwood number: A={phys['A']:.6f}",
        f"Gravity: g={phys['g']:.4f} m/s^2",
        f"Viscosity: nu={phys['NU']:.6e} m^2/s",
    ]
    return '\n'.join(lines)
