"""
Rayleigh-Taylor Physics Nondimensionalization Module.

Centralizes all RT instability nondimensionalization for consistent
use across fractal and multifractal analysis toolkits.

Dimensionless variables (Dalziel et al., 1999):
    - Time:       tau = t * sqrt(A*g/H)
    - Length:     x/H (vertical), x/L (horizontal)
    - Box size:   delta/H
    - Wavenumber: k = 2*pi*f*L (normalized by domain width)
    - Velocity:   u / sqrt(A*g*H)

where A = Atwood number, g = gravity, H = domain height, L = domain width.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union

ArrayLike = Union[float, np.ndarray, list]


@dataclass
class RTPhysics:
    """RT instability physics parameters and nondimensionalization.

    Attributes:
        A: Atwood number (rho2 - rho1) / (rho2 + rho1)
        g: Gravitational acceleration (m/s^2)
        H: Domain height (m)
        L: Domain width (m)
    """
    A: float
    g: float
    H: float
    L: float

    # Derived quantities (computed in __post_init__)
    tau_factor: float = field(init=False, repr=False)
    velocity_scale: float = field(init=False, repr=False)
    time_scale: float = field(init=False, repr=False)

    def __post_init__(self):
        if self.A <= 0 or self.A > 1:
            raise ValueError(f"Atwood number must be in (0, 1], got {self.A}")
        if self.g <= 0:
            raise ValueError(f"Gravity must be positive, got {self.g}")
        if self.H <= 0:
            raise ValueError(f"Domain height must be positive, got {self.H}")
        if self.L <= 0:
            raise ValueError(f"Domain width must be positive, got {self.L}")

        self.tau_factor = np.sqrt(self.A * self.g / self.H)
        self.velocity_scale = np.sqrt(self.A * self.g * self.H)
        self.time_scale = np.sqrt(self.H / (self.A * self.g))

    # --- Nondimensionalization ---

    def nondim_time(self, t: ArrayLike) -> np.ndarray:
        """Convert physical time to dimensionless time tau = t * sqrt(Ag/H)."""
        return np.asarray(t) * self.tau_factor

    def nondim_length(self, x: ArrayLike, scale: str = 'H') -> np.ndarray:
        """Normalize length by H (vertical) or L (horizontal).

        Args:
            x: Physical length(s)
            scale: 'H' for vertical normalization, 'L' for horizontal
        """
        if scale == 'H':
            return np.asarray(x) / self.H
        elif scale == 'L':
            return np.asarray(x) / self.L
        else:
            raise ValueError(f"scale must be 'H' or 'L', got '{scale}'")

    def nondim_box_size(self, delta: ArrayLike) -> np.ndarray:
        """Normalize box/cube size by domain height: delta/H."""
        return np.asarray(delta) / self.H

    def nondim_wavenumber(self, f: ArrayLike) -> np.ndarray:
        """Convert spatial frequency to dimensionless wavenumber k = 2*pi*f*L."""
        return 2.0 * np.pi * np.asarray(f) * self.L

    def nondim_velocity(self, u: ArrayLike) -> np.ndarray:
        """Normalize velocity by sqrt(AgH)."""
        return np.asarray(u) / self.velocity_scale

    # --- Inverse (re-dimensionalization) ---

    def dim_time(self, tau: ArrayLike) -> np.ndarray:
        """Convert dimensionless time back to physical time."""
        return np.asarray(tau) / self.tau_factor

    def dim_length(self, x_star: ArrayLike, scale: str = 'H') -> np.ndarray:
        """Convert normalized length back to physical length."""
        if scale == 'H':
            return np.asarray(x_star) * self.H
        elif scale == 'L':
            return np.asarray(x_star) * self.L
        else:
            raise ValueError(f"scale must be 'H' or 'L', got '{scale}'")

    def dim_box_size(self, delta_star: ArrayLike) -> np.ndarray:
        """Convert normalized box size back to physical size."""
        return np.asarray(delta_star) * self.H

    # --- Factory methods ---

    @classmethod
    def from_dalziel(cls, A: float = 0.5, g: float = 9.81,
                     H: float = 0.5, L: float = 0.4) -> 'RTPhysics':
        """Create with Dalziel (1999) experimental defaults.

        Default parameters: A=0.5, g=9.81 m/s^2, H=0.5 m, L=0.4 m.
        """
        return cls(A=A, g=g, H=H, L=L)

    @classmethod
    def from_vtk_file(cls, vtk_path: str, A: float = 0.5,
                      g: float = 9.81) -> 'RTPhysics':
        """Auto-detect H and L from a VTK rectilinear grid file.

        Args:
            vtk_path: Path to a VTK file
            A: Atwood number
            g: Gravitational acceleration
        """
        H, L = _read_domain_from_vtk(vtk_path)
        return cls(A=A, g=g, H=H, L=L)

    def summary(self) -> str:
        """Return a formatted summary of the physics parameters."""
        lines = [
            f"RT Physics Parameters:",
            f"  A = {self.A:.4f}  (Atwood number)",
            f"  g = {self.g:.4f}  m/s^2",
            f"  H = {self.H:.4f}  m  (domain height)",
            f"  L = {self.L:.4f}  m  (domain width)",
            f"  tau_factor = sqrt(Ag/H) = {self.tau_factor:.4f}  s^-1",
            f"  velocity_scale = sqrt(AgH) = {self.velocity_scale:.4f}  m/s",
            f"  time_scale = sqrt(H/(Ag)) = {self.time_scale:.4f}  s",
        ]
        return "\n".join(lines)


def _read_domain_from_vtk(vtk_path: str):
    """Extract domain height H and width L from VTK rectilinear grid.

    Returns:
        (H, L): Domain height and width in meters
    """
    with open(vtk_path, 'r') as f:
        lines = f.readlines()

    x_coords = []
    y_coords = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("X_COORDINATES"):
            n = int(line.split()[1])
            coords = []
            i += 1
            while len(coords) < n:
                coords.extend(float(v) for v in lines[i].strip().split())
                i += 1
            x_coords = coords
            continue
        elif line.startswith("Y_COORDINATES"):
            n = int(line.split()[1])
            coords = []
            i += 1
            while len(coords) < n:
                coords.extend(float(v) for v in lines[i].strip().split())
                i += 1
            y_coords = coords
            continue
        i += 1

    if not x_coords or not y_coords:
        raise ValueError(f"Could not extract coordinates from {vtk_path}")

    L = max(x_coords) - min(x_coords)
    H = max(y_coords) - min(y_coords)
    return H, L
