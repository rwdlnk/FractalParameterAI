#!/usr/bin/env python3
"""
RT Mixing Analysis

Analyzes mixing characteristics in Rayleigh-Taylor instability simulations:
- Mixing zone width and boundaries
- Mixed/unmixed fractions
- Mixing efficiency
- Concentration variance
- Segregation index

This module is part of the FractalParameterAI framework.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class MixingStatistics:
    """Container for mixing analysis results."""
    # Mixing zone geometry
    mixing_zone_width: float  # Total width of mixing zone
    upper_boundary: float  # Y-coordinate of upper boundary (F ≈ 0.95)
    lower_boundary: float  # Y-coordinate of lower boundary (F ≈ 0.05)
    interface_position: float  # Y-coordinate of interface (F ≈ 0.5)

    # Fractions
    mixed_fraction: float  # Fraction of domain that is mixed (0.1 < F < 0.9)
    unmixed_light_fraction: float  # Fraction that is pure light fluid (F < 0.05)
    unmixed_heavy_fraction: float  # Fraction that is pure heavy fluid (F > 0.95)

    # Mixing quality metrics
    mixing_efficiency: float  # Efficiency of mixing (0 = unmixed, 1 = perfect)
    mixing_width_integral: float  # Dalziel's h_{1,1} = 2 × ∫ C̄(1-C̄) dz [meters]
    concentration_variance: float  # Variance of VOF field
    segregation_index: float  # Danckwerts segregation index

    # Vertical profiles
    mean_f_profile: np.ndarray  # Mean F(y) profile
    variance_f_profile: np.ndarray  # Variance of F at each y
    y_coords: np.ndarray  # Y-coordinates for profiles

    # Analysis metadata
    grid_shape: Tuple[int, int]
    n_points: int
    analysis_time: float


class MixingAnalyzer:
    """
    Analyzes mixing characteristics in RT simulations.

    Uses volume fraction field F to quantify mixing:
    - F = 0: Pure light fluid
    - F = 1: Pure heavy fluid
    - 0 < F < 1: Mixed region
    """

    def __init__(self,
                 mixed_threshold_low: float = 0.1,
                 mixed_threshold_high: float = 0.9,
                 debug: bool = False):
        """
        Initialize mixing analyzer.

        Args:
            mixed_threshold_low: Lower threshold for "mixed" region (default 0.1)
            mixed_threshold_high: Upper threshold for "mixed" region (default 0.9)
            debug: Enable debug output
        """
        self.mixed_threshold_low = mixed_threshold_low
        self.mixed_threshold_high = mixed_threshold_high
        self.debug = debug

    def analyze_mixing(self,
                      f_grid: np.ndarray,
                      y_grid: np.ndarray) -> MixingStatistics:
        """
        Compute comprehensive mixing statistics.

        Args:
            f_grid: Volume fraction field (2D array)
            y_grid: Y-coordinate grid (2D array)

        Returns:
            MixingStatistics object with computed metrics
        """
        start_time = time.time()

        if self.debug:
            print(f"   📊 Computing mixing statistics...")
            print(f"      Grid shape: {f_grid.shape}")
            print(f"      F range: [{np.min(f_grid):.3f}, {np.max(f_grid):.3f}]")

        grid_shape = f_grid.shape
        n_points = f_grid.size

        # ============================================
        # 1. Compute vertical profiles
        # ============================================
        # Average F along x-direction to get F(y) profile
        mean_f_profile = np.mean(f_grid, axis=0)  # Average along x (axis 0)
        variance_f_profile = np.var(f_grid, axis=0)  # Variance along x

        # Get y-coordinates (take first column since y is uniform in x)
        y_coords = y_grid[0, :]

        if self.debug:
            print(f"      Profile points: {len(y_coords)}")

        # ============================================
        # 2. Find mixing zone boundaries
        # ============================================
        # Find where F is in the mixed range (between 0.05 and 0.95)
        # The mixing zone is where 0.05 < F < 0.95

        # Find all indices where F is in mixed range
        mixed_indices = np.where((mean_f_profile > 0.05) & (mean_f_profile < 0.95))[0]

        if len(mixed_indices) > 0:
            # Lower boundary: minimum y where F is in mixed range
            lower_boundary = y_coords[mixed_indices[0]]
            # Upper boundary: maximum y where F is in mixed range
            upper_boundary = y_coords[mixed_indices[-1]]
            mixing_zone_width = upper_boundary - lower_boundary
        else:
            # Fallback if no mixed region found
            upper_boundary = np.max(y_coords)
            lower_boundary = np.min(y_coords)
            mixing_zone_width = 0.0

        # Interface position (F ≈ 0.5)
        interface_idx = np.argmin(np.abs(mean_f_profile - 0.5))
        interface_position = y_coords[interface_idx]

        if self.debug:
            print(f"      Mixing zone: y ∈ [{lower_boundary:.4f}, {upper_boundary:.4f}]")
            print(f"      Width: {mixing_zone_width:.4f}")
            print(f"      Interface at y = {interface_position:.4f}")

        # ============================================
        # 3. Compute mixed/unmixed fractions
        # ============================================
        # Mixed: cells where mixed_threshold_low < F < mixed_threshold_high
        mixed_mask = (f_grid > self.mixed_threshold_low) & (f_grid < self.mixed_threshold_high)
        mixed_fraction = np.sum(mixed_mask) / n_points

        # Unmixed light: F < 0.05
        unmixed_light_mask = f_grid < 0.05
        unmixed_light_fraction = np.sum(unmixed_light_mask) / n_points

        # Unmixed heavy: F > 0.95
        unmixed_heavy_mask = f_grid > 0.95
        unmixed_heavy_fraction = np.sum(unmixed_heavy_mask) / n_points

        if self.debug:
            print(f"      Mixed fraction: {mixed_fraction:.3f}")
            print(f"      Unmixed light: {unmixed_light_fraction:.3f}")
            print(f"      Unmixed heavy: {unmixed_heavy_fraction:.3f}")

        # ============================================
        # 4. Compute mixing efficiency
        # ============================================
        # Mixing efficiency: based on how close F distribution is to uniform
        # Perfect mixing would have all F ≈ 0.5 (assuming equal volumes initially)
        # We use normalized variance: η = 1 - σ²/σ²_max
        # where σ²_max is variance of completely unmixed state

        mean_f = np.mean(f_grid)
        variance_f = np.var(f_grid)

        # Maximum variance occurs when half domain is F=0, half is F=1
        # For binary mixture: σ²_max = mean_f * (1 - mean_f)
        variance_max = mean_f * (1 - mean_f)

        if variance_max > 1e-12:
            mixing_efficiency = 1.0 - (variance_f / variance_max)
        else:
            mixing_efficiency = 1.0  # Already perfectly mixed

        # Clamp to [0, 1]
        mixing_efficiency = np.clip(mixing_efficiency, 0.0, 1.0)

        if self.debug:
            print(f"      Mixing efficiency: {mixing_efficiency:.4f}")

        # ============================================
        # 4.5. Compute mixing width integral (Youngs' W, Dalziel's h_{1,1})
        # ============================================
        # Youngs (1984) Eq. (3): W = ∫ C̄(1-C̄) dz (mixedness)
        #
        # Dalziel (1999) Eq. (6): h_Integral = ∫_{-H/2}^{H/2} C̄(1-C̄) dz
        # Dalziel (1999) Eq. (7): h_{m,n} = [(m+n)^(m+n)] / [m^n × n^m] × ∫_{-H/2}^{0} C̄^m(1-C̄)^n dz
        #
        # For m = n = 1:
        #   h_{1,1} = [2^2] / [1^1 × 1^1] × ∫_{-H/2}^{0} C̄(1-C̄) dz
        #   h_{1,1} = 4 ∫_{-H/2}^{0} C̄(1-C̄) dz
        #
        # Assuming symmetry about the initial interface:
        #   h_{1,1} = 4 ∫_{-H/2}^{0} C̄(1-C̄) dz = 2 ∫_{-H/2}^{H/2} C̄(1-C̄) dz = 2W
        #
        # Our domain is 0 to H, equivalent to Dalziel's -H/2 to H/2:
        #   W = ∫_{0}^{H} C̄(1-C̄) dy
        #   h_{1,1} = 2W
        #
        integrand = mean_f_profile * (1.0 - mean_f_profile)
        W = np.trapz(integrand, y_coords)
        mixing_width_integral = 2.0 * W  # Apply Dalziel's factor of 2 for h_{1,1}

        if self.debug:
            print(f"      Youngs' W: {W:.6f} m")
            print(f"      Dalziel h_{{1,1}}: {mixing_width_integral:.6f} m")

        # ============================================
        # 5. Compute segregation index (Danckwerts)
        # ============================================
        # Segregation index: I = σ²/σ²_max
        # I = 0: perfectly mixed
        # I = 1: completely segregated

        if variance_max > 1e-12:
            segregation_index = variance_f / variance_max
        else:
            segregation_index = 0.0

        segregation_index = np.clip(segregation_index, 0.0, 1.0)

        if self.debug:
            print(f"      Segregation index: {segregation_index:.4f}")

        # ============================================
        # 6. Package results
        # ============================================
        analysis_time = time.time() - start_time

        statistics = MixingStatistics(
            mixing_zone_width=mixing_zone_width,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            interface_position=interface_position,
            mixed_fraction=mixed_fraction,
            unmixed_light_fraction=unmixed_light_fraction,
            unmixed_heavy_fraction=unmixed_heavy_fraction,
            mixing_efficiency=mixing_efficiency,
            mixing_width_integral=mixing_width_integral,
            concentration_variance=variance_f,
            segregation_index=segregation_index,
            mean_f_profile=mean_f_profile,
            variance_f_profile=variance_f_profile,
            y_coords=y_coords,
            grid_shape=grid_shape,
            n_points=n_points,
            analysis_time=analysis_time
        )

        if self.debug:
            print(f"      ✅ Analysis complete in {analysis_time:.4f}s")

        return statistics

    def analyze_mixing_from_vtk(self, vtk_data) -> Optional[MixingStatistics]:
        """
        Analyze mixing from VTKData object.

        Args:
            vtk_data: VTKData object containing VOF field

        Returns:
            MixingStatistics object or None if VOF field not available
        """
        # Check for VOF field (handle both lowercase and uppercase)
        f_key = None

        for key in vtk_data.scalar_fields:
            if key.lower() == 'f':
                f_key = key
                break

        if f_key is None:
            if self.debug:
                print(f"   ⚠️  VOF field not available")
                print(f"      Available fields: {list(vtk_data.scalar_fields.keys())}")
            return None

        f_grid = vtk_data.scalar_fields[f_key]
        y_grid = vtk_data.y_grid

        return self.analyze_mixing(f_grid, y_grid)

    def compute_growth_rate(self,
                           times: np.ndarray,
                           widths: np.ndarray) -> Dict:
        """
        Compute mixing zone growth rate from temporal data.

        For RT instability: h(t) ~ α * A * g * t²
        where α is the growth rate coefficient, A is Atwood number, g is gravity

        Args:
            times: Array of time values
            widths: Array of mixing zone widths

        Returns:
            Dictionary with growth rate analysis
        """
        if len(times) < 3:
            return {
                'error': 'Insufficient data points for growth rate analysis'
            }

        # Fit power law: h = c * t^n
        log_t = np.log(times)
        log_h = np.log(widths)

        # Linear fit in log-log space
        coeffs = np.polyfit(log_t, log_h, 1)
        exponent = coeffs[0]
        log_c = coeffs[1]
        c = np.exp(log_c)

        # Compute R²
        predicted = exponent * log_t + log_c
        r_squared = 1 - np.sum((log_h - predicted)**2) / np.sum((log_h - np.mean(log_h))**2)

        # For RT: h ~ t² in nonlinear regime, so exponent should be ~2
        return {
            'growth_exponent': exponent,
            'growth_coefficient': c,
            'r_squared': r_squared,
            'fit_formula': f'h = {c:.4e} * t^{exponent:.3f}',
            'regime': 'quadratic (nonlinear)' if abs(exponent - 2.0) < 0.3 else 'non-quadratic'
        }


def main():
    """Test mixing analysis."""
    import argparse
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from integration.rt_vtk_parser import VTKParser

    parser = argparse.ArgumentParser(description='Test mixing analysis')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Parse VTK file
    print(f"📂 Loading VTK file...")
    vtk_parser = VTKParser(debug=args.debug)
    vtk_data = vtk_parser.parse_vtk_file(args.vtk_file)

    if not vtk_data:
        print(f"❌ Failed to parse VTK file")
        return

    # Analyze mixing
    print(f"\n🧪 Analyzing mixing...")
    analyzer = MixingAnalyzer(debug=args.debug)
    stats = analyzer.analyze_mixing_from_vtk(vtk_data)

    if not stats:
        print(f"❌ Failed to analyze mixing")
        return

    # Print results
    print(f"\n{'='*70}")
    print(f"🎉 MIXING ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\n📏 Mixing Zone Geometry:")
    print(f"   Width: {stats.mixing_zone_width:.6f}")
    print(f"   Upper boundary (F=0.95): y = {stats.upper_boundary:.6f}")
    print(f"   Interface (F=0.5): y = {stats.interface_position:.6f}")
    print(f"   Lower boundary (F=0.05): y = {stats.lower_boundary:.6f}")

    print(f"\n📊 Fractions:")
    print(f"   Mixed (0.1 < F < 0.9): {stats.mixed_fraction:.4f} ({stats.mixed_fraction*100:.2f}%)")
    print(f"   Unmixed light (F < 0.05): {stats.unmixed_light_fraction:.4f} ({stats.unmixed_light_fraction*100:.2f}%)")
    print(f"   Unmixed heavy (F > 0.95): {stats.unmixed_heavy_fraction:.4f} ({stats.unmixed_heavy_fraction*100:.2f}%)")

    print(f"\n🎯 Mixing Quality:")
    print(f"   Mixing efficiency: {stats.mixing_efficiency:.4f}")
    print(f"   Segregation index: {stats.segregation_index:.4f}")
    print(f"   Concentration variance: {stats.concentration_variance:.6e}")

    print(f"\n📐 Grid Info:")
    print(f"   Grid shape: {stats.grid_shape}")
    print(f"   Total points: {stats.n_points}")
    print(f"   Analysis time: {stats.analysis_time:.4f}s")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
