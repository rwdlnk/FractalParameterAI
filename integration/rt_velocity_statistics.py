#!/usr/bin/env python3
"""
RT Velocity Statistics

Computes velocity field statistics for RT simulation data:
- Spatial mean velocities ⟨u⟩, ⟨v⟩
- RMS velocities u_rms, v_rms (relative to mean)
- Turbulent kinetic energy (TKE)
- Reynolds stress ⟨u'v'⟩
- Velocity magnitude statistics

This module is part of the FractalParameterAI framework.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class VelocityStatistics:
    """Container for velocity field statistics."""
    # Mean velocities
    u_mean: float  # Spatial average of u
    v_mean: float  # Spatial average of v

    # RMS velocities (relative to mean)
    u_rms: float  # RMS of u' = u - ⟨u⟩
    v_rms: float  # RMS of v' = v - ⟨v⟩

    # Magnitude statistics
    velocity_magnitude_mean: float  # Mean |V| = sqrt(u² + v²)
    velocity_magnitude_rms: float  # RMS of |V|
    velocity_magnitude_max: float  # Maximum |V|

    # Turbulence quantities
    turbulent_kinetic_energy: float  # TKE = 0.5 * (u_rms² + v_rms²)
    reynolds_stress: float  # ⟨u'v'⟩ correlation
    turbulence_intensity: float  # sqrt(TKE) / |⟨V⟩|

    # Domain info
    n_points: int  # Number of grid points
    grid_shape: Tuple[int, int]  # Grid dimensions

    # Analysis metadata
    analysis_time: float  # Time taken for analysis


class VelocityAnalyzer:
    """
    Analyzes velocity field statistics from RT simulation data.

    Computes spatial statistics of velocity fields with proper handling
    of mean flow (important for non-zero mean flow cases).
    """

    def __init__(self, debug: bool = False):
        """
        Initialize velocity analyzer.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def analyze_velocity_field(self,
                               u_grid: np.ndarray,
                               v_grid: np.ndarray) -> VelocityStatistics:
        """
        Compute comprehensive velocity field statistics.

        Args:
            u_grid: Horizontal velocity component (2D array)
            v_grid: Vertical velocity component (2D array)

        Returns:
            VelocityStatistics object with computed statistics
        """
        start_time = time.time()

        if self.debug:
            print(f"   📊 Computing velocity statistics...")
            print(f"      Grid shape: {u_grid.shape}")

        # Verify grids match
        if u_grid.shape != v_grid.shape:
            raise ValueError(f"u_grid and v_grid shapes must match: {u_grid.shape} vs {v_grid.shape}")

        grid_shape = u_grid.shape
        n_points = u_grid.size

        # ============================================
        # 1. Compute spatial mean velocities
        # ============================================
        u_mean = np.mean(u_grid)
        v_mean = np.mean(v_grid)

        if self.debug:
            print(f"      ⟨u⟩ = {u_mean:.6e}")
            print(f"      ⟨v⟩ = {v_mean:.6e}")

        # ============================================
        # 2. Compute velocity fluctuations
        # ============================================
        u_prime = u_grid - u_mean  # u' = u - ⟨u⟩
        v_prime = v_grid - v_mean  # v' = v - ⟨v⟩

        # ============================================
        # 3. Compute RMS velocities (WRT mean)
        # ============================================
        u_rms = np.sqrt(np.mean(u_prime**2))
        v_rms = np.sqrt(np.mean(v_prime**2))

        if self.debug:
            print(f"      u_rms = {u_rms:.6e}")
            print(f"      v_rms = {v_rms:.6e}")

        # ============================================
        # 4. Compute velocity magnitude statistics
        # ============================================
        # Magnitude at each point
        velocity_magnitude = np.sqrt(u_grid**2 + v_grid**2)

        velocity_magnitude_mean = np.mean(velocity_magnitude)
        velocity_magnitude_rms = np.sqrt(np.mean(velocity_magnitude**2))
        velocity_magnitude_max = np.max(velocity_magnitude)

        if self.debug:
            print(f"      |V|_mean = {velocity_magnitude_mean:.6e}")
            print(f"      |V|_max = {velocity_magnitude_max:.6e}")

        # ============================================
        # 5. Compute turbulent kinetic energy
        # ============================================
        # TKE = 0.5 * (⟨u'²⟩ + ⟨v'²⟩)
        # Note: For 2D flow, this is the in-plane TKE
        tke = 0.5 * (u_rms**2 + v_rms**2)

        if self.debug:
            print(f"      TKE = {tke:.6e}")

        # ============================================
        # 6. Compute Reynolds stress
        # ============================================
        # Reynolds stress: ⟨u'v'⟩ = correlation between u and v fluctuations
        reynolds_stress = np.mean(u_prime * v_prime)

        if self.debug:
            print(f"      ⟨u'v'⟩ = {reynolds_stress:.6e}")

        # ============================================
        # 7. Compute turbulence intensity
        # ============================================
        # Turbulence intensity: sqrt(TKE) / |⟨V⟩|
        # This is undefined if mean velocity is zero
        mean_velocity_magnitude = np.sqrt(u_mean**2 + v_mean**2)

        if mean_velocity_magnitude > 1e-12:
            turbulence_intensity = np.sqrt(tke) / mean_velocity_magnitude
        else:
            # For RT with zero mean flow, use alternate definition:
            # TI = sqrt(TKE) / V_rms where V_rms is characteristic velocity
            if velocity_magnitude_rms > 1e-12:
                turbulence_intensity = np.sqrt(tke) / velocity_magnitude_rms
            else:
                turbulence_intensity = 0.0

        if self.debug:
            print(f"      Turbulence intensity = {turbulence_intensity:.6f}")

        # ============================================
        # 8. Package results
        # ============================================
        analysis_time = time.time() - start_time

        statistics = VelocityStatistics(
            u_mean=u_mean,
            v_mean=v_mean,
            u_rms=u_rms,
            v_rms=v_rms,
            velocity_magnitude_mean=velocity_magnitude_mean,
            velocity_magnitude_rms=velocity_magnitude_rms,
            velocity_magnitude_max=velocity_magnitude_max,
            turbulent_kinetic_energy=tke,
            reynolds_stress=reynolds_stress,
            turbulence_intensity=turbulence_intensity,
            n_points=n_points,
            grid_shape=grid_shape,
            analysis_time=analysis_time
        )

        if self.debug:
            print(f"      ✅ Analysis complete in {analysis_time:.4f}s")

        return statistics

    def analyze_velocity_field_from_vtk(self, vtk_data) -> Optional[VelocityStatistics]:
        """
        Analyze velocity field from VTKData object.

        Args:
            vtk_data: VTKData object containing velocity fields

        Returns:
            VelocityStatistics object or None if velocity fields not available
        """
        # Check for velocity fields (handle both lowercase and uppercase)
        u_key = None
        v_key = None

        for key in vtk_data.scalar_fields:
            if key.lower() == 'u':
                u_key = key
            elif key.lower() == 'v':
                v_key = key

        if u_key is None or v_key is None:
            if self.debug:
                print(f"   ⚠️  Velocity fields not available")
                print(f"      Available fields: {list(vtk_data.scalar_fields.keys())}")
            return None

        u_grid = vtk_data.scalar_fields[u_key]
        v_grid = vtk_data.scalar_fields[v_key]

        return self.analyze_velocity_field(u_grid, v_grid)

    def compute_spatial_profiles(self,
                                 u_grid: np.ndarray,
                                 v_grid: np.ndarray,
                                 direction: str = 'y') -> Dict:
        """
        Compute spatial profiles of velocity statistics.

        Useful for analyzing vertical/horizontal profiles in RT.

        Args:
            u_grid: Horizontal velocity component
            v_grid: Vertical velocity component
            direction: 'x' or 'y' for profile direction

        Returns:
            Dictionary with profile data
        """
        if direction == 'y':
            # Average along x-direction to get y-profiles
            axis = 0
        elif direction == 'x':
            # Average along y-direction to get x-profiles
            axis = 1
        else:
            raise ValueError(f"direction must be 'x' or 'y', got '{direction}'")

        # Compute mean profiles
        u_mean = np.mean(u_grid - np.mean(u_grid))
        v_mean = np.mean(v_grid - np.mean(v_grid))

        u_profile = np.mean(u_grid, axis=axis)
        v_profile = np.mean(v_grid, axis=axis)

        # Compute RMS profiles
        u_prime = u_grid - np.mean(u_grid)
        v_prime = v_grid - np.mean(v_grid)

        u_rms_profile = np.sqrt(np.mean(u_prime**2, axis=axis))
        v_rms_profile = np.sqrt(np.mean(v_prime**2, axis=axis))

        # TKE profile
        tke_profile = 0.5 * (u_rms_profile**2 + v_rms_profile**2)

        # Reynolds stress profile
        reynolds_stress_profile = np.mean(u_prime * v_prime, axis=axis)

        return {
            'direction': direction,
            'u_profile': u_profile,
            'v_profile': v_profile,
            'u_rms_profile': u_rms_profile,
            'v_rms_profile': v_rms_profile,
            'tke_profile': tke_profile,
            'reynolds_stress_profile': reynolds_stress_profile,
            'n_points': len(u_profile)
        }


def main():
    """Test velocity statistics analyzer."""
    import argparse
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from integration.rt_vtk_parser import VTKParser

    parser = argparse.ArgumentParser(description='Test velocity statistics')
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

    # Analyze velocity statistics
    print(f"\n📊 Analyzing velocity statistics...")
    analyzer = VelocityAnalyzer(debug=args.debug)
    stats = analyzer.analyze_velocity_field_from_vtk(vtk_data)

    if not stats:
        print(f"❌ Failed to analyze velocity statistics")
        return

    # Print results
    print(f"\n{'='*70}")
    print(f"🎉 VELOCITY STATISTICS RESULTS")
    print(f"{'='*70}")
    print(f"\n📍 Mean Velocities:")
    print(f"   ⟨u⟩ = {stats.u_mean:.6e}")
    print(f"   ⟨v⟩ = {stats.v_mean:.6e}")

    print(f"\n📊 RMS Velocities (relative to mean):")
    print(f"   u_rms = {stats.u_rms:.6e}")
    print(f"   v_rms = {stats.v_rms:.6e}")

    print(f"\n🌊 Velocity Magnitude:")
    print(f"   |V|_mean = {stats.velocity_magnitude_mean:.6e}")
    print(f"   |V|_rms = {stats.velocity_magnitude_rms:.6e}")
    print(f"   |V|_max = {stats.velocity_magnitude_max:.6e}")

    print(f"\n⚡ Turbulence Quantities:")
    print(f"   TKE = {stats.turbulent_kinetic_energy:.6e}")
    print(f"   ⟨u'v'⟩ = {stats.reynolds_stress:.6e}")
    print(f"   Turbulence intensity = {stats.turbulence_intensity:.6f}")

    print(f"\n📐 Grid Info:")
    print(f"   Grid shape: {stats.grid_shape}")
    print(f"   Total points: {stats.n_points}")
    print(f"   Analysis time: {stats.analysis_time:.4f}s")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
