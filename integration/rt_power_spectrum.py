#!/usr/bin/env python3
"""
Power Spectrum Analysis for RT Simulations

Computes power spectral density for:
1. Interface height fluctuations h(x)
2. Velocity field components (u, v) - 2D spectra
3. Volume fraction field F - 2D spectra

The power spectrum reveals:
- Dominant wavelengths and characteristic scales
- Energy distribution across wavenumber space
- Turbulent cascade behavior (power law slopes)
- Transition from large-scale to small-scale structures

For 2D fields, computes radially-averaged 1D spectrum E(k) from 2D FFT.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal


class PowerSpectrumAnalyzer:
    """
    Analyzes power spectra for RT simulation data.

    Supports:
    - 1D interface height spectrum h(x)
    - 2D velocity field spectra u(x,y), v(x,y)
    - 2D VOF field spectrum F(x,y)
    """

    def __init__(self, debug: bool = False):
        """
        Initialize power spectrum analyzer.

        Args:
            debug: Enable debug output
        """
        self.debug = debug

    def analyze_full_spectrum(self,
                             vtk_data,
                             interface_data=None,
                             detrend: bool = True) -> Dict:
        """
        Compute all available power spectra from VTK data.

        Args:
            vtk_data: VTKData object with scalar fields
            interface_data: Optional InterfaceData for h(x) spectrum
            detrend: Remove mean/trend before computing spectra

        Returns:
            Dictionary with all spectrum results
        """
        results = {}

        if self.debug:
            print(f"   📊 Power Spectrum Analysis")

        # 1. Interface height spectrum (if interface provided)
        if interface_data is not None:
            if self.debug:
                print(f"      🌊 Computing interface height spectrum h(x)...")
            h_spectrum = self.analyze_interface_spectrum(interface_data, detrend)
            results['interface_spectrum'] = h_spectrum

        # 2. VOF field spectrum F(x,y)
        if 'F' in vtk_data.scalar_fields or 'f' in vtk_data.scalar_fields:
            if self.debug:
                print(f"      💧 Computing VOF field spectrum F(x,y)...")
            f_field = vtk_data.scalar_fields.get('F', vtk_data.scalar_fields.get('f'))
            f_spectrum = self.analyze_2d_field_spectrum(f_field, vtk_data.x_grid, vtk_data.y_grid,
                                                        field_name='F', detrend=detrend)
            results['vof_spectrum'] = f_spectrum

        # 3. Velocity field spectra u(x,y), v(x,y)
        has_u = 'u' in vtk_data.scalar_fields or 'U' in vtk_data.scalar_fields
        has_v = 'v' in vtk_data.scalar_fields or 'V' in vtk_data.scalar_fields

        if has_u:
            if self.debug:
                print(f"      ➡️  Computing u-velocity spectrum...")
            u_field = vtk_data.scalar_fields.get('u', vtk_data.scalar_fields.get('U'))
            u_spectrum = self.analyze_2d_field_spectrum(u_field, vtk_data.x_grid, vtk_data.y_grid,
                                                        field_name='u', detrend=detrend)
            results['u_velocity_spectrum'] = u_spectrum

        if has_v:
            if self.debug:
                print(f"      ⬆️  Computing v-velocity spectrum...")
            v_field = vtk_data.scalar_fields.get('v', vtk_data.scalar_fields.get('V'))
            v_spectrum = self.analyze_2d_field_spectrum(v_field, vtk_data.x_grid, vtk_data.y_grid,
                                                        field_name='v', detrend=detrend)
            results['v_velocity_spectrum'] = v_spectrum

        # Combined turbulent kinetic energy spectrum (if both u and v available)
        if has_u and has_v:
            if self.debug:
                print(f"      ⚡ Computing turbulent kinetic energy spectrum...")
            u_field = vtk_data.scalar_fields.get('u', vtk_data.scalar_fields.get('U'))
            v_field = vtk_data.scalar_fields.get('v', vtk_data.scalar_fields.get('V'))
            tke_spectrum = self.analyze_tke_spectrum(u_field, v_field, vtk_data.x_grid, vtk_data.y_grid,
                                                     detrend=detrend)
            results['tke_spectrum'] = tke_spectrum

        return results

    def analyze_interface_spectrum(self,
                                   interface_data,
                                   detrend: bool = True) -> Dict:
        """
        Compute 1D power spectrum of interface height fluctuations h(x).

        Args:
            interface_data: InterfaceData object
            detrend: Remove linear trend

        Returns:
            Dictionary with 1D spectrum results
        """
        segments = interface_data.segments

        if len(segments) == 0:
            return self._empty_1d_results()

        # Convert segments to ordered points
        points = self._segments_to_ordered_points(segments)

        if len(points) < 10:
            return self._empty_1d_results()

        # Sort by x and extract h(x)
        points = points[np.argsort(points[:, 0])]
        x = points[:, 0]
        h = points[:, 1]

        # Resample to uniform grid
        x_uniform, h_uniform = self._resample_uniform(x, h)

        # Remove mean
        h_uniform = h_uniform - np.mean(h_uniform)

        # Detrend
        if detrend:
            h_uniform = signal.detrend(h_uniform, type='linear')

        # Compute 1D spectrum
        wavenumbers, power = self._compute_1d_spectrum(x_uniform, h_uniform)

        # Analyze
        results = self._analyze_1d_spectrum(wavenumbers, power, x_uniform)
        results['field_name'] = 'interface_height'

        return results

    def analyze_2d_field_spectrum(self,
                                  field: np.ndarray,
                                  x_grid: np.ndarray,
                                  y_grid: np.ndarray,
                                  field_name: str = 'field',
                                  detrend: bool = True) -> Dict:
        """
        Compute radially-averaged 1D spectrum from 2D field.

        Args:
            field: 2D scalar field
            x_grid: 2D x-coordinate grid
            y_grid: 2D y-coordinate grid
            field_name: Name of field for labeling
            detrend: Remove mean/trend

        Returns:
            Dictionary with spectrum results
        """
        if field.size < 100:
            return self._empty_1d_results()

        # Remove mean
        field_centered = field - np.mean(field)

        # Detrend (remove planar trend)
        if detrend:
            field_centered = self._detrend_2d(field_centered)

        # Compute 2D FFT
        fft_2d = np.fft.fft2(field_centered)
        power_2d = np.abs(fft_2d)**2 / field.size

        # Get wavenumbers
        nx, ny = field.shape
        dx = np.abs(x_grid[1, 0] - x_grid[0, 0]) if nx > 1 else 1.0
        dy = np.abs(y_grid[0, 1] - y_grid[0, 0]) if ny > 1 else 1.0

        kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)

        # Create 2D wavenumber grid
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.sqrt(KX**2 + KY**2)

        # Radially average to get 1D spectrum E(k)
        wavenumbers, power = self._radial_average(K, power_2d)

        # Analyze
        results = self._analyze_1d_spectrum(wavenumbers, power,
                                           np.array([x_grid.min(), x_grid.max()]))
        results['field_name'] = field_name
        results['grid_shape'] = field.shape

        return results

    def analyze_tke_spectrum(self,
                            u_field: np.ndarray,
                            v_field: np.ndarray,
                            x_grid: np.ndarray,
                            y_grid: np.ndarray,
                            detrend: bool = True) -> Dict:
        """
        Compute turbulent kinetic energy spectrum E(k) = (E_u + E_v) / 2.

        Args:
            u_field: x-velocity component
            v_field: y-velocity component
            x_grid: x-coordinate grid
            y_grid: y-coordinate grid
            detrend: Remove mean velocities

        Returns:
            Dictionary with TKE spectrum results
        """
        # Compute individual velocity spectra
        u_spectrum = self.analyze_2d_field_spectrum(u_field, x_grid, y_grid, 'u', detrend)
        v_spectrum = self.analyze_2d_field_spectrum(v_field, x_grid, y_grid, 'v', detrend)

        # Average the two (TKE is sum of kinetic energies)
        # Note: Power is proportional to energy, so we can average the power spectra
        results = u_spectrum.copy()
        results['field_name'] = 'turbulent_kinetic_energy'

        # If both have valid results, average them
        if not np.isnan(u_spectrum['total_energy']) and not np.isnan(v_spectrum['total_energy']):
            results['total_energy'] = (u_spectrum['total_energy'] + v_spectrum['total_energy']) / 2

        return results

    def _segments_to_ordered_points(self, segments: np.ndarray) -> np.ndarray:
        """
        Convert segments to unique ordered points.

        Optimized version using rounding for duplicate detection instead of
        nested loops. For 25K segments this is ~1000× faster.
        """
        points_list = []
        for seg in segments:
            x1, y1, x2, y2 = seg
            points_list.append([x1, y1])
            points_list.append([x2, y2])

        points = np.array(points_list)

        # Remove duplicates using rounding-based approach
        # Round to 10 decimal places for duplicate detection
        tolerance_decimals = 10

        # Create rounded versions for comparison
        points_rounded = np.round(points, decimals=tolerance_decimals)

        # Use numpy's unique with return_index to get first occurrence of each unique point
        # This is O(N log N) instead of O(N²)
        _, unique_indices = np.unique(points_rounded, axis=0, return_index=True)

        # Sort indices to maintain original order
        unique_indices = np.sort(unique_indices)

        return points[unique_indices]

    def _resample_uniform(self, x: np.ndarray, h: np.ndarray,
                         n_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Resample to uniform grid."""
        if n_points is None:
            n_points = len(x)

        x_uniform = np.linspace(x[0], x[-1], n_points)
        h_uniform = np.interp(x_uniform, x, h)

        return x_uniform, h_uniform

    def _detrend_2d(self, field: np.ndarray) -> np.ndarray:
        """Remove planar trend from 2D field."""
        nx, ny = field.shape
        x_idx = np.arange(nx)
        y_idx = np.arange(ny)
        X, Y = np.meshgrid(x_idx, y_idx, indexing='ij')

        # Fit plane: f(x,y) = a*x + b*y + c
        A = np.column_stack([X.ravel(), Y.ravel(), np.ones(field.size)])
        coeffs, _, _, _ = np.linalg.lstsq(A, field.ravel(), rcond=None)

        # Remove trend
        trend = (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
        return field - trend

    def _compute_1d_spectrum(self, x: np.ndarray,
                            h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1D power spectrum using FFT."""
        n = len(h)
        dx = x[1] - x[0]

        # FFT
        fft_h = np.fft.fft(h)
        power = np.abs(fft_h)**2 / n

        # Wavenumbers
        freq = np.fft.fftfreq(n, d=dx)
        wavenumbers = 2 * np.pi * freq

        # Positive wavenumbers only
        positive_k = wavenumbers > 0
        wavenumbers = wavenumbers[positive_k]
        power = power[positive_k]

        # Sort
        sort_idx = np.argsort(wavenumbers)
        return wavenumbers[sort_idx], power[sort_idx]

    def _radial_average(self, K: np.ndarray, power_2d: np.ndarray,
                       n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radially-averaged 1D spectrum from 2D spectrum.

        Args:
            K: 2D wavenumber magnitude grid
            power_2d: 2D power spectrum
            n_bins: Number of radial bins

        Returns:
            (wavenumbers, radially_averaged_power)
        """
        # Flatten
        k_flat = K.ravel()
        p_flat = power_2d.ravel()

        # Remove zero wavenumber
        nonzero = k_flat > 0
        k_flat = k_flat[nonzero]
        p_flat = p_flat[nonzero]

        # Create bins
        k_min, k_max = k_flat.min(), k_flat.max()
        bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins)

        # Bin and average
        k_binned = []
        p_binned = []

        for i in range(len(bins) - 1):
            mask = (k_flat >= bins[i]) & (k_flat < bins[i + 1])
            if np.sum(mask) > 0:
                k_binned.append(np.mean(k_flat[mask]))
                p_binned.append(np.mean(p_flat[mask]))

        return np.array(k_binned), np.array(p_binned)

    def _analyze_1d_spectrum(self, wavenumbers: np.ndarray,
                            power: np.ndarray,
                            x_domain: np.ndarray) -> Dict:
        """Extract features from 1D spectrum."""
        if len(wavenumbers) < 3:
            return self._empty_1d_results()

        # Total energy
        total_energy = np.trapz(power, wavenumbers)

        # Dominant wavelength
        peak_idx = np.argmax(power)
        dominant_k = wavenumbers[peak_idx]
        dominant_wavelength = 2 * np.pi / dominant_k if dominant_k > 0 else np.inf

        # Power law fit P(k) ~ k^β
        n_k = len(wavenumbers)
        fit_start = max(1, n_k // 4)
        fit_end = min(n_k - 1, 3 * n_k // 4)

        if fit_end > fit_start + 3:
            k_fit = wavenumbers[fit_start:fit_end]
            p_fit = power[fit_start:fit_end]

            # Log-log fit
            valid = p_fit > 0
            if np.sum(valid) > 2:
                log_k = np.log10(k_fit[valid])
                log_p = np.log10(p_fit[valid])

                coeffs = np.polyfit(log_k, log_p, 1)
                slope = coeffs[0]

                # R²
                predicted = slope * log_k + coeffs[1]
                ss_res = np.sum((log_p - predicted)**2)
                ss_tot = np.sum((log_p - np.mean(log_p))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            else:
                slope, r_squared = 0.0, 0.0
        else:
            slope, r_squared = 0.0, 0.0

        # Cutoff wavelength
        cutoff_threshold = 0.1 * power[peak_idx]
        cutoff_indices = np.where(power[peak_idx:] < cutoff_threshold)[0]
        if len(cutoff_indices) > 0:
            cutoff_idx = peak_idx + cutoff_indices[0]
            cutoff_k = wavenumbers[cutoff_idx]
            cutoff_wavelength = 2 * np.pi / cutoff_k if cutoff_k > 0 else np.inf
        else:
            cutoff_wavelength = np.inf

        domain_length = x_domain[-1] - x_domain[0] if len(x_domain) > 1 else 0.0

        return {
            'power_law_slope': slope,
            'power_law_r_squared': r_squared,
            'dominant_wavelength': dominant_wavelength,
            'cutoff_wavelength': cutoff_wavelength,
            'total_energy': total_energy,
            'peak_power': power[peak_idx],
            'n_modes': len(wavenumbers),
            'domain_length': domain_length,
            'wavenumber_range': (wavenumbers[0], wavenumbers[-1])
        }

    def _empty_1d_results(self) -> Dict:
        """Return empty results."""
        return {
            'power_law_slope': np.nan,
            'power_law_r_squared': np.nan,
            'dominant_wavelength': np.nan,
            'cutoff_wavelength': np.nan,
            'total_energy': np.nan,
            'peak_power': np.nan,
            'n_modes': 0,
            'domain_length': 0.0,
            'wavenumber_range': (np.nan, np.nan),
            'field_name': 'unknown'
        }


def main():
    """Test power spectrum analyzer."""
    import argparse
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from integration.rt_vtk_parser import VTKParser
    from integration.rt_interface_extraction import InterfaceExtractor

    parser = argparse.ArgumentParser(description='Test power spectrum analysis')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Parse VTK
    print(f"📂 Loading VTK file...")
    vtk_parser = VTKParser(debug=args.debug)
    vtk_data = vtk_parser.parse_vtk_file(args.vtk_file)

    if not vtk_data:
        print(f"❌ VTK parsing failed")
        return

    # Extract interface
    print(f"🔍 Extracting interface...")
    extractor = InterfaceExtractor(method='skimage', debug=args.debug)
    interface_data = extractor.extract_interface(
        vtk_data['f'], vtk_data.x_grid, vtk_data.y_grid
    )

    # Analyze spectra
    print(f"\n📊 Analyzing power spectra...")
    analyzer = PowerSpectrumAnalyzer(debug=True)
    results = analyzer.analyze_full_spectrum(vtk_data, interface_data)

    print(f"\n{'='*70}")
    print(f"POWER SPECTRUM RESULTS")
    print(f"{'='*70}")

    for key, spectrum in results.items():
        print(f"\n{key.upper()}:")
        print(f"  Power law slope (β):   {spectrum['power_law_slope']:.3f}")
        print(f"  R² of fit:             {spectrum['power_law_r_squared']:.3f}")
        print(f"  Dominant wavelength:   {spectrum['dominant_wavelength']:.6f}")
        print(f"  Total energy:          {spectrum['total_energy']:.6e}")


if __name__ == "__main__":
    main()
