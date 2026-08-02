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

    def analyze_dalziel_horizontal_spectrum(self,
                                            field: np.ndarray,
                                            x_grid: np.ndarray,
                                            y_grid: np.ndarray,
                                            y0: float,
                                            H: float,
                                            slab_frac: float = 0.1,
                                            bands=((10.0, 25.0), (10.0, 50.0)),
                                            field_name: str = 'F') -> Dict:
        """
        Dalziel, Linden & Youngs (1999) JFM §6.1 horizontal concentration spectrum.

        Faithful reproduction of their method (verified against the paper, pp.29-34):
          * 1-D FFT *along the tank* (x, axis 0) only -- NOT a 2-D radial spectrum.
          * Computed for each horizontal row in the slab
                y0 - slab_frac*H <= y <= y0
            i.e. -slab_frac <= (y - y0)/H <= 0, just below the *initial* interface
            (Dalziel: -0.1 <= z/H <= 0). Power is arithmetically averaged over the
            slab rows.
          * Non-periodic continuation: each row is padded to the next power of two by
            *linear interpolation between its two end concentrations* (Dalziel tested
            window functions and deliberately chose end-ramp padding instead).
          * Wavenumber normalised by k0 = 2*pi/L, L = along-tank domain length, so
            k/k0 is the physical mode number (k/k0 = m * N_orig / N_pad for padded mode m).
          * Weighted (uniform-per-log-interval, w = 1/sqrt(k/k0)) least-squares power-law
            fit P ~ (k/k0)^beta. The concentration spectrum is curved (Dalziel notes this
            explicitly), so the slope is band-dependent; we fit a *fixed* set of bands at
            every grid for an apples-to-apples cross-grid/cross-code comparison. Dalziel
            used 10 <= k/k0 <= 25 for his 160-point simulations and 10 <= k/k0 <= 50 for
            his higher-resolution experimental images; we report both. The 6*dx Nyquist-ish
            limit k_6dx = N_orig/6 is recorded so a band reaching past it (e.g. 10-50 on a
            coarse grid) can be flagged as not fully resolved.

        Args:
            field:     2-D scalar field, indexed [i_x, j_y] (axis 0 = along-tank x).
            x_grid:    2-D x-coordinate grid (same shape as field).
            y_grid:    2-D y-coordinate grid (same shape as field).
            y0:        Initial interface height (fixed; e.g. domain centre 0.25 m).
            H:         Vertical domain extent (e.g. 0.5 m).
            slab_frac: Slab thickness as a fraction of H below y0 (Dalziel: 0.1).
            bands:     Iterable of (k/k0 lower, k/k0 upper) fit bands, fixed across grids.
            field_name: Label.

        Returns:
            Dict with shared k_over_k0, power, k0, L, n_slab_rows, n_pad, n_orig, k_6dx,
            plus one entry per band keyed 'fit_<lo>_<hi>' (e.g. 'fit_10_25'), each a
            sub-dict with power_law_slope (=beta, negative), beta_abs, R^2, fit edges,
            n_fit_points and fully_resolved. Top-level power_law_slope/beta_abs/R^2 mirror
            the first band for convenience.
        """
        N_orig, ny = field.shape
        if N_orig < 8 or ny < 2:
            return self._empty_dalziel_results()

        dx = np.abs(x_grid[1, 0] - x_grid[0, 0])
        L = N_orig * dx
        k0 = 2.0 * np.pi / L
        k_6dx = N_orig / 6.0

        # --- Select slab rows just below the initial interface ---
        y_values = y_grid[0, :]
        slab_mask = (y_values >= y0 - slab_frac * H) & (y_values <= y0)
        slab_idx = np.where(slab_mask)[0]
        if len(slab_idx) == 0:
            # Fall back to the single row nearest y0
            slab_idx = np.array([int(np.argmin(np.abs(y_values - y0)))])

        # --- Next power of two for padding ---
        N_pad = 1 << (int(N_orig - 1).bit_length())
        if N_pad < N_orig:
            N_pad <<= 1
        M = N_pad - N_orig  # number of continuation points

        # --- 1-D FFT per slab row, arithmetic mean of power ---
        n_modes = N_pad // 2 + 1
        power_sum = np.zeros(n_modes)
        for j in slab_idx:
            line = field[:, j].astype(np.float64)
            if M > 0:
                # Linear ramp from the last value back to the first (periodic close)
                ramp = np.linspace(line[-1], line[0], M + 1)[1:]
                padded = np.concatenate([line, ramp])
            else:
                padded = line
            fft_line = np.fft.rfft(padded)
            power_sum += (np.abs(fft_line) ** 2) / N_pad

        power = power_sum / len(slab_idx)

        # --- Physical dimensionless wavenumber k/k0 (drop DC) ---
        m = np.arange(n_modes)
        k_over_k0 = m * (N_orig / N_pad)
        nz = m > 0
        k_over_k0 = k_over_k0[nz]
        power = power[nz]

        # --- Fit each fixed band ---
        results = {
            'field_name': field_name,
            'k_over_k0': k_over_k0,
            'power': power,
            'k0': k0,
            'L': L,
            'n_slab_rows': int(len(slab_idx)),
            'slab_y_range': (float(y0 - slab_frac * H), float(y0)),
            'n_pad': int(N_pad),
            'n_orig': int(N_orig),
            'k_6dx': float(k_6dx),
            'band_keys': [],
        }
        first_fit = None
        for lo, hi in bands:
            fit = self._fit_dalziel_powerlaw(k_over_k0, power, lo, hi)
            fit['fully_resolved'] = bool(hi <= k_6dx)
            key = f'fit_{int(round(lo))}_{int(round(hi))}'
            results[key] = fit
            results['band_keys'].append(key)
            if first_fit is None:
                first_fit = fit
        # Mirror the first band at top level for convenience / backward compat.
        if first_fit is not None:
            results['power_law_slope'] = first_fit['power_law_slope']
            results['beta_abs'] = first_fit['beta_abs']
            results['power_law_r_squared'] = first_fit['power_law_r_squared']
        return results

    def _fit_dalziel_powerlaw(self, k_over_k0: np.ndarray, power: np.ndarray,
                              k_k0_min: float, k_k0_max: float) -> Dict:
        """Weighted (uniform-per-log-interval) LSQ power-law fit over a k/k0 band."""
        band = (k_over_k0 >= k_k0_min) & (k_over_k0 <= k_k0_max) & (power > 0)
        if np.sum(band) < 3:
            return {'power_law_slope': np.nan, 'beta_abs': np.nan,
                    'power_law_r_squared': np.nan,
                    'fit_k_k0_min': k_k0_min, 'fit_k_k0_max': k_k0_max,
                    'n_fit_points': int(np.sum(band))}

        kf = k_over_k0[band]
        pf = power[band]
        log_k = np.log10(kf)
        log_p = np.log10(pf)
        # Weight uniformly per logarithmic interval (else the densely sampled
        # high-k end dominates an ordinary log-log fit). np.polyfit weights act
        # on residuals, so pass sqrt of the per-point weight 1/(k/k0).
        w = 1.0 / np.sqrt(kf)
        slope, intercept = np.polyfit(log_k, log_p, 1, w=w)

        predicted = slope * log_k + intercept
        ss_res = np.sum(w**2 * (log_p - predicted) ** 2)
        ss_tot = np.sum(w**2 * (log_p - np.average(log_p, weights=w**2)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'power_law_slope': float(slope),
            'beta_abs': float(abs(slope)),
            'power_law_intercept': float(intercept),
            'power_law_r_squared': float(r_squared),
            'fit_k_k0_min': float(k_k0_min),
            'fit_k_k0_max': float(k_k0_max),
            'n_fit_points': int(np.sum(band)),
        }

    def _empty_dalziel_results(self) -> Dict:
        """Empty Dalziel-spectrum result."""
        return {
            'power_law_slope': np.nan, 'beta_abs': np.nan,
            'power_law_r_squared': np.nan,
            'k_over_k0': np.array([]), 'power': np.array([]),
            'k0': np.nan, 'L': np.nan, 'n_slab_rows': 0,
            'slab_y_range': (np.nan, np.nan), 'n_pad': 0, 'n_orig': 0,
            'k_6dx': np.nan, 'band_keys': [],
            'field_name': 'unknown',
        }

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
