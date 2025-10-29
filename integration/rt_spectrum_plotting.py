#!/usr/bin/env python3
"""
Power Spectrum Plotting for RT Analysis

Provides visualization tools for:
- Single-time power spectra (log-log plots)
- Temporal evolution of spectral features
- Multi-field spectral comparisons
- Power law fits and inertial ranges
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import os


class SpectrumPlotter:
    """
    Creates publication-quality plots of power spectra.
    """

    def __init__(self, output_dir: str = "./", dpi: int = 300):
        """
        Initialize spectrum plotter.

        Args:
            output_dir: Directory for saving plots
            dpi: Plot resolution
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    def plot_single_spectrum(self,
                            wavenumbers: np.ndarray,
                            power: np.ndarray,
                            spectrum_results: Dict,
                            field_name: str = "field",
                            filename: Optional[str] = None) -> str:
        """
        Plot a single power spectrum with power law fit.

        Args:
            wavenumbers: Array of wavenumbers
            power: Power spectral density
            spectrum_results: Dictionary with analysis results
            field_name: Name of field for labeling
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot spectrum
        ax.loglog(wavenumbers, power, 'b-', linewidth=2, alpha=0.7, label='Spectrum')

        # Mark dominant wavelength
        dominant_k = 2 * np.pi / spectrum_results['dominant_wavelength']
        if np.isfinite(dominant_k) and dominant_k > 0:
            ax.axvline(dominant_k, color='red', linestyle='--', linewidth=2,
                      label=f'Dominant λ = {spectrum_results["dominant_wavelength"]:.4f}')

        # Show power law fit region
        slope = spectrum_results['power_law_slope']
        r_squared = spectrum_results['power_law_r_squared']

        if np.isfinite(slope) and np.isfinite(r_squared):
            # Fit line through middle portion
            n_k = len(wavenumbers)
            fit_start = max(1, n_k // 4)
            fit_end = min(n_k - 1, 3 * n_k // 4)

            if fit_end > fit_start:
                k_fit = wavenumbers[fit_start:fit_end]
                # Compute fitted line
                log_k_fit = np.log10(k_fit)
                log_p_fit_center = np.log10(power[fit_start:fit_end])
                intercept = np.mean(log_p_fit_center - slope * log_k_fit)
                p_fit = 10**(slope * log_k_fit + intercept)

                ax.loglog(k_fit, p_fit, 'r--', linewidth=2.5, alpha=0.8,
                         label=f'Power law: k^{slope:.2f} (R² = {r_squared:.3f})')

        # Add theoretical reference slopes
        k_ref = wavenumbers[len(wavenumbers)//2]
        p_ref = power[len(power)//2]

        # -5/3 slope (Kolmogorov turbulence)
        k_53 = np.logspace(np.log10(k_ref*0.5), np.log10(k_ref*2), 10)
        p_53 = p_ref * (k_53/k_ref)**(-5/3)
        ax.loglog(k_53, p_53, 'k:', linewidth=1.5, alpha=0.5, label='k^{-5/3} (Kolmogorov)')

        # -3 slope (steeper cascade)
        p_3 = p_ref * (k_53/k_ref)**(-3)
        ax.loglog(k_53, p_3, 'gray', linestyle=':', linewidth=1.5, alpha=0.5, label='k^{-3}')

        ax.set_xlabel('Wavenumber k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power Spectral Density', fontsize=14, fontweight='bold')
        ax.set_title(f'Power Spectrum: {field_name}', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        # Add text box with statistics
        stats_text = f'Total Energy: {spectrum_results["total_energy"]:.2e}\n'
        stats_text += f'Dominant λ: {spectrum_results["dominant_wavelength"]:.4f}\n'
        stats_text += f'Peak Power: {spectrum_results["peak_power"]:.2e}\n'
        stats_text += f'N modes: {spectrum_results["n_modes"]}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if filename is None:
            filename = f'spectrum_{field_name}.png'

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_temporal_spectrum_evolution(self,
                                        temporal_results: List[Dict],
                                        spectrum_key: str = 'vof_spectrum',
                                        filename: Optional[str] = None) -> str:
        """
        Plot temporal evolution of spectral features.

        Args:
            temporal_results: List of results dictionaries with time series
            spectrum_key: Which spectrum to plot ('vof_spectrum', 'interface_spectrum', etc.)
            filename: Output filename

        Returns:
            Path to saved plot
        """
        # Extract time series
        times = []
        slopes = []
        dominant_wavelengths = []
        total_energies = []
        r_squareds = []

        for result in temporal_results:
            # Check if spectrum is at top level or nested under 'power_spectrum'
            spectrum = None
            if spectrum_key in result and result[spectrum_key]:
                spectrum = result[spectrum_key]
            elif 'power_spectrum' in result and spectrum_key in result['power_spectrum']:
                spectrum = result['power_spectrum'][spectrum_key]

            if spectrum:
                times.append(result.get('time', 0))
                slopes.append(spectrum.get('power_law_slope', np.nan))
                dominant_wavelengths.append(spectrum.get('dominant_wavelength', np.nan))
                total_energies.append(spectrum.get('total_energy', np.nan))
                r_squareds.append(spectrum.get('power_law_r_squared', np.nan))

        if len(times) == 0:
            print(f"⚠️  No temporal data for {spectrum_key}")
            return ""

        times = np.array(times)
        slopes = np.array(slopes)
        dominant_wavelengths = np.array(dominant_wavelengths)
        total_energies = np.array(total_energies)
        r_squareds = np.array(r_squareds)

        # Create multi-panel plot
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: Power law slope evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, slopes, 'b-o', linewidth=2, markersize=6)
        ax1.axhline(-5/3, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='Kolmogorov (-5/3)')
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Power Law Slope β', fontsize=12, fontweight='bold')
        ax1.set_title('Spectral Slope Evolution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 2: Dominant wavelength evolution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, dominant_wavelengths, 'g-s', linewidth=2, markersize=6)
        ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Dominant Wavelength λ', fontsize=12, fontweight='bold')
        ax2.set_title('Dominant Scale Evolution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Total energy evolution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.semilogy(times, total_energies, 'r-^', linewidth=2, markersize=6)
        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Total Spectral Energy', fontsize=12, fontweight='bold')
        ax3.set_title('Energy Evolution', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Panel 4: Fit quality evolution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(times, r_squareds, 'm-d', linewidth=2, markersize=6)
        ax4.axhline(0.95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                   label='R² = 0.95 threshold')
        ax4.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Power Law R²', fontsize=12, fontweight='bold')
        ax4.set_title('Fit Quality Evolution', fontsize=13, fontweight='bold')
        ax4.set_ylim([0, 1.05])
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        field_name = spectrum_key.replace('_spectrum', '').replace('_', ' ').title()
        fig.suptitle(f'Temporal Evolution: {field_name} Spectrum',
                    fontsize=16, fontweight='bold', y=0.995)

        if filename is None:
            filename = f'spectrum_evolution_{spectrum_key}.png'

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_multi_field_comparison(self,
                                    spectrum_results: Dict,
                                    time: float = 0.0,
                                    filename: Optional[str] = None) -> str:
        """
        Compare multiple field spectra at same time.

        Args:
            spectrum_results: Dictionary with multiple spectrum results
            time: Simulation time
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = {
            'interface_spectrum': 'blue',
            'vof_spectrum': 'green',
            'u_velocity_spectrum': 'red',
            'v_velocity_spectrum': 'orange',
            'tke_spectrum': 'purple'
        }

        labels = {
            'interface_spectrum': 'Interface h(x)',
            'vof_spectrum': 'VOF Field F(x,y)',
            'u_velocity_spectrum': 'u-velocity',
            'v_velocity_spectrum': 'v-velocity',
            'tke_spectrum': 'Turbulent KE'
        }

        for key, spectrum in spectrum_results.items():
            if isinstance(spectrum, dict) and 'spectrum' in key and 'wavenumber_range' in spectrum:
                # This is actual spectrum data - would need to store full spectrum
                # For now, just plot the fitted power law
                k_range = spectrum['wavenumber_range']
                if np.isfinite(k_range[0]) and np.isfinite(k_range[1]):
                    k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 50)
                    slope = spectrum['power_law_slope']
                    # Normalize at dominant k
                    k_dom = 2 * np.pi / spectrum['dominant_wavelength']
                    if np.isfinite(slope) and np.isfinite(k_dom):
                        p_dom = spectrum['peak_power']
                        p = p_dom * (k / k_dom)**slope

                        color = colors.get(key, 'black')
                        label = labels.get(key, key)

                        ax.loglog(k, p, linewidth=2.5, alpha=0.8, color=color,
                                 label=f'{label} (β={slope:.2f})')

        ax.set_xlabel('Wavenumber k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power Spectral Density', fontsize=14, fontweight='bold')
        ax.set_title(f'Multi-Field Spectrum Comparison (t = {time:.3f})',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        if filename is None:
            filename = f'spectrum_comparison_t{time:.3f}.png'

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_spectrum_waterfall(self,
                               temporal_results: List[Dict],
                               spectrum_key: str = 'vof_spectrum',
                               n_times: int = 10,
                               filename: Optional[str] = None) -> str:
        """
        Create waterfall plot showing spectrum evolution over time.

        Args:
            temporal_results: Time series of results
            spectrum_key: Which spectrum to plot
            n_times: Number of time snapshots to show
            filename: Output filename

        Returns:
            Path to saved plot
        """
        # Select evenly-spaced times
        n_total = len(temporal_results)
        if n_total <= n_times:
            indices = range(n_total)
        else:
            indices = [int(i * n_total / n_times) for i in range(n_times)]

        fig, ax = plt.subplots(figsize=(12, 8))

        cmap = plt.cm.viridis
        colors = [cmap(i / len(indices)) for i in range(len(indices))]

        for idx, i in enumerate(indices):
            result = temporal_results[i]

            # Check if spectrum is at top level or nested under 'power_spectrum'
            spectrum = None
            if spectrum_key in result and result[spectrum_key]:
                spectrum = result[spectrum_key]
            elif 'power_spectrum' in result and spectrum_key in result['power_spectrum']:
                spectrum = result['power_spectrum'][spectrum_key]

            if not spectrum:
                continue

            time = result.get('time', 0)

            # Reconstruct spectrum from stored info
            k_range = spectrum['wavenumber_range']
            if np.isfinite(k_range[0]) and np.isfinite(k_range[1]):
                k = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 50)
                slope = spectrum['power_law_slope']
                k_dom = 2 * np.pi / spectrum['dominant_wavelength']

                if np.isfinite(slope) and np.isfinite(k_dom):
                    p_dom = spectrum['peak_power']
                    p = p_dom * (k / k_dom)**slope

                    # Offset for waterfall effect
                    offset = idx * 0.5
                    ax.loglog(k, p * 10**offset, linewidth=2, alpha=0.8,
                             color=colors[idx], label=f't = {time:.3f}')

        ax.set_xlabel('Wavenumber k', fontsize=14, fontweight='bold')
        ax.set_ylabel('Power Spectral Density (offset)', fontsize=14, fontweight='bold')

        field_name = spectrum_key.replace('_spectrum', '').replace('_', ' ').title()
        ax.set_title(f'Spectrum Evolution: {field_name}',
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, which='both')

        if filename is None:
            filename = f'spectrum_waterfall_{spectrum_key}.png'

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        return filepath


def main():
    """Test spectrum plotting."""
    import argparse

    parser = argparse.ArgumentParser(description='Test spectrum plotting')
    parser.add_argument('--output-dir', default='./', help='Output directory')

    args = parser.parse_args()

    # Create synthetic test data
    print("Creating synthetic spectrum for testing...")

    k = np.logspace(0, 3, 100)
    # Synthetic spectrum with -5/3 slope
    p = k**(-5/3) * np.exp(-k/500)
    p += 0.1 * np.random.randn(len(k))**2  # Add noise

    spectrum_results = {
        'power_law_slope': -1.67,
        'power_law_r_squared': 0.98,
        'dominant_wavelength': 2 * np.pi / 10,
        'cutoff_wavelength': 2 * np.pi / 100,
        'total_energy': np.trapz(p, k),
        'peak_power': np.max(p),
        'n_modes': len(k),
        'domain_length': 1.0,
        'wavenumber_range': (k[0], k[-1])
    }

    plotter = SpectrumPlotter(output_dir=args.output_dir)

    # Test single spectrum plot
    print("Plotting single spectrum...")
    path = plotter.plot_single_spectrum(k, p, spectrum_results, "Test Field")
    print(f"✅ Saved to: {path}")

    # Test temporal evolution (synthetic)
    print("\nCreating synthetic temporal data...")
    temporal_results = []
    for i, t in enumerate(np.linspace(0, 10, 20)):
        # Evolving spectrum
        slope = -1.5 - 0.2 * (t / 10)
        result = {
            'time': t,
            'vof_spectrum': {
                'power_law_slope': slope,
                'power_law_r_squared': 0.95 + 0.04 * np.random.rand(),
                'dominant_wavelength': 0.5 + 0.1 * t,
                'total_energy': 1e-3 * (1 + t),
                'peak_power': 1e-4 * (1 + 0.5 * t),
                'wavenumber_range': (1.0, 1000.0)
            }
        }
        temporal_results.append(result)

    print("Plotting temporal evolution...")
    path = plotter.plot_temporal_spectrum_evolution(temporal_results, 'vof_spectrum')
    print(f"✅ Saved to: {path}")

    print("Plotting waterfall...")
    path = plotter.plot_spectrum_waterfall(temporal_results, 'vof_spectrum', n_times=10)
    print(f"✅ Saved to: {path}")

    print("\n✅ All test plots created successfully!")


if __name__ == "__main__":
    main()
