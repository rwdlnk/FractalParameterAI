"""
RT Plotting Framework
====================

Publication-ready plotting for Rayleigh-Taylor analysis results.
Reads data from comprehensive CSV files and generates both PNG and EPS plots.

Key Features:
- Evolution plots (single resolution)
- Grid convergence analysis (multi-resolution)
- Multifractal and mixing analysis
- Publication-quality EPS output
- Automatic styling and formatting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import seaborn as sns
from datetime import datetime
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class RTPlotter:
    """
    Publication-ready plotting for RT analysis results.

    Supports both PNG (quick visualization) and EPS (publication) formats.
    """

    def __init__(self, style: str = 'journal', dpi: int = 300, figsize: Tuple[float, float] = (10, 6)):
        """
        Initialize RT plotter with publication settings.

        Args:
            style: Plot style ('journal', 'presentation', 'paper')
            dpi: Resolution for PNG output
            figsize: Default figure size in inches
        """
        self.style = style
        self.dpi = dpi
        self.figsize = figsize
        self.setup_publication_style()

    def setup_publication_style(self):
        """Configure matplotlib for publication-quality plots."""
        # Use a clean, publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Publication parameters
        params = {
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'grid.alpha': 0.3,
            'axes.grid': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        }

        if self.style == 'journal':
            params.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'Times', 'serif'],
                'mathtext.fontset': 'stix',
                'axes.linewidth': 1.2,
                'xtick.direction': 'in',
                'ytick.direction': 'in',
                'xtick.top': True,
                'ytick.right': True
            })
        elif self.style == 'presentation':
            params.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'lines.linewidth': 3,
                'lines.markersize': 8
            })

        plt.rcParams.update(params)

    def save_plot(self, output_dir: Path, filename: str, formats: List[str] = ['png', 'eps']):
        """Save plot in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            filepath = output_dir / f"{filename}.{fmt}"
            if fmt == 'eps':
                plt.savefig(filepath, format='eps', bbox_inches='tight', pad_inches=0.1)
            else:
                plt.savefig(filepath, format=fmt, dpi=self.dpi, bbox_inches='tight', pad_inches=0.1)

        print(f"   📊 Saved {filename} in {formats}")

    def plot_dimension_evolution(self, csv_file: Path, output_dir: Path,
                                formats: List[str] = ['png', 'eps']) -> bool:
        """
        Plot fractal dimension evolution over time.

        Args:
            csv_file: Path to comprehensive CSV results
            output_dir: Output directory for plots
            formats: Output formats ['png', 'eps']

        Returns:
            True if successful
        """
        try:
            df = pd.read_csv(csv_file)

            # Convert time to seconds
            time_sec = df['time'] / 1000.0

            plt.figure(figsize=self.figsize)

            # Main dimension plot
            plt.plot(time_sec, df['fractal_dimension'], 'b-o',
                    linewidth=2, markersize=4, alpha=0.8, label='Fractal Dimension')

            # Add R² quality indicator as alpha
            if 'fractal_r_squared' in df.columns:
                r2_mask = df['fractal_r_squared'] >= 0.99
                plt.scatter(time_sec[r2_mask], df['fractal_dimension'][r2_mask],
                          c='darkblue', s=30, alpha=0.8, zorder=5, label='R² ≥ 0.99')

            plt.xlabel('Time (s)')
            plt.ylabel('Fractal Dimension')
            plt.title('Fractal Dimension Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Dynamic Y-axis based on data range
            y_min, y_max = df['fractal_dimension'].min(), df['fractal_dimension'].max()
            y_range = y_max - y_min
            y_margin = max(0.05, y_range * 0.1)
            plt.ylim(y_min - y_margin, y_max + y_margin)

            self.save_plot(output_dir, 'dimension_evolution', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error plotting dimension evolution: {e}")
            return False

    def plot_mixing_evolution(self, csv_file: Path, output_dir: Path,
                             formats: List[str] = ['png', 'eps']) -> bool:
        """Plot mixing width and thickness evolution."""
        try:
            df = pd.read_csv(csv_file)
            time_sec = df['time'] / 1000.0

            plt.figure(figsize=self.figsize)

            # Plot mixing width and thickness
            if 'mixing_width' in df.columns:
                valid_mixing = ~pd.isna(df['mixing_width'])
                plt.plot(time_sec[valid_mixing], df['mixing_width'][valid_mixing],
                        'g-o', linewidth=2, markersize=4, label='Mixing Width')

            if 'thickness' in df.columns:
                valid_thickness = ~pd.isna(df['thickness'])
                plt.plot(time_sec[valid_thickness], df['thickness'][valid_thickness],
                        'r-s', linewidth=2, markersize=4, label='Interface Thickness')

            if 'dalziel_h' in df.columns:
                valid_dalziel = ~pd.isna(df['dalziel_h'])
                plt.plot(time_sec[valid_dalziel], df['dalziel_h'][valid_dalziel],
                        'm-^', linewidth=2, markersize=4, label='Dalziel h')

            plt.xlabel('Time (s)')
            plt.ylabel('Length Scale')
            plt.title('Mixing Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            self.save_plot(output_dir, 'mixing_evolution', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error plotting mixing evolution: {e}")
            return False

    def plot_rms_evolution(self, csv_file: Path, output_dir: Path,
                          formats: List[str] = ['png', 'eps']) -> bool:
        """Plot RMS velocity evolution."""
        try:
            df = pd.read_csv(csv_file)
            time_sec = df['time'] / 1000.0

            plt.figure(figsize=self.figsize)

            # Plot RMS velocities
            if 'u_rms' in df.columns:
                valid_u = ~pd.isna(df['u_rms'])
                plt.plot(time_sec[valid_u], df['u_rms'][valid_u],
                        'b-o', linewidth=2, markersize=4, label='u_rms')

            if 'v_rms' in df.columns:
                valid_v = ~pd.isna(df['v_rms'])
                plt.plot(time_sec[valid_v], df['v_rms'][valid_v],
                        'r-s', linewidth=2, markersize=4, label='v_rms')

            if 'total_rms' in df.columns:
                valid_total = ~pd.isna(df['total_rms'])
                plt.plot(time_sec[valid_total], df['total_rms'][valid_total],
                        'g-^', linewidth=2, markersize=4, label='total_rms')

            plt.xlabel('Time (s)')
            plt.ylabel('RMS Velocity')
            plt.title('RMS Velocity Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            self.save_plot(output_dir, 'rms_evolution', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error plotting RMS evolution: {e}")
            return False

    def plot_multifractal_evolution(self, csv_file: Path, output_dir: Path,
                                   formats: List[str] = ['png', 'eps']) -> bool:
        """Plot multifractal dimension evolution."""
        try:
            df = pd.read_csv(csv_file)
            time_sec = df['time'] / 1000.0

            # Check if multifractal data exists
            mf_columns = ['D0_capacity', 'D1_information', 'D2_correlation']
            if not any(col in df.columns for col in mf_columns):
                print("   ⚠️  No multifractal data found in CSV")
                return False

            plt.figure(figsize=self.figsize)

            # Plot generalized dimensions
            if 'D0_capacity' in df.columns:
                valid_d0 = ~pd.isna(df['D0_capacity'])
                plt.plot(time_sec[valid_d0], df['D0_capacity'][valid_d0],
                        'b-o', linewidth=2, markersize=4, label='D₀ (capacity)')

            if 'D1_information' in df.columns:
                valid_d1 = ~pd.isna(df['D1_information'])
                plt.plot(time_sec[valid_d1], df['D1_information'][valid_d1],
                        'r-s', linewidth=2, markersize=4, label='D₁ (information)')

            if 'D2_correlation' in df.columns:
                valid_d2 = ~pd.isna(df['D2_correlation'])
                plt.plot(time_sec[valid_d2], df['D2_correlation'][valid_d2],
                        'g-^', linewidth=2, markersize=4, label='D₂ (correlation)')

            plt.xlabel('Time (s)')
            plt.ylabel('Generalized Dimension')
            plt.title('Multifractal Dimension Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            self.save_plot(output_dir, 'multifractal_evolution', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error plotting multifractal evolution: {e}")
            return False

    def plot_grid_convergence(self, csv_files: List[Path], resolutions: List[str],
                             output_dir: Path, formats: List[str] = ['png', 'eps']) -> bool:
        """
        Plot grid convergence analysis across multiple resolutions.

        Args:
            csv_files: List of CSV files for different resolutions
            resolutions: List of resolution labels (e.g., ['320x400', '640x800'])
            output_dir: Output directory
            formats: Output formats
        """
        try:
            plt.figure(figsize=self.figsize)

            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

            for i, (csv_file, resolution) in enumerate(zip(csv_files, resolutions)):
                if not csv_file.exists():
                    print(f"   ⚠️  CSV file not found: {csv_file}")
                    continue

                df = pd.read_csv(csv_file)
                time_sec = df['time'] / 1000.0
                color = colors[i % len(colors)]

                plt.plot(time_sec, df['fractal_dimension'],
                        color=color, linewidth=2, marker='o', markersize=3,
                        alpha=0.8, label=f'{resolution}')

            plt.xlabel('Time (s)')
            plt.ylabel('Fractal Dimension')
            plt.title('Grid Convergence Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)

            self.save_plot(output_dir, 'grid_convergence', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error plotting grid convergence: {e}")
            return False

    def plot_comprehensive_summary(self, csv_file: Path, output_dir: Path,
                                  formats: List[str] = ['png', 'eps']) -> bool:
        """Create comprehensive multi-panel summary plot."""
        try:
            df = pd.read_csv(csv_file)
            time_sec = df['time'] / 1000.0

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('RT Analysis Summary', fontsize=16)

            # Panel 1: Fractal dimension
            axes[0,0].plot(time_sec, df['fractal_dimension'], 'b-o', linewidth=2, markersize=3)
            axes[0,0].set_xlabel('Time (s)')
            axes[0,0].set_ylabel('Fractal Dimension')
            axes[0,0].set_title('Fractal Evolution')
            axes[0,0].grid(True, alpha=0.3)

            # Panel 2: Mixing analysis
            if 'mixing_width' in df.columns:
                valid_mixing = ~pd.isna(df['mixing_width'])
                axes[0,1].plot(time_sec[valid_mixing], df['mixing_width'][valid_mixing],
                              'g-o', linewidth=2, markersize=3, label='Mixing Width')
            if 'thickness' in df.columns:
                valid_thickness = ~pd.isna(df['thickness'])
                axes[0,1].plot(time_sec[valid_thickness], df['thickness'][valid_thickness],
                              'r-s', linewidth=2, markersize=3, label='Thickness')
            axes[0,1].set_xlabel('Time (s)')
            axes[0,1].set_ylabel('Length Scale')
            axes[0,1].set_title('Mixing Evolution')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Panel 3: RMS velocities
            if 'total_rms' in df.columns:
                valid_rms = ~pd.isna(df['total_rms'])
                axes[1,0].plot(time_sec[valid_rms], df['total_rms'][valid_rms],
                              'm-^', linewidth=2, markersize=3, label='Total RMS')
            if 'u_rms' in df.columns:
                valid_u = ~pd.isna(df['u_rms'])
                axes[1,0].plot(time_sec[valid_u], df['u_rms'][valid_u],
                              'b-o', linewidth=2, markersize=3, label='u_rms')
            axes[1,0].set_xlabel('Time (s)')
            axes[1,0].set_ylabel('RMS Velocity')
            axes[1,0].set_title('Velocity Evolution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)

            # Panel 4: Quality metrics
            if 'fractal_r_squared' in df.columns:
                axes[1,1].plot(time_sec, df['fractal_r_squared'], 'k-o', linewidth=2, markersize=3)
                axes[1,1].axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='R² = 0.99')
            axes[1,1].set_xlabel('Time (s)')
            axes[1,1].set_ylabel('R² Quality')
            axes[1,1].set_title('Analysis Quality')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            self.save_plot(output_dir, 'comprehensive_summary', formats)
            plt.close()
            return True

        except Exception as e:
            print(f"   ❌ Error creating comprehensive summary: {e}")
            return False

    def generate_all_plots(self, csv_file: Path, output_dir: Path,
                          formats: List[str] = ['png', 'eps']) -> Dict[str, bool]:
        """Generate all available plots from CSV data."""
        results = {}

        print(f"📊 Generating RT plots from {csv_file.name}")
        print(f"   Output directory: {output_dir}")
        print(f"   Formats: {formats}")

        # Individual plot types
        results['dimension_evolution'] = self.plot_dimension_evolution(csv_file, output_dir, formats)
        results['mixing_evolution'] = self.plot_mixing_evolution(csv_file, output_dir, formats)
        results['rms_evolution'] = self.plot_rms_evolution(csv_file, output_dir, formats)
        results['multifractal_evolution'] = self.plot_multifractal_evolution(csv_file, output_dir, formats)
        results['comprehensive_summary'] = self.plot_comprehensive_summary(csv_file, output_dir, formats)

        # Summary
        successful = sum(results.values())
        total = len(results)
        print(f"   ✅ Generated {successful}/{total} plot types successfully")

        return results