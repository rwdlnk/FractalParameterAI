#!/usr/bin/env python3
"""
RT PLOTTER: Standalone Plotting Tool for RT Analysis Results
============================================================

Reads CSV files from rt_analyzer.py or enhanced_analyzer.py and creates
publication-ready plots with customizable styling and format options.

Features:
- Auto-detects analysis type and available data
- Multiple plot styles (journal, presentation, manuscript)
- Multi-file comparison support
- High-resolution output for publications
- Interactive plotting mode
- Selective plot generation

Author: RT Analysis Framework
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PLOTTING STYLES AND CONFIGURATION
# ============================================================================

class PlotStyles:
    """Predefined plotting styles for different use cases."""
    
    @staticmethod
    def apply_journal_style():
        """Clean, professional style for journal submissions."""
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.linewidth': 1.0,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'grid.alpha': 0.3,
            'lines.linewidth': 1.5,
            'lines.markersize': 6,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    @staticmethod
    def apply_presentation_style():
        """Bold, high-contrast style for presentations."""
        plt.rcParams.update({
            'font.size': 14,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'axes.linewidth': 2.0,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'legend.frameon': True,
            'legend.framealpha': 0.9,
            'grid.alpha': 0.4,
            'lines.linewidth': 3.0,
            'lines.markersize': 8,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.2
        })
    
    @staticmethod
    def apply_manuscript_style():
        """Black and white style for manuscript drafts."""
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times', 'DejaVu Serif', 'Liberation Serif'],
            'axes.linewidth': 1.0,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.framealpha': 1.0,
            'grid.alpha': 0.5,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'figure.dpi': 100,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

# ============================================================================
# DATA ANALYSIS AND DETECTION
# ============================================================================

class CSVAnalyzer:
    """Analyzes CSV files to determine analysis type and available data."""
    
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.analysis_info = self._analyze_data()
    
    def _analyze_data(self):
        """Analyze the CSV data to determine analysis type and capabilities."""
        info = {
            'file': self.csv_file,
            'rows': len(self.df),
            'analysis_mode': self._detect_analysis_mode(),
            'has_fractal': self._has_fractal_data(),
            'has_mixing': self._has_mixing_data(),
            'has_physics': self._has_physics_data(),
            'has_multifractal': self._has_multifractal_data(),
            'mixing_methods': self._detect_mixing_methods(),
            'resolutions': self._get_resolutions(),
            'time_range': self._get_time_range(),
            'physics_params': self._get_physics_params()
        }
        return info
    
    def _detect_analysis_mode(self):
        """Detect the analysis mode from the data."""
        if 'analysis_mode' in self.df.columns:
            return self.df['analysis_mode'].iloc[0]
        
        # Infer from data structure
        unique_resolutions = len(self.df['resolution_str'].unique()) if 'resolution_str' in self.df.columns else 1
        unique_times = len(self.df['actual_time'].unique()) if 'actual_time' in self.df.columns else 1
        
        if unique_resolutions == 1 and unique_times > 1:
            return 'temporal_evolution'
        elif unique_resolutions > 1 and unique_times == 1:
            return 'convergence_study'
        elif unique_resolutions > 1 and unique_times > 1:
            return 'multi_time_convergence'
        else:
            return 'single_point'
    
    def _has_fractal_data(self):
        """Check if fractal dimension data is available."""
        fractal_cols = ['fractal_dim', 'fractal_dimension', 'fd_error', 'fd_r_squared']
        return any(col in self.df.columns for col in fractal_cols)
    
    def _has_mixing_data(self):
        """Check if mixing data is available."""
        mixing_cols = ['h_total', 'ht', 'hb', 'h_01', 'h_10', 'h_penetration_up', 'h_penetration_down', 'W', 'mixing_ratio']
        return any(col in self.df.columns for col in mixing_cols)
    
    def _has_physics_data(self):
        """Check if dimensionless physics data is available."""
        physics_cols = ['tau', 'ht_normalized', 'hb_normalized', 'h_total_normalized']
        return any(col in self.df.columns for col in physics_cols)
    
    def _has_multifractal_data(self):
        """Check if multifractal data is available."""
        mf_cols = ['mf_D0', 'mf_D1', 'mf_D2', 'mf_alpha_width', 'mf_degree_multifractality']
        return any(col in self.df.columns for col in mf_cols)
    
    def _detect_mixing_methods(self):
        """Detect which mixing analysis methods are available."""
        methods = []
        
        # Check for four-method framework
        if 'h_01' in self.df.columns and 'h_10' in self.df.columns:
            methods.append('mass_conservation')
        if 'h_penetration_up' in self.df.columns and 'h_penetration_down' in self.df.columns:
            methods.append('geometric')
        if 'W' in self.df.columns:
            methods.append('youngs')
        if 'mixing_ratio' in self.df.columns:
            methods.append('diagnostic')
        
        # Check for legacy methods
        if 'ht' in self.df.columns and 'hb' in self.df.columns:
            if not methods:  # Only if no modern methods detected
                methods.append('legacy')
        
        # Check for comprehensive
        if len(methods) >= 3:
            methods.append('comprehensive')
        
        return methods
    
    def _get_resolutions(self):
        """Get list of resolutions in the data."""
        if 'resolution_str' in self.df.columns:
            return sorted(self.df['resolution_str'].unique().tolist())
        elif 'grid_resolution' in self.df.columns:
            return sorted(self.df['grid_resolution'].unique().tolist())
        else:
            return ['unknown']
    
    def _get_time_range(self):
        """Get time range of the data."""
        if 'actual_time' in self.df.columns:
            times = self.df['actual_time']
            return (times.min(), times.max())
        elif 'time' in self.df.columns:
            times = self.df['time']
            return (times.min(), times.max())
        else:
            return (0, 0)
    
    def _get_physics_params(self):
        """Extract physics parameters if available."""
        params = {}
        
        # Try to infer from dimensionless data
        if 'tau' in self.df.columns and 'actual_time' in self.df.columns:
            # Calculate tau_factor = tau / actual_time
            valid_data = self.df[(self.df['tau'] > 0) & (self.df['actual_time'] > 0)]
            if len(valid_data) > 0:
                tau_factor = (valid_data['tau'] / valid_data['actual_time']).mean()
                params['tau_factor'] = tau_factor
        
        return params
    
    def print_summary(self):
        """Print a summary of the data analysis."""
        info = self.analysis_info
        
        print(f"\n📊 CSV ANALYSIS SUMMARY")
        print(f"=" * 50)
        print(f"File: {os.path.basename(info['file'])}")
        print(f"Rows: {info['rows']}")
        print(f"Analysis mode: {info['analysis_mode']}")
        print(f"Resolutions: {info['resolutions']}")
        print(f"Time range: {info['time_range'][0]:.1f} → {info['time_range'][1]:.1f}")
        
        print(f"\n📈 AVAILABLE DATA:")
        print(f"  Fractal dimension: {'✅' if info['has_fractal'] else '❌'}")
        print(f"  Mixing analysis: {'✅' if info['has_mixing'] else '❌'}")
        print(f"  Physics (dimensionless): {'✅' if info['has_physics'] else '❌'}")
        print(f"  Multifractal: {'✅' if info['has_multifractal'] else '❌'}")
        
        if info['mixing_methods']:
            print(f"  Mixing methods: {', '.join(info['mixing_methods'])}")
        
        if info['physics_params']:
            print(f"\n🧮 PHYSICS PARAMETERS:")
            for param, value in info['physics_params'].items():
                print(f"  {param}: {value:.4f}")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

class RTPlotter:
    """Main plotting class for RT analysis results."""
    
    def __init__(self, csv_files, output_dir="./plots", style="journal", 
                 format="png", dpi=300, no_titles=False, interactive=False):
        self.csv_files = csv_files if isinstance(csv_files, list) else [csv_files]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.style = style
        self.format = format
        self.dpi = dpi
        self.no_titles = no_titles
        self.interactive = interactive
        
        # Apply style
        self._apply_style()
        
        # Analyze CSV files
        self.analyzers = [CSVAnalyzer(csv_file) for csv_file in self.csv_files]
        self.primary_analyzer = self.analyzers[0]  # Use first file as primary
        
        # Set up colors for multi-file plots
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.csv_files)))
        
    def _apply_style(self):
        """Apply the selected plotting style."""
        if self.style == "journal":
            PlotStyles.apply_journal_style()
        elif self.style == "presentation":
            PlotStyles.apply_presentation_style()
        elif self.style == "manuscript":
            PlotStyles.apply_manuscript_style()
        
        # Override DPI if specified
        plt.rcParams['savefig.dpi'] = self.dpi
        
        # Set interactive mode
        if self.interactive:
            plt.ion()
        else:
            plt.ioff()
    
    def _save_plot(self, filename, **kwargs):
        """Save plot with consistent settings."""
        full_path = self.output_dir / f"{filename}.{self.format}"
        plt.savefig(full_path, format=self.format, dpi=self.dpi, **kwargs)
        print(f"   ✅ Saved: {full_path}")
        
        if not self.interactive:
            plt.close()
    
    def _get_title(self, base_title):
        """Get title based on no_titles setting."""
        return "" if self.no_titles else base_title
    
    def plot_fractal_evolution(self):
        """Plot fractal dimension evolution over time."""
        if not self.primary_analyzer.analysis_info['has_fractal']:
            print("⚠️  No fractal data available")
            return
        
        print("📐 Creating fractal dimension evolution plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.df
            if 'actual_time' not in df.columns or 'fractal_dim' not in df.columns:
                continue
            
            # Filter successful results
            successful_df = df[df.get('status', 'success') == 'success'].copy()
            if len(successful_df) == 0:
                continue
            
            # Sort by time
            successful_df = successful_df.sort_values('actual_time')
            
            # Plot by resolution if multiple resolutions
            if analyzer.analysis_info['analysis_mode'] in ['multi_time_convergence', 'matrix_analysis']:
                resolutions = analyzer.analysis_info['resolutions']
                resolution_colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions)))
                
                for j, resolution in enumerate(resolutions):
                    res_data = successful_df[successful_df['resolution_str'] == resolution]
                    if len(res_data) > 0:
                        label = f"{resolution}" if len(self.csv_files) == 1 else f"{os.path.basename(analyzer.csv_file)} - {resolution}"
                        
                        if 'fd_error' in res_data.columns:
                            ax.errorbar(res_data['actual_time'], res_data['fractal_dim'],
                                       yerr=res_data['fd_error'], fmt='o-', capsize=3,
                                       color=resolution_colors[j], linewidth=2, markersize=4,
                                       label=label, alpha=0.8)
                        else:
                            ax.plot(res_data['actual_time'], res_data['fractal_dim'], 'o-',
                                   color=resolution_colors[j], linewidth=2, markersize=4,
                                   label=label, alpha=0.8)
            else:
                # Single resolution temporal evolution
                label = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else 'Fractal Dimension'
                
                if 'fd_error' in successful_df.columns:
                    ax.errorbar(successful_df['actual_time'], successful_df['fractal_dim'],
                               yerr=successful_df['fd_error'], fmt='o-', capsize=3,
                               color=self.colors[i], linewidth=2, markersize=5,
                               label=label, alpha=0.8)
                else:
                    ax.plot(successful_df['actual_time'], successful_df['fractal_dim'], 'o-',
                           color=self.colors[i], linewidth=2, markersize=5,
                           label=label, alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Fractal Dimension')
        ax.set_title(self._get_title('Fractal Dimension Evolution'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add reference lines
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Smooth line (D=1)')
        ax.axhline(y=2.0, color='red', linestyle=':', alpha=0.5, label='Space-filling (D=2)')
        
        plt.tight_layout()
        self._save_plot('fractal_evolution')
    
    def plot_fractal_convergence(self):
        """Plot fractal dimension convergence with resolution."""
        if not self.primary_analyzer.analysis_info['has_fractal']:
            print("⚠️  No fractal data available")
            return
        
        # Only plot if we have multiple resolutions
        if len(self.primary_analyzer.analysis_info['resolutions']) < 2:
            print("⚠️  Convergence plot requires multiple resolutions")
            return
        
        print("📈 Creating fractal dimension convergence plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.df
            successful_df = df[df.get('status', 'success') == 'success'].copy()
            
            if len(successful_df) == 0 or 'effective_resolution' not in successful_df.columns:
                continue
            
            # Sort by resolution
            successful_df = successful_df.sort_values('effective_resolution')
            
            label = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else 'Fractal Dimension'
            
            if 'fd_error' in successful_df.columns:
                ax.errorbar(successful_df['effective_resolution'], successful_df['fractal_dim'],
                           yerr=successful_df['fd_error'], fmt='o-', capsize=5,
                           color=self.colors[i], linewidth=2, markersize=6,
                           label=label, alpha=0.8)
            else:
                ax.plot(successful_df['effective_resolution'], successful_df['fractal_dim'], 'o-',
                       color=self.colors[i], linewidth=2, markersize=6,
                       label=label, alpha=0.8)
        
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Effective Resolution')
        ax.set_ylabel('Fractal Dimension')
        ax.set_title(self._get_title('Fractal Dimension Convergence'))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        self._save_plot('fractal_convergence')
    
    def plot_mixing_evolution(self):
        """Plot mixing thickness evolution over time."""
        if not self.primary_analyzer.analysis_info['has_mixing']:
            print("⚠️  No mixing data available")
            return
        
        print("🧪 Creating mixing evolution plot...")
        
        # Determine subplot layout based on available methods
        mixing_methods = self.primary_analyzer.analysis_info['mixing_methods']
        
        if 'comprehensive' in mixing_methods or len(mixing_methods) >= 3:
            # Four-method comparison
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            axes = [ax1, ax2, ax3, ax4]
            titles = ['Mass-Conservation Method', 'Geometric Method', 'Youngs Method', 'Method Comparison']
        else:
            # Single or dual method
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            if not isinstance(axes, list):
                axes = [axes]
            titles = ['Upper Thickness', 'Lower Thickness']
        
        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.df
            successful_df = df[df.get('status', 'success') == 'success'].copy()
            
            if len(successful_df) == 0:
                continue
            
            successful_df = successful_df.sort_values('actual_time')
            
            # Plot based on available methods
            if 'comprehensive' in mixing_methods or len(mixing_methods) >= 3:
                self._plot_comprehensive_mixing(successful_df, axes, i, analyzer)
            else:
                self._plot_simple_mixing(successful_df, axes, i, analyzer)
        
        # Set titles and formatting
        for ax, title in zip(axes, titles):
            ax.set_xlabel('Time')
            ax.set_ylabel('Mixing Thickness')
            ax.set_title(self._get_title(title))
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        self._save_plot('mixing_evolution')
    
    def _plot_comprehensive_mixing(self, df, axes, file_idx, analyzer):
        """Plot comprehensive mixing analysis with four methods."""
        ax1, ax2, ax3, ax4 = axes
        color = self.colors[file_idx]
        label_prefix = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else ""
        
        # Mass-conservation method
        if 'h_01' in df.columns and 'h_10' in df.columns:
            ax1.plot(df['actual_time'], df['h_01'], 'o-', color=color, 
                    linewidth=2, markersize=4, label=f'{label_prefix} h₀₁', alpha=0.8)
            ax1.plot(df['actual_time'], df['h_10'], 's--', color=color, 
                    linewidth=2, markersize=3, label=f'{label_prefix} h₁₀', alpha=0.6)
        
        # Geometric method
        if 'h_penetration_up' in df.columns and 'h_penetration_down' in df.columns:
            ax2.plot(df['actual_time'], df['h_penetration_up'], 'o-', color=color,
                    linewidth=2, markersize=4, label=f'{label_prefix} h_up', alpha=0.8)
            ax2.plot(df['actual_time'], df['h_penetration_down'], 's--', color=color,
                    linewidth=2, markersize=3, label=f'{label_prefix} h_down', alpha=0.6)
        
        # Youngs method
        if 'W' in df.columns:
            ax3.plot(df['actual_time'], df['W'], 'o-', color=color,
                    linewidth=2, markersize=4, label=f'{label_prefix} W', alpha=0.8)
        
        # Method comparison
        if 'h_total_mass' in df.columns:
            ax4.plot(df['actual_time'], df['h_total_mass'], 'o-', color=color,
                    linewidth=2, markersize=4, label=f'{label_prefix} Mass', alpha=0.8)
        if 'h_total_geometric' in df.columns:
            ax4.plot(df['actual_time'], df['h_total_geometric'], 's--', color=color,
                    linewidth=2, markersize=3, label=f'{label_prefix} Geometric', alpha=0.6)
        if 'W' in df.columns:
            ax4.plot(df['actual_time'], df['W'], '^:', color=color,
                    linewidth=2, markersize=3, label=f'{label_prefix} Youngs', alpha=0.6)
    
    def _plot_simple_mixing(self, df, axes, file_idx, analyzer):
        """Plot simple mixing analysis with legacy fields."""
        color = self.colors[file_idx]
        label_prefix = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else ""
        
        if len(axes) >= 2:
            # Upper and lower thickness
            if 'ht' in df.columns:
                axes[0].plot(df['actual_time'], df['ht'], 'o-', color=color,
                           linewidth=2, markersize=4, label=f'{label_prefix} Upper', alpha=0.8)
            if 'hb' in df.columns:
                axes[1].plot(df['actual_time'], df['hb'], 'o-', color=color,
                           linewidth=2, markersize=4, label=f'{label_prefix} Lower', alpha=0.8)
        else:
            # Total thickness only
            if 'h_total' in df.columns:
                axes[0].plot(df['actual_time'], df['h_total'], 'o-', color=color,
                           linewidth=2, markersize=4, label=f'{label_prefix} Total', alpha=0.8)
    
    def plot_dimensionless_physics(self):
        """Plot dimensionless physics (h/H vs τ)."""
        if not self.primary_analyzer.analysis_info['has_physics']:
            print("⚠️  No dimensionless physics data available")
            return
        
        print("🧮 Creating dimensionless physics plot...")
        
        # Determine layout based on available methods
        mixing_methods = self.primary_analyzer.analysis_info['mixing_methods']
        
        if 'comprehensive' in mixing_methods or len(mixing_methods) >= 3:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.df
            successful_df = df[df.get('status', 'success') == 'success'].copy()
            
            if len(successful_df) == 0 or 'tau' not in successful_df.columns:
                continue
            
            successful_df = successful_df.sort_values('tau')
            self._plot_dimensionless_data(successful_df, fig.axes, i, analyzer)
        
        # Set labels and titles
        if len(fig.axes) == 4:
            titles = ['Mass-Conservation: h₀₁/H, h₁₀/H vs τ', 'Geometric: h_up/H, h_down/H vs τ',
                     'Youngs: W/H vs τ', 'Method Comparison vs τ']
        else:
            titles = ['Upper Thickness: h_t/H vs τ', 'Lower Thickness: h_b/H vs τ']
        
        for ax, title in zip(fig.axes, titles):
            ax.set_xlabel('Dimensionless Time τ = √(Ag/H)t')
            ax.set_ylabel('Normalized Thickness')
            ax.set_title(self._get_title(title))
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        self._save_plot('dimensionless_physics')
    
    def _plot_dimensionless_data(self, df, axes, file_idx, analyzer):
        """Plot dimensionless data on the provided axes."""
        color = self.colors[file_idx]
        label_prefix = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else ""
        
        if len(axes) == 4:
            # Four-method layout
            ax1, ax2, ax3, ax4 = axes
            
            # Mass-conservation
            if 'h_01_normalized' in df.columns:
                ax1.plot(df['tau'], df['h_01_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} h₀₁/H', alpha=0.8)
            if 'h_10_normalized' in df.columns:
                ax1.plot(df['tau'], df['h_10_normalized'], 's--', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} h₁₀/H', alpha=0.6)
            
            # Geometric
            if 'h_penetration_up_normalized' in df.columns:
                ax2.plot(df['tau'], df['h_penetration_up_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} h_up/H', alpha=0.8)
            if 'h_penetration_down_normalized' in df.columns:
                ax2.plot(df['tau'], df['h_penetration_down_normalized'], 's--', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} h_down/H', alpha=0.6)
            
            # Youngs
            if 'W_normalized' in df.columns:
                ax3.plot(df['tau'], df['W_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} W/H', alpha=0.8)
            
            # Comparison
            if 'h_total_mass_normalized' in df.columns:
                ax4.plot(df['tau'], df['h_total_mass_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} Mass', alpha=0.8)
            if 'h_total_geometric_normalized' in df.columns:
                ax4.plot(df['tau'], df['h_total_geometric_normalized'], 's--', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} Geometric', alpha=0.6)
            if 'W_normalized' in df.columns:
                ax4.plot(df['tau'], df['W_normalized'], '^:', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} Youngs', alpha=0.6)
        
        else:
            # Simple two-axis layout
            ax1, ax2 = axes
            
            if 'ht_normalized' in df.columns:
                ax1.plot(df['tau'], df['ht_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} h_t/H', alpha=0.8)
            if 'hb_normalized' in df.columns:
                ax2.plot(df['tau'], df['hb_normalized'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} h_b/H', alpha=0.8)
    
    def plot_multifractal(self):
        """Plot multifractal analysis results."""
        if not self.primary_analyzer.analysis_info['has_multifractal']:
            print("⚠️  No multifractal data available")
            return
        
        print("🔬 Creating multifractal analysis plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, analyzer in enumerate(self.analyzers):
            df = analyzer.df
            mf_df = df[(df.get('mf_status', 'failed') == 'success')].copy()
            
            if len(mf_df) == 0:
                continue
            
            mf_df = mf_df.sort_values('actual_time')
            color = self.colors[i]
            label_prefix = os.path.basename(analyzer.csv_file) if len(self.csv_files) > 1 else ""
            
            # Generalized dimensions evolution
            if 'mf_D0' in mf_df.columns:
                ax1.plot(mf_df['actual_time'], mf_df['mf_D0'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} D₀', alpha=0.8)
            if 'mf_D1' in mf_df.columns:
                ax1.plot(mf_df['actual_time'], mf_df['mf_D1'], 's--', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} D₁', alpha=0.6)
            if 'mf_D2' in mf_df.columns:
                ax1.plot(mf_df['actual_time'], mf_df['mf_D2'], '^:', color=color,
                        linewidth=2, markersize=3, label=f'{label_prefix} D₂', alpha=0.6)
            
            # Multifractal width
            if 'mf_alpha_width' in mf_df.columns:
                ax2.plot(mf_df['actual_time'], mf_df['mf_alpha_width'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} α width', alpha=0.8)
            
            # Degree of multifractality
            if 'mf_degree_multifractality' in mf_df.columns:
                ax3.plot(mf_df['actual_time'], mf_df['mf_degree_multifractality'], 'o-', color=color,
                        linewidth=2, markersize=4, label=f'{label_prefix} Degree MF', alpha=0.8)
                
                # Add classification thresholds
                ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # D₁ vs D₀ (multifractal scatter)
            if 'mf_D0' in mf_df.columns and 'mf_D1' in mf_df.columns:
                scatter = ax4.scatter(mf_df['mf_D0'], mf_df['mf_D1'], 
                                    c=mf_df['actual_time'], cmap='viridis',
                                    s=50, alpha=0.7, label=f'{label_prefix}')
                
                # Add diagonal line for reference
                ax4.plot([1, 2], [1, 2], 'k--', alpha=0.5, label='D₁ = D₀')
        
        # Set labels and titles
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Generalized Dimensions')
        ax1.set_title(self._get_title('Multifractal Dimensions Evolution'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('α Width')
        ax2.set_title(self._get_title('Multifractal Spectrum Width'))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Degree of Multifractality')
        ax3.set_title(self._get_title('Interface Classification'))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_xlabel('D₀ (Capacity Dimension)')
        ax4.set_ylabel('D₁ (Information Dimension)')
        ax4.set_title(self._get_title('Multifractal Phase Space'))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot('multifractal_analysis')
    
    def plot_all(self):
        """Generate all available plots based on the data."""
        print(f"\n📊 GENERATING ALL AVAILABLE PLOTS")
        print(f"Output directory: {self.output_dir}")
        print(f"Style: {self.style}")
        print(f"Format: {self.format} (DPI: {self.dpi})")
        
        plots_created = 0
        
        # Always try fractal plots
        if self.primary_analyzer.analysis_info['has_fractal']:
            self.plot_fractal_evolution()
            plots_created += 1
            
            if len(self.primary_analyzer.analysis_info['resolutions']) > 1:
                self.plot_fractal_convergence()
                plots_created += 1
        
        # Mixing plots
        if self.primary_analyzer.analysis_info['has_mixing']:
            self.plot_mixing_evolution()
            plots_created += 1
        
        # Physics plots
        if self.primary_analyzer.analysis_info['has_physics']:
            self.plot_dimensionless_physics()
            plots_created += 1
        
        # Multifractal plots
        if self.primary_analyzer.analysis_info['has_multifractal']:
            self.plot_multifractal()
            plots_created += 1
        
        print(f"\n✅ Created {plots_created} plots in {self.output_dir}")
        
        if self.interactive:
            print("\n🖱️  Interactive mode: Close plot windows to continue")
            plt.show()

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description='RT PLOTTER: Standalone plotting tool for RT analysis results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📊 PLOT TYPES (auto-detected from CSV data):
  fractal         Fractal dimension evolution/convergence
  mixing          Mixing thickness evolution (all methods)
  physics         Dimensionless physics plots (h/H vs τ)
  multifractal    Multifractal spectrum analysis
  all             Generate all available plots (default)

🎨 PLOTTING STYLES:
  journal         Clean, professional (12pt fonts, 300 DPI)
  presentation    Bold, high-contrast (16pt fonts, large markers)
  manuscript      Black & white, serif fonts (600 DPI)

📁 OUTPUT FORMATS:
  png             Portable Network Graphics (default)
  pdf             Portable Document Format (vector)
  svg             Scalable Vector Graphics (vector)
  eps             Encapsulated PostScript (vector)

📈 EXAMPLES:

# Simple fractal dimension plot
python rt_plotter.py results.csv --plots fractal

# High-res journal plots
python rt_plotter.py comprehensive_results.csv --style journal --dpi 600 --format pdf

# Compare multiple runs
python rt_plotter.py run1.csv run2.csv run3.csv --plots physics --style presentation

# All plots for manuscript
python rt_plotter.py results.csv --style manuscript --format svg --no-titles

# Interactive mode for data exploration
python rt_plotter.py results.csv --interactive --plots all
""")

    # Required arguments
    parser.add_argument('csv_files', nargs='+',
                       help='CSV file(s) from rt_analyzer.py or enhanced_analyzer.py')

    # Plot selection
    parser.add_argument('--plots', nargs='+', 
                       choices=['fractal', 'mixing', 'physics', 'multifractal', 'all'],
                       default=['all'],
                       help='Plot types to generate (default: all available)')

    # Styling options
    parser.add_argument('--style', choices=['journal', 'presentation', 'manuscript'],
                       default='journal',
                       help='Plotting style (default: journal)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg', 'eps'],
                       default='png',
                       help='Output format (default: png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution in DPI (default: 300)')

    # Output options
    parser.add_argument('--output-dir', default='./plots',
                       help='Output directory for plots (default: ./plots)')
    parser.add_argument('--no-titles', action='store_true',
                       help='Disable plot titles (clean for journals)')

    # Interactive options
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive plotting mode')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze CSV files, do not generate plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Validate CSV files
    for csv_file in args.csv_files:
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return 1

    print(f"🚀 RT PLOTTER - Standalone RT Analysis Plotting Tool")
    print(f"=" * 60)

    # Analyze CSV files
    analyzers = []
    for csv_file in args.csv_files:
        try:
            analyzer = CSVAnalyzer(csv_file)
            analyzers.append(analyzer)
            if args.verbose:
                analyzer.print_summary()
        except Exception as e:
            print(f"❌ Error analyzing {csv_file}: {str(e)}")
            return 1

    if args.analyze_only:
        print("\n📊 Analysis complete. No plots generated (--analyze-only mode).")
        return 0

    # Create plotter
    plotter = RTPlotter(
        csv_files=args.csv_files,
        output_dir=args.output_dir,
        style=args.style,
        format=args.format,
        dpi=args.dpi,
        no_titles=args.no_titles,
        interactive=args.interactive
    )

    # Generate plots
    try:
        if 'all' in args.plots:
            plotter.plot_all()
        else:
            plots_created = 0
            if 'fractal' in args.plots:
                plotter.plot_fractal_evolution()
                if len(plotter.primary_analyzer.analysis_info['resolutions']) > 1:
                    plotter.plot_fractal_convergence()
                plots_created += 1
            
            if 'mixing' in args.plots:
                plotter.plot_mixing_evolution()
                plots_created += 1
            
            if 'physics' in args.plots:
                plotter.plot_dimensionless_physics()
                plots_created += 1
            
            if 'multifractal' in args.plots:
                plotter.plot_multifractal()
                plots_created += 1
            
            print(f"\n✅ Created {plots_created} plot types in {args.output_dir}")

    except Exception as e:
        print(f"❌ Error generating plots: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    print(f"\n🎉 RT PLOTTER COMPLETE!")
    print(f"📁 Check {args.output_dir} for your plots")
    
    return 0

if __name__ == "__main__":
    exit(main())
