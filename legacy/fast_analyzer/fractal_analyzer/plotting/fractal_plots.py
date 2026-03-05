"""
Publication-quality plotting for fractal analysis results.

Provides clean, professional plots with EPS and PNG output options,
avoiding transparency issues for compatibility.
"""

import os
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass

from ..core.data_types import SegmentArray, FractalResult


@dataclass
class PlotConfig:
    """Configuration for plot appearance and output."""
    # Output formats
    save_eps: bool = False
    save_png: bool = True
    dpi: int = 300

    # Appearance
    use_titles: bool = True
    figure_size: Tuple[float, float] = (8, 6)
    font_size: int = 12
    line_width: float = 1.5
    marker_size: float = 4.0

    # Colors (no transparency to avoid EPS issues)
    primary_color: str = '#1f77b4'      # Blue
    secondary_color: str = '#ff7f0e'    # Orange
    fit_color: str = '#d62728'          # Red
    interface_color: str = '#2ca02c'    # Green
    background_color: str = '#ffffff'   # White
    grid_color: str = '#cccccc'         # Light gray
    text_color: str = '#000000'         # Black


class FractalPlotter:
    """
    Publication-quality plotter for fractal analysis results.

    Creates clean, professional plots suitable for journal publications
    with EPS and PNG output options.
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize plotter.

        Args:
            config: Plot configuration options
        """
        self.config = config or PlotConfig()
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Configure matplotlib for publication-quality output."""
        # Set backend and basic configuration
        mpl.rcParams['font.size'] = self.config.font_size
        mpl.rcParams['axes.linewidth'] = 1.0
        mpl.rcParams['grid.linewidth'] = 0.5
        mpl.rcParams['lines.linewidth'] = self.config.line_width
        mpl.rcParams['patch.linewidth'] = 0.5
        mpl.rcParams['xtick.major.width'] = 1.0
        mpl.rcParams['ytick.major.width'] = 1.0
        mpl.rcParams['xtick.minor.width'] = 0.5
        mpl.rcParams['ytick.minor.width'] = 0.5

        # Font configuration for better EPS compatibility
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Times', 'DejaVu Serif']
        mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

        # Disable transparency globally to avoid EPS issues
        mpl.rcParams['savefig.transparent'] = False
        mpl.rcParams['figure.facecolor'] = self.config.background_color
        mpl.rcParams['axes.facecolor'] = self.config.background_color
        mpl.rcParams['savefig.facecolor'] = self.config.background_color
        mpl.rcParams['savefig.edgecolor'] = 'none'

    def plot_box_counting_loglog(self, result: FractalResult, segments: SegmentArray,
                                output_dir: str, filename_base: str = "box_counting",
                                fractal_type: str = None, iteration_level: int = None) -> List[str]:
        """
        Create enhanced box counting log-log plot with clear scaling region visualization.

        Args:
            result: Fractal analysis result
            segments: Original segments (for metadata)
            output_dir: Output directory for plots
            filename_base: Base filename for output files
            fractal_type: Type of fractal (e.g., 'koch', 'sierpinski', 'hilbert') if generated
            iteration_level: Iteration level if fractal was generated

        Returns:
            List of saved file paths
        """
        # Create main plot with subplots for additional analysis
        fig = plt.figure(figsize=(12, 8))

        # Main box counting plot (larger subplot)
        ax_main = plt.subplot2grid((2, 2), (0, 0), colspan=2, rowspan=1)

        # Residuals plot (bottom left)
        ax_residuals = plt.subplot2grid((2, 2), (1, 0))

        # Statistics panel (bottom right)
        ax_stats = plt.subplot2grid((2, 2), (1, 1))

        log_sizes = np.log10(result.box_sizes)
        log_counts = np.log10(result.box_counts)

        # Determine scaling region
        scaling_mask = np.ones(len(result.box_sizes), dtype=bool)
        if hasattr(result, 'scaling_range') and result.scaling_range:
            min_size, max_size = result.scaling_range
            scaling_mask = (result.box_sizes >= min_size) & (result.box_sizes <= max_size)

        # Plot excluded points (grayed out)
        excluded_mask = ~scaling_mask
        if np.any(excluded_mask):
            ax_main.scatter(log_sizes[excluded_mask], log_counts[excluded_mask],
                          color='lightgray',
                          s=self.config.marker_size**2,
                          marker='o',
                          edgecolors='gray',
                          linewidth=0.5,
                          label='Excluded points')

        # Plot scaling region points (prominent)
        if np.any(scaling_mask):
            ax_main.scatter(log_sizes[scaling_mask], log_counts[scaling_mask],
                          color=self.config.primary_color,
                          s=(self.config.marker_size * 1.3)**2,
                          marker='o',
                          edgecolors=self.config.text_color,
                          linewidth=1.0,
                          label=f'Scaling region ({np.sum(scaling_mask)} points)')

        # Plot fitted line only through scaling region
        if hasattr(result, 'slope') and hasattr(result, 'intercept'):
            if np.any(scaling_mask):
                fit_x = log_sizes[scaling_mask]
                fit_y = result.slope * fit_x + result.intercept
                ax_main.plot(fit_x, fit_y,
                           color=self.config.fit_color,
                           linewidth=self.config.line_width * 1.5,
                           linestyle='-',
                           label=f'Linear fit (D = {result.dimension:.4f})')

        # Formatting main plot
        ax_main.set_xlabel('log₁₀(Box Size)')
        ax_main.set_ylabel('log₁₀(Box Count)')

        if self.config.use_titles:
            # Add quality indicator to title
            quality_indicator = "✓" if result.r_squared >= 0.99 else "⚠"
            quality_text = "High Quality" if result.r_squared >= 0.99 else "Low Quality"
            title_color = 'black' if result.r_squared >= 0.99 else 'red'

            # Build title with fractal context if available
            if fractal_type and iteration_level is not None:
                # Capitalize and format fractal type name
                fractal_name = {
                    'koch': 'Koch Curve',
                    'sierpinski': 'Sierpinski Triangle',
                    'dragon': 'Dragon Curve',
                    'minkowski': 'Minkowski Sausage',
                    'hilbert': 'Hilbert Curve'
                }.get(fractal_type.lower(), fractal_type.title())

                title_main = f'{fractal_name} Analysis (Level {iteration_level}): D = {result.dimension:.4f} | R² = {result.r_squared:.4f} {quality_indicator}'
            else:
                title_main = f'Fractal Dimension Analysis: D = {result.dimension:.4f} | R² = {result.r_squared:.4f} {quality_indicator}'

            ax_main.set_title(f'{title_main}\n'
                            f'Quality: {quality_text} (Threshold: R² ≥ 0.99)',
                            color=title_color)

        ax_main.grid(True, color=self.config.grid_color, linestyle='-', linewidth=0.5)

        # Improved legend positioning - try different locations to avoid data
        legend_locations = ['upper left', 'upper right', 'lower left', 'lower right']
        ax_main.legend(loc='best')

        # --- RESIDUALS PLOT ---
        if hasattr(result, 'slope') and hasattr(result, 'intercept') and np.any(scaling_mask):
            fit_values = result.slope * log_sizes[scaling_mask] + result.intercept
            residuals = log_counts[scaling_mask] - fit_values

            ax_residuals.scatter(log_sizes[scaling_mask], residuals,
                               color=self.config.secondary_color,
                               s=self.config.marker_size**2,
                               )
            ax_residuals.axhline(y=0, color=self.config.text_color, linestyle='--')
            ax_residuals.set_xlabel('log₁₀(Box Size)')
            ax_residuals.set_ylabel('Residuals')
            ax_residuals.set_title('Fit Quality')
            ax_residuals.grid(True)

        # --- STATISTICS PANEL ---
        ax_stats.axis('off')
        stats_text = self._create_enhanced_stats_text(result, segments, fractal_type, iteration_level)

        # Color-code statistics panel based on quality
        if result.r_squared >= 0.99:
            stats_edge_color = self.config.text_color
            stats_face_color = self.config.background_color
        else:
            stats_edge_color = 'red'
            stats_face_color = '#ffe6e6'  # Light red background for warnings

        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5',
                              facecolor=stats_face_color,
                              edgecolor=stats_edge_color,
                              linewidth=2.0 if result.r_squared < 0.99 else 0.5))

        plt.tight_layout()

        # Save plots
        saved_files = self._save_figure(fig, output_dir, filename_base)
        plt.close(fig)

        return saved_files

    def plot_interface_segments(self, segments: SegmentArray, output_dir: str,
                               filename_base: str = "interface",
                               max_segments: int = 2000,
                               fractal_type: str = None, iteration_level: int = None) -> List[str]:
        """
        Plot interface segments to visualize extracted contours.

        Args:
            segments: Interface segments to plot
            output_dir: Output directory for plots
            filename_base: Base filename for output files
            max_segments: Maximum segments to plot (for performance)
            fractal_type: Type of fractal (e.g., 'koch', 'sierpinski', 'hilbert') if generated
            iteration_level: Iteration level if fractal was generated

        Returns:
            List of saved file paths
        """
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Handle sampling based on data type
        if fractal_type and iteration_level is not None:
            # For generated fractals, show all segments (they're mathematically perfect)
            plot_segments = segments.segments
            title_suffix = f" ({segments.n_segments} segments)"
        elif segments.n_segments > max_segments:
            # For large imported datasets, use intelligent sampling
            step = segments.n_segments / max_segments

            # Create indices that span the entire curve including the end
            indices = []
            for i in range(max_segments):
                idx = int(i * step)
                if idx < segments.n_segments:
                    indices.append(idx)

            # Always include the last segment to ensure complete curve visualization
            if indices[-1] != segments.n_segments - 1:
                indices[-1] = segments.n_segments - 1

            indices = np.array(indices)
            plot_segments = segments.segments[indices]
            title_suffix = f" (showing {len(indices)}/{segments.n_segments} segments)"
        else:
            plot_segments = segments.segments
            title_suffix = f" ({segments.n_segments} segments)"

        # Plot segments as lines
        for segment in plot_segments:
            x_coords = [segment[0][0], segment[1][0]]
            y_coords = [segment[0][1], segment[1][1]]
            ax.plot(x_coords, y_coords,
                   color=self.config.interface_color,
                   linewidth=0.8)

        # Formatting
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')

        if self.config.use_titles:
            # Build title with fractal context if available
            if fractal_type and iteration_level is not None:
                # Capitalize and format fractal type name
                fractal_name = {
                    'koch': 'Koch Curve',
                    'sierpinski': 'Sierpinski Triangle',
                    'dragon': 'Dragon Curve',
                    'minkowski': 'Minkowski Sausage',
                    'hilbert': 'Hilbert Curve'
                }.get(fractal_type.lower(), fractal_type.title())

                title_main = f'{fractal_name} Interface (Level {iteration_level}){title_suffix}'
            else:
                title_main = f'Interface Segments{title_suffix}'

            ax.set_title(title_main)

        # Expand plot domain by 10% to give breathing room
        bbox = segments.bbox
        x_margin = bbox.width * 0.10
        y_margin = bbox.height * 0.10
        ax.set_xlim(bbox.min_x - x_margin, bbox.max_x + x_margin)
        ax.set_ylim(bbox.min_y - y_margin, bbox.max_y + y_margin)

        ax.grid(True, color=self.config.grid_color, linestyle='-', linewidth=0.5)

        # Add compact metadata inside the plot (bottom-left corner)
        if fractal_type and iteration_level is not None:
            # For generated fractals, show fractal identification (two lines, smaller)
            fractal_name = {
                'koch': 'Koch Curve',
                'sierpinski': 'Sierpinski Triangle',
                'dragon': 'Dragon Curve',
                'minkowski': 'Minkowski Sausage',
                'hilbert': 'Hilbert Curve'
            }.get(fractal_type.lower(), fractal_type.title())

            info_text = f'{fractal_name} Level {iteration_level}\nTotal length: {segments.total_length:.4f}'
        else:
            # For imported data, show bounding box information
            info_text = (f'Bounding box: ({bbox.min_x:.4f}, {bbox.min_y:.4f}) to\n'
                        f'({bbox.max_x:.4f}, {bbox.max_y:.4f})\n'
                        f'Total length: {segments.total_length:.4f}')

        # Place compact text inside plot (bottom-left corner)
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='left',
                fontsize=8,  # Smaller font
                bbox=dict(boxstyle='round,pad=0.3',  # Compact padding
                         facecolor=self.config.background_color,
                         edgecolor=self.config.text_color,
                         linewidth=0.5,
                         alpha=0.9))  # Slight transparency

        plt.tight_layout()

        # Save plots
        saved_files = self._save_figure(fig, output_dir, filename_base)
        plt.close(fig)

        return saved_files

    def plot_multi_scale_analysis(self, results: List[Dict[str, Any]], output_dir: str,
                                 filename_base: str = "multi_scale") -> List[str]:
        """
        Plot multi-scale analysis results (from --analyze_iterations).
        Shows all results (valid and invalid) with R² values superimposed.

        Args:
            results: List of results from multiple iteration levels
            output_dir: Output directory for plots
            filename_base: Base filename for output files

        Returns:
            List of saved file paths
        """
        if not results:
            return []

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0],
                                                     self.config.figure_size[1] * 1.5))

        levels = [r['level'] for r in results]
        dimensions = [r['dimension'] for r in results]
        r_squared = [r['r_squared'] for r in results]
        errors = [r['error'] for r in results if 'error' in r]
        valid_flags = [r['valid'] for r in results]

        # Separate valid and invalid results for different styling
        valid_results = [i for i, valid in enumerate(valid_flags) if valid]
        invalid_results = [i for i, valid in enumerate(valid_flags) if not valid]

        # Plot 1: Fractal dimension vs iteration level with R² annotations

        # Plot valid results (prominent)
        if valid_results:
            valid_levels = [levels[i] for i in valid_results]
            valid_dimensions = [dimensions[i] for i in valid_results]
            valid_r_squared_vals = [r_squared[i] for i in valid_results]

            ax1.scatter(valid_levels, valid_dimensions,
                       color=self.config.primary_color,
                       s=self.config.marker_size**2,
                       marker='o',
                       edgecolors=self.config.text_color,
                       linewidth=0.5,
                       label=f'Valid results (R² ≥ 0.99)')

            ax1.plot(valid_levels, valid_dimensions,
                    color=self.config.primary_color,
                    linewidth=1.0,
                    linestyle='-')

            # Add R² annotations for valid results
            for level, dim, r2 in zip(valid_levels, valid_dimensions, valid_r_squared_vals):
                ax1.annotate(f'{r2:.3f}',
                           xy=(level, dim),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           color=self.config.primary_color,
                           weight='bold')

        # Plot invalid results (grayed out)
        if invalid_results:
            invalid_levels = [levels[i] for i in invalid_results]
            invalid_dimensions = [dimensions[i] for i in invalid_results]
            invalid_r_squared_vals = [r_squared[i] for i in invalid_results]

            ax1.scatter(invalid_levels, invalid_dimensions,
                       color='red',
                       s=self.config.marker_size**2,
                       marker='x',
                       linewidth=1.0,
                       label=f'Invalid results (R² < 0.99)')

            # Add R² annotations for invalid results
            for level, dim, r2 in zip(invalid_levels, invalid_dimensions, invalid_r_squared_vals):
                ax1.annotate(f'{r2:.3f}',
                           xy=(level, dim),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           color='red',
                           style='italic')

        # Add theoretical dimension line if errors are available
        if errors and len(errors) == len(results) and valid_results:
            valid_dims = [dimensions[i] for i in valid_results]
            theoretical_dim = valid_dims[0] + errors[valid_results[0]] if errors[valid_results[0]] != 0 else None
            if theoretical_dim:
                ax1.axhline(y=theoretical_dim,
                           color=self.config.secondary_color,
                           linestyle='--',
                           linewidth=self.config.line_width,
                           label=f'Theoretical (D = {theoretical_dim:.3f})')

        ax1.set_xlabel('Iteration Level')
        ax1.set_ylabel('Fractal Dimension')
        ax1.grid(True, color=self.config.grid_color, linestyle='-', linewidth=0.5)
        ax1.legend()

        # Plot 2: R-squared vs iteration level (all results)
        # Color code by quality
        high_quality = [i for i, r2 in enumerate(r_squared) if r2 >= 0.99]
        low_quality = [i for i, r2 in enumerate(r_squared) if r2 < 0.99]

        if high_quality:
            hq_levels = [levels[i] for i in high_quality]
            hq_r_squared = [r_squared[i] for i in high_quality]
            ax2.scatter(hq_levels, hq_r_squared,
                       color=self.config.fit_color,
                       s=self.config.marker_size**2,
                       marker='s',
                       edgecolors=self.config.text_color,
                       linewidth=0.5,
                       label='High quality (R² ≥ 0.99)')

        if low_quality:
            lq_levels = [levels[i] for i in low_quality]
            lq_r_squared = [r_squared[i] for i in low_quality]
            ax2.scatter(lq_levels, lq_r_squared,
                       color='red',
                       s=self.config.marker_size**2,
                       marker='x',
                       linewidth=1.0,
                       label='Low quality (R² < 0.99)')

        # Connect all points with line
        ax2.plot(levels, r_squared,
                color='gray',
                linewidth=1.0,
                linestyle='-',
                alpha=0.7)

        # Add R² threshold line
        ax2.axhline(y=0.99, color='red', linestyle='--', linewidth=1.0,
                   label='Quality threshold (R² = 0.99)')

        ax2.set_xlabel('Iteration Level')
        ax2.set_ylabel('R-squared')
        ax2.grid(True, color=self.config.grid_color, linestyle='-', linewidth=0.5)
        ax2.legend()

        # Adjust y-limits to show all data
        if r_squared:
            min_r2 = min(r_squared)
            max_r2 = max(r_squared)
            padding = (max_r2 - min_r2) * 0.1
            ax2.set_ylim(max(0.8, min_r2 - padding), min(1.0, max_r2 + padding))

        if self.config.use_titles:
            ax1.set_title('Multi-Scale Fractal Analysis with R² Quality Indicators')

        plt.tight_layout()

        # Save plots
        saved_files = self._save_figure(fig, output_dir, filename_base)
        plt.close(fig)

        return saved_files

    def plot_performance_comparison(self, performance_data: Dict[str, Any],
                                   output_dir: str,
                                   filename_base: str = "performance") -> List[str]:
        """
        Plot performance comparison showing timing and optimization benefits.

        Args:
            performance_data: Dictionary with performance metrics
            output_dir: Output directory for plots
            filename_base: Base filename for output files

        Returns:
            List of saved file paths
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                                     figsize=(self.config.figure_size[0] * 1.5,
                                                             self.config.figure_size[1] * 1.2))

        # This is a placeholder structure - will be expanded based on actual performance data
        # that gets collected during analysis

        # Plot 1: Algorithm selection (spatial index vs vectorized)
        if 'algorithm_usage' in performance_data:
            usage_data = performance_data['algorithm_usage']
            ax1.bar(usage_data.keys(), usage_data.values(),
                   color=self.config.primary_color,
                   edgecolor=self.config.text_color,
                   linewidth=0.5)
            ax1.set_title('Algorithm Usage')
            ax1.set_ylabel('Frequency')

        # Plot 2: Cache performance
        if 'cache_stats' in performance_data:
            cache_data = performance_data['cache_stats']
            cache_labels = ['Hits', 'Misses']
            cache_values = [cache_data.get('hits', 0), cache_data.get('misses', 0)]
            ax2.pie(cache_values, labels=cache_labels,
                   colors=[self.config.primary_color, self.config.secondary_color],
                   startangle=90)
            ax2.set_title('Cache Performance')

        # Plot 3: Timing breakdown
        if 'timing_breakdown' in performance_data:
            timing_data = performance_data['timing_breakdown']
            ax3.barh(list(timing_data.keys()), list(timing_data.values()),
                    color=self.config.interface_color,
                    edgecolor=self.config.text_color,
                    linewidth=0.5)
            ax3.set_title('Analysis Time Breakdown')
            ax3.set_xlabel('Time (seconds)')

        # Plot 4: Segment count optimization
        if 'segment_optimization' in performance_data:
            opt_data = performance_data['segment_optimization']
            categories = ['Original', 'Optimized']
            values = [opt_data.get('original', 0), opt_data.get('optimized', 0)]
            ax4.bar(categories, values,
                   color=[self.config.secondary_color, self.config.primary_color],
                   edgecolor=self.config.text_color,
                   linewidth=0.5)
            ax4.set_title('Segment Count Reduction')
            ax4.set_ylabel('Number of Segments')

        # Remove empty subplots
        for ax in [ax1, ax2, ax3, ax4]:
            if not ax.has_data():
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

        # Save plots
        saved_files = self._save_figure(fig, output_dir, filename_base)
        plt.close(fig)

        return saved_files

    def _create_stats_text(self, result: FractalResult, segments: SegmentArray) -> str:
        """Create statistics text for box counting plot."""
        stats_lines = [
            f'Segments: {segments.n_segments}',
            f'Total length: {segments.total_length:.4f}',
            f'Data points: {result.n_points}',
            f'R² = {result.r_squared:.4f}',
        ]

        if hasattr(result, 'scaling_range') and result.scaling_range:
            min_size, max_size = result.scaling_range
            decades = np.log10(max_size / min_size)
            stats_lines.append(f'Scaling: {decades:.1f} decades')

        return '\n'.join(stats_lines)

    def _create_enhanced_stats_text(self, result: FractalResult, segments: SegmentArray,
                                  fractal_type: str = None, iteration_level: int = None) -> str:
        """Create enhanced statistics text with more details."""
        stats_lines = []

        # Add fractal context if available
        if fractal_type and iteration_level is not None:
            fractal_name = {
                'koch': 'Koch Curve',
                'sierpinski': 'Sierpinski Triangle',
                'dragon': 'Dragon Curve',
                'minkowski': 'Minkowski Sausage',
                'hilbert': 'Hilbert Curve'
            }.get(fractal_type.lower(), fractal_type.title())

            # Add theoretical dimension if known
            theoretical_dims = {
                'koch': 1.2619,
                'sierpinski': 1.5850,
                'dragon': 2.0000,
                'minkowski': 1.5000,
                'hilbert': 2.0000
            }
            theoretical_dim = theoretical_dims.get(fractal_type.lower())

            stats_lines.extend([
                f'Fractal Type: {fractal_name}',
                f'Iteration Level: {iteration_level}',
                f'',
            ])

            if theoretical_dim:
                error_pct = abs(result.dimension - theoretical_dim) / theoretical_dim * 100
                stats_lines.extend([
                    f'Theoretical D: {theoretical_dim:.4f}',
                    f'Computed D: {result.dimension:.6f}',
                    f'Error: {error_pct:.2f}%',
                    f'',
                ])
            else:
                stats_lines.extend([
                    f'Fractal Dimension: {result.dimension:.6f}',
                    f'',
                ])
        else:
            stats_lines.extend([
                f'Fractal Dimension: {result.dimension:.6f}',
                f'',
            ])

        stats_lines.extend([
            f'R-squared: {result.r_squared:.6f}',
            f'',
            f'Input Data:',
            f'  • Segments: {segments.n_segments:,}',
            f'  • Total length: {segments.total_length:.4f}',
            f'',
            f'Box Counting:',
            f'  • Total points: {len(result.box_sizes)}',
            f'  • Used points: {result.n_points}',
        ])

        if hasattr(result, 'scaling_range') and result.scaling_range:
            min_size, max_size = result.scaling_range
            decades = np.log10(max_size / min_size)
            stats_lines.extend([
                f'  • Box range: {max_size:.1e} to {min_size:.1e}',
                f'  • Scaling range: {decades:.1f} decades'
            ])

        # Add quality assessment with emphasis on R² threshold
        if result.r_squared >= 0.99:
            quality = 'Excellent ✓'
            quality_note = '(Meets R² ≥ 0.99 threshold)'
        elif result.r_squared > 0.95:
            quality = 'Good ⚠'
            quality_note = '(Below R² ≥ 0.99 threshold)'
        elif result.r_squared > 0.90:
            quality = 'Fair ⚠'
            quality_note = '(Below R² ≥ 0.99 threshold)'
        else:
            quality = 'Poor ✗'
            quality_note = '(Far below R² ≥ 0.99 threshold)'

        stats_lines.extend([
            f'',
            f'Quality: {quality}',
            f'R² Status: {quality_note}',
            f'Valid result: {"Yes" if result.is_valid else "No"}',
        ])

        # Add warning for low quality results
        if result.r_squared < 0.99:
            stats_lines.extend([
                f'',
                f'⚠ WARNING: Low R² value!',
                f'Consider adjusting parameters',
                f'or examining data quality.'
            ])

        return '\n'.join(stats_lines)

    def plot_sliding_window_analysis(self, result: FractalResult, segments: SegmentArray,
                                    output_dir: str, filename_base: str = "sliding_window") -> List[str]:
        """
        Create sliding window analysis plot showing how dimension and R² vary with window selection.

        Args:
            result: Fractal analysis result
            segments: Original segments
            output_dir: Output directory for plots
            filename_base: Base filename for output files

        Returns:
            List of saved file paths
        """
        # This requires the sliding window analysis data from the analyzer
        # For now, we'll create a demonstration of what the analysis looks like

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        log_sizes = np.log10(result.box_sizes)
        log_counts = np.log10(result.box_counts)

        # Simulate sliding window analysis
        n_points = len(result.box_sizes)
        min_window_size = 5  # Minimum points for reliable fit

        window_starts = []
        window_ends = []
        dimensions = []
        r_squared_values = []

        # Simulate different window positions
        for start in range(n_points - min_window_size + 1):
            for end in range(start + min_window_size, n_points + 1):
                window_log_sizes = log_sizes[start:end]
                window_log_counts = log_counts[start:end]

                if len(window_log_sizes) >= min_window_size:
                    # Fit line to this window
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        window_log_sizes, window_log_counts)

                    window_starts.append(start)
                    window_ends.append(end - 1)
                    dimensions.append(-slope)  # Negative slope = positive dimension
                    r_squared_values.append(r_value**2)

        # Convert to arrays
        window_starts = np.array(window_starts)
        window_ends = np.array(window_ends)
        dimensions = np.array(dimensions)
        r_squared_values = np.array(r_squared_values)

        # Plot 1: Original data with different window options
        ax1.scatter(log_sizes, log_counts, color='lightgray', s=30, label='All data')

        # Highlight the selected window
        if hasattr(result, 'scaling_range') and result.scaling_range:
            min_size, max_size = result.scaling_range
            scaling_mask = (result.box_sizes >= min_size) & (result.box_sizes <= max_size)
            ax1.scatter(log_sizes[scaling_mask], log_counts[scaling_mask],
                       color=self.config.primary_color, s=60,
                       label=f'Selected window (D = {result.dimension:.3f})')

        ax1.set_xlabel('log₁₀(Box Size)')
        ax1.set_ylabel('log₁₀(Box Count)')
        ax1.set_title('Box Counting Data with Selected Scaling Window')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Dimension vs window position
        if len(dimensions) > 0:
            # Color code by R-squared
            scatter = ax2.scatter(window_starts, dimensions, c=r_squared_values,
                                 cmap='viridis', s=30)

            # Highlight the selected window
            best_idx = np.argmax(r_squared_values)
            ax2.scatter(window_starts[best_idx], dimensions[best_idx],
                       color='red', s=100, marker='*',
                       label=f'Selected (R² = {r_squared_values[best_idx]:.4f})')

            ax2.set_xlabel('Window Start Index')
            ax2.set_ylabel('Fractal Dimension')
            ax2.set_title('Dimension Estimates vs Window Position')
            ax2.legend()
            ax2.grid(True)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('R-squared')

        # Plot 3: R-squared vs window size
        if len(r_squared_values) > 0:
            window_sizes = window_ends - window_starts + 1
            ax3.scatter(window_sizes, r_squared_values, color=self.config.secondary_color,
                       s=30)

            # Highlight the selected window
            selected_size = np.sum(scaling_mask) if 'scaling_mask' in locals() else result.n_points
            selected_r2 = result.r_squared
            ax3.scatter(selected_size, selected_r2, color='red', s=100, marker='*',
                       label=f'Selected ({selected_size} points)')

            ax3.set_xlabel('Window Size (number of points)')
            ax3.set_ylabel('R-squared')
            ax3.set_title('Fit Quality vs Window Size')
            ax3.legend()
            ax3.grid(True)

        plt.tight_layout()

        # Save plots
        saved_files = self._save_figure(fig, output_dir, filename_base)
        plt.close(fig)

        return saved_files

    def _estimate_error(self, result: FractalResult) -> float:
        """Estimate uncertainty in fractal dimension."""
        # Simple error estimate based on R-squared
        if result.r_squared > 0.99:
            return 0.01
        elif result.r_squared > 0.95:
            return 0.02
        else:
            return 0.05

    def _save_figure(self, fig, output_dir: str, filename_base: str) -> List[str]:
        """
        Save figure in requested formats.

        Args:
            fig: Matplotlib figure
            output_dir: Output directory
            filename_base: Base filename

        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        # Save PNG
        if self.config.save_png:
            png_path = os.path.join(output_dir, f"{filename_base}.png")
            fig.savefig(png_path, format='png', dpi=self.config.dpi,
                       bbox_inches='tight', facecolor=self.config.background_color,
                       edgecolor='none', transparent=False)
            saved_files.append(png_path)

        # Save EPS
        if self.config.save_eps:
            eps_path = os.path.join(output_dir, f"{filename_base}.eps")
            fig.savefig(eps_path, format='eps', bbox_inches='tight',
                       facecolor=self.config.background_color, edgecolor='none',
                       transparent=False)
            saved_files.append(eps_path)

        return saved_files