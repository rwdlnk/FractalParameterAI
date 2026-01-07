#!/usr/bin/env python3
"""
Modified Simple Fractal Analyzer - Koch Curve Generator and Analyzer

Generates Koch curve at specified level, saves segments to file,
and performs box-counting fractal dimension analysis with visualization.

Usage:
    python simple_analyzer_mod.py --level 5
    python simple_analyzer_mod.py --level 4 --output koch_level4.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse


def generate_koch_curve(level):
    """
    Generate Koch curve segments using recursive subdivision.

    The Koch curve is constructed by iteratively replacing each line segment
    with 4 smaller segments forming a triangular bump.

    Parameters:
    -----------
    level : int
        Iteration level (0 = straight line, higher = more detail)

    Returns:
    --------
    segments : numpy array of shape (n, 4)
        Each row is [x1, y1, x2, y2] for a segment
    """

    def koch_subdivide(p1, p2, current_level):
        """Recursively subdivide a segment into Koch curve pattern."""
        if current_level == 0:
            # Base case: return the segment as-is
            return [(*p1, *p2)]

        x1, y1 = p1
        x2, y2 = p2

        # Calculate the 5 key points of Koch subdivision
        # Point 1: Start (original p1)
        # Point 2: 1/3 along the segment
        p2_x = x1 + (x2 - x1) / 3
        p2_y = y1 + (y2 - y1) / 3

        # Point 3: Peak of the equilateral triangle
        # Rotate the vector from p2 to center by 60 degrees
        dx = (x2 - x1) / 3
        dy = (y2 - y1) / 3
        angle = np.pi / 3  # 60 degrees
        p3_x = p2_x + dx * np.cos(angle) - dy * np.sin(angle)
        p3_y = p2_y + dx * np.sin(angle) + dy * np.cos(angle)

        # Point 4: 2/3 along the segment
        p4_x = x1 + 2 * (x2 - x1) / 3
        p4_y = y1 + 2 * (y2 - y1) / 3

        # Point 5: End (original p2)

        # Recursively subdivide each of the 4 segments
        segments = []
        segments.extend(koch_subdivide((x1, y1), (p2_x, p2_y), current_level - 1))
        segments.extend(koch_subdivide((p2_x, p2_y), (p3_x, p3_y), current_level - 1))
        segments.extend(koch_subdivide((p3_x, p3_y), (p4_x, p4_y), current_level - 1))
        segments.extend(koch_subdivide((p4_x, p4_y), (x2, y2), current_level - 1))

        return segments

    # Start with a unit horizontal segment from (0, 0) to (1, 0)
    segments = koch_subdivide((0.0, 0.0), (1.0, 0.0), level)

    return np.array(segments)


def save_segments_to_file(segments, filename):
    """
    Save line segments to a text file.

    Parameters:
    -----------
    segments : numpy array of shape (n, 4)
        Line segments [x1, y1, x2, y2]
    filename : str
        Output filename
    """
    with open(filename, 'w') as f:
        f.write("# Koch curve line segment data\n")
        f.write("# Format: x1 y1 x2 y2\n")
        f.write(f"# Total segments: {len(segments)}\n")
        f.write("#\n")

        for seg in segments:
            f.write(f"{seg[0]:.10f} {seg[1]:.10f} {seg[2]:.10f} {seg[3]:.10f}\n")

    print(f"Saved {len(segments)} segments to: {filename}")


def check_segment_box_intersection(seg, box_min, box_max):
    """
    Check if a line segment intersects a rectangular box.

    Parameters:
    -----------
    seg : array-like [x1, y1, x2, y2]
        Line segment endpoints
    box_min : array-like [xmin, ymin]
        Bottom-left corner of box
    box_max : array-like [xmax, ymax]
        Top-right corner of box

    Returns:
    --------
    bool : True if segment intersects box
    """
    x1, y1, x2, y2 = seg
    xmin, ymin = box_min
    xmax, ymax = box_max

    # Quick rejection: check if both endpoints are completely outside box
    if (x1 < xmin and x2 < xmin) or (x1 > xmax and x2 > xmax):
        return False
    if (y1 < ymin and y2 < ymin) or (y1 > ymax and y2 > ymax):
        return False

    # Check if either endpoint is inside the box
    if (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
        return True
    if (xmin <= x2 <= xmax and ymin <= y2 <= ymax):
        return True

    # Check for edge intersections using parametric line equation
    dx = x2 - x1
    dy = y2 - y1

    # Check intersection with box edges
    if abs(dx) > 1e-10:  # Not vertical
        # Left edge (x = xmin)
        t = (xmin - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True

        # Right edge (x = xmax)
        t = (xmax - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True

    if abs(dy) > 1e-10:  # Not horizontal
        # Bottom edge (y = ymin)
        t = (ymin - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True

        # Top edge (y = ymax)
        t = (ymax - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True

    return False


def count_boxes_intersecting_curve(segments, delta, domain_x, domain_y):
    """
    Count boxes of size delta that intersect the curve.

    Parameters:
    -----------
    segments : numpy array of shape (n, 4)
        Line segments [x1, y1, x2, y2]
    delta : float
        Box size
    domain_x : tuple (xmin, xmax)
        X domain bounds
    domain_y : tuple (ymin, ymax)
        Y domain bounds

    Returns:
    --------
    int : Number of boxes intersecting the curve
    """
    xmin, xmax = domain_x
    ymin, ymax = domain_y

    # Create grid of boxes
    nx = int(np.ceil((xmax - xmin) / delta))
    ny = int(np.ceil((ymax - ymin) / delta))

    intersecting_boxes = set()

    for segment in segments:
        # For each segment, check which boxes it intersects
        for i in range(nx):
            for j in range(ny):
                box_min = np.array([xmin + i * delta, ymin + j * delta])
                box_max = np.array([xmin + (i + 1) * delta, ymin + (j + 1) * delta])

                if check_segment_box_intersection(segment, box_min, box_max):
                    intersecting_boxes.add((i, j))

    return len(intersecting_boxes)


def analyze_koch_curve(level, initial_delta=0.5, delta_factor=1.5, num_steps=15,
                       output_file=None, save_plot=None):
    """
    Generate Koch curve, perform box-counting analysis, and visualize results.

    Parameters:
    -----------
    level : int
        Koch curve iteration level
    initial_delta : float
        Initial box size (largest)
    delta_factor : float
        Factor to divide delta by at each step
    num_steps : int
        Number of different box sizes to test
    output_file : str or None
        If provided, save segments to this file
    save_plot : str or None
        If provided, save plot to this file

    Returns:
    --------
    dict : Results including dimension, R², and data arrays
    """

    print("=" * 80)
    print("KOCH CURVE FRACTAL DIMENSION ANALYSIS")
    print("=" * 80)
    print(f"Koch curve level: {level}")
    print(f"Theoretical dimension: ~1.2619 (log(4)/log(3))")
    print()

    # Generate Koch curve
    print("Generating Koch curve...")
    segments = generate_koch_curve(level)
    n_segments = len(segments)
    print(f"Generated {n_segments} segments (4^{level} = {4**level})")

    # Save to file if requested
    if output_file:
        save_segments_to_file(segments, output_file)

    # Auto-detect domain
    x_coords = np.concatenate([segments[:, 0], segments[:, 2]])
    y_coords = np.concatenate([segments[:, 1], segments[:, 3]])

    xmin, xmax = x_coords.min(), x_coords.max()
    ymin, ymax = y_coords.min(), y_coords.max()

    # Add small padding
    padding_x = (xmax - xmin) * 0.02
    padding_y = max((ymax - ymin) * 0.02, 0.01)  # Ensure some y-padding even if flat

    domain_x = (xmin - padding_x, xmax + padding_x)
    domain_y = (ymin - padding_y, ymax + padding_y)

    domain_width = domain_x[1] - domain_x[0]
    domain_height = domain_y[1] - domain_y[0]

    print(f"Domain: X=[{domain_x[0]:.6f}, {domain_x[1]:.6f}], Y=[{domain_y[0]:.6f}, {domain_y[1]:.6f}]")
    print(f"Domain size: {domain_width:.6f} × {domain_height:.6f}")
    print()

    # Generate delta values (box sizes)
    deltas = []
    delta = initial_delta
    for i in range(num_steps):
        deltas.append(delta)
        delta = delta / delta_factor

    deltas = np.array(deltas)

    # Count boxes for each delta
    print("Performing box counting...")
    print(f"{'Step':<6} {'Delta':<12} {'N_boxes':<10} {'Time (s)':<10}")
    print("-" * 42)

    n_boxes = []
    import time
    for i, delta in enumerate(deltas):
        start_time = time.time()
        n = count_boxes_intersecting_curve(segments, delta, domain_x, domain_y)
        elapsed = time.time() - start_time
        n_boxes.append(n)
        print(f"{i+1:<6} {delta:<12.6f} {n:<10} {elapsed:<10.3f}")

    n_boxes = np.array(n_boxes)

    # Calculate logarithms for log-log plot
    log_inv_delta = np.log(1 / deltas)
    log_n_boxes = np.log(n_boxes)

    # Linear regression to find slope (fractal dimension)
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_inv_delta, log_n_boxes
    )

    r_squared = r_value ** 2

    # Theoretical dimension
    theoretical_dim = np.log(4) / np.log(3)
    error_percent = abs(slope - theoretical_dim) / theoretical_dim * 100

    # Print results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Fractal Dimension (measured): {slope:.8f}")
    print(f"Theoretical Dimension:         {theoretical_dim:.8f} (log(4)/log(3))")
    print(f"Error:                         {error_percent:.4f}%")
    print(f"R-squared:                     {r_squared:.10f}")
    print(f"Standard Error:                {std_err:.8f}")
    print(f"Intercept:                     {intercept:.8f}")
    print("=" * 80)
    print()

    # Create visualization
    fig = plt.figure(figsize=(16, 5))

    # Subplot 1: The Koch curve
    ax1 = plt.subplot(131)
    for seg in segments:
        ax1.plot([seg[0], seg[2]], [seg[1], seg[3]], 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('X', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax1.set_title(f'Koch Curve (Level {level})', fontsize=14, fontweight='bold')
    ax1.set_xlim(domain_x)
    ax1.set_ylim(domain_y)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.text(0.05, 0.95, f'{n_segments} segments',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Subplot 2: Log-log plot
    ax2 = plt.subplot(132)
    ax2.scatter(log_inv_delta, log_n_boxes, s=100, c='blue',
               alpha=0.7, edgecolors='darkblue', linewidth=2,
               label='Data points', zorder=3)

    # Fitted line
    fitted_line = slope * log_inv_delta + intercept
    ax2.plot(log_inv_delta, fitted_line, 'r-', linewidth=2.5,
            label=f'Fit: D = {slope:.6f}', zorder=2)

    ax2.set_xlabel('log(1/δ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('log(N_boxes)', fontsize=12, fontweight='bold')
    ax2.set_title('Log-Log Plot: Box-Counting', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='upper left')

    # Add detailed annotation
    textstr = f'Measured D = {slope:.6f}\n'
    textstr += f'Theoretical D = {theoretical_dim:.6f}\n'
    textstr += f'Error = {error_percent:.3f}%\n'
    textstr += f'R² = {r_squared:.8f}'

    ax2.text(0.95, 0.05, textstr,
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Subplot 3: Regular plot with log scale
    ax3 = plt.subplot(133)
    ax3.loglog(1/deltas, n_boxes, 'go-', markersize=8, linewidth=2,
               markeredgecolor='darkgreen', markeredgewidth=1.5, alpha=0.7)

    ax3.set_xlabel('1/δ (Resolution)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('N_boxes', fontsize=12, fontweight='bold')
    ax3.set_title('Box Count vs Resolution (log-log)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', which='both')

    # Add power law reference line
    ax3.text(0.05, 0.95, f'Power law: N ∝ (1/δ)^{slope:.3f}',
            transform=ax3.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_plot}")

    plt.show()

    # Return results
    return {
        'level': level,
        'n_segments': n_segments,
        'fractal_dimension': slope,
        'theoretical_dimension': theoretical_dim,
        'error_percent': error_percent,
        'r_squared': r_squared,
        'std_err': std_err,
        'intercept': intercept,
        'deltas': deltas,
        'n_boxes': n_boxes,
        'log_inv_delta': log_inv_delta,
        'log_n_boxes': log_n_boxes,
        'segments': segments,
        'domain_x': domain_x,
        'domain_y': domain_y
    }


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Koch Curve Fractal Dimension Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze Koch curve at level 5 (default)
  python %(prog)s

  # Specify level
  python %(prog)s --level 4

  # Save segments to file
  python %(prog)s --level 5 --output koch_level5.txt

  # Save plot
  python %(prog)s --level 6 --save-plot koch_level6.png

  # Customize box counting parameters
  python %(prog)s --level 5 --initial-delta 0.8 --delta-factor 2.0 --num-steps 20

  # Complete analysis with all outputs
  python %(prog)s --level 5 --output koch_segments.txt --save-plot koch_analysis.png

Note: Higher levels generate more segments (4^level):
  Level 3: 64 segments    (fast)
  Level 4: 256 segments   (fast)
  Level 5: 1024 segments  (moderate)
  Level 6: 4096 segments  (slower)
  Level 7: 16384 segments (very slow)
        """
    )

    parser.add_argument('--level', type=int, default=5,
                       help='Koch curve iteration level (default: 5)')
    parser.add_argument('--initial-delta', type=float, default=0.5,
                       help='Initial box size (default: 0.5)')
    parser.add_argument('--delta-factor', type=float, default=1.5,
                       help='Factor to divide delta by each step (default: 1.5)')
    parser.add_argument('--num-steps', type=int, default=15,
                       help='Number of box sizes to test (default: 15)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for segment data (optional)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to file (optional)')

    args = parser.parse_args()

    # Validate level
    if args.level < 0:
        print("Error: Level must be non-negative")
        return

    if args.level > 8:
        print(f"Warning: Level {args.level} will generate {4**args.level} segments.")
        print("This may be very slow. Consider using level 6 or lower.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Run analysis
    results = analyze_koch_curve(
        level=args.level,
        initial_delta=args.initial_delta,
        delta_factor=args.delta_factor,
        num_steps=args.num_steps,
        output_file=args.output,
        save_plot=args.save_plot
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Koch curve level:      {results['level']}")
    print(f"Number of segments:    {results['n_segments']}")
    print(f"Measured dimension:    {results['fractal_dimension']:.8f}")
    print(f"Theoretical dimension: {results['theoretical_dimension']:.8f}")
    print(f"Error:                 {results['error_percent']:.4f}%")
    print(f"R-squared:             {results['r_squared']:.10f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
