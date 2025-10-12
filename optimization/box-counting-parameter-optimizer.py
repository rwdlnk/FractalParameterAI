import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import re

def parse_segment_file(filename):
    """Parse file with segment coordinates."""
    segments = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(numbers) >= 4:
                x1, y1, x2, y2 = map(float, numbers[:4])
                segments.append([x1, y1, x2, y2])
    return np.array(segments)


def check_segment_box_intersection(seg, box_min, box_max):
    """Check if line segment intersects box."""
    x1, y1, x2, y2 = seg
    xmin, ymin = box_min
    xmax, ymax = box_max
    
    if (x1 < xmin and x2 < xmin) or (x1 > xmax and x2 > xmax):
        return False
    if (y1 < ymin and y2 < ymin) or (y1 > ymax and y2 > ymax):
        return False
    
    if (xmin <= x1 <= xmax and ymin <= y1 <= ymax):
        return True
    if (xmin <= x2 <= xmax and ymin <= y2 <= ymax):
        return True
    
    dx = x2 - x1
    dy = y2 - y1
    
    if abs(dx) > 1e-10:
        t = (xmin - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True
        t = (xmax - x1) / dx
        if 0 <= t <= 1:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True
    
    if abs(dy) > 1e-10:
        t = (ymin - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True
        t = (ymax - y1) / dy
        if 0 <= t <= 1:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True
    
    return False


def count_boxes_intersecting_curve(segments, delta, domain_x, domain_y):
    """Count boxes intersecting the curve."""
    xmin, xmax = domain_x
    ymin, ymax = domain_y
    
    nx = int(np.ceil((xmax - xmin) / delta))
    ny = int(np.ceil((ymax - ymin) / delta))
    
    intersecting_boxes = set()
    
    for segment in segments:
        for i in range(nx):
            for j in range(ny):
                box_min = np.array([xmin + i * delta, ymin + j * delta])
                box_max = np.array([xmin + (i + 1) * delta, ymin + (j + 1) * delta])
                
                if check_segment_box_intersection(segment, box_min, box_max):
                    intersecting_boxes.add((i, j))
    
    return len(intersecting_boxes)


def calculate_fractal_dimension(segments, domain_x, domain_y, 
                                initial_delta, delta_factor, num_steps):
    """
    Calculate fractal dimension for given parameters.
    
    Returns:
    --------
    dict with 'dimension', 'r_squared', 'std_err', and validity flag
    """
    # Generate delta values
    deltas = []
    delta = initial_delta
    for i in range(num_steps):
        deltas.append(delta)
        delta = delta / delta_factor
    
    deltas = np.array(deltas)
    
    # Check if smallest delta is reasonable
    domain_width = domain_x[1] - domain_x[0]
    domain_height = domain_y[1] - domain_y[0]
    min_domain = min(domain_width, domain_height)
    
    if deltas[-1] < min_domain / 1000:  # Too small
        return {'dimension': np.nan, 'r_squared': 0, 'std_err': np.inf, 
                'valid': False, 'reason': 'delta_too_small'}
    
    if deltas[0] > min_domain:  # Initial delta too large
        return {'dimension': np.nan, 'r_squared': 0, 'std_err': np.inf, 
                'valid': False, 'reason': 'initial_delta_too_large'}
    
    # Count boxes
    n_boxes = []
    for delta in deltas:
        n = count_boxes_intersecting_curve(segments, delta, domain_x, domain_y)
        if n == 0:
            return {'dimension': np.nan, 'r_squared': 0, 'std_err': np.inf, 
                    'valid': False, 'reason': 'no_boxes_counted'}
        n_boxes.append(n)
    
    n_boxes = np.array(n_boxes)
    
    # Check for sufficient variation
    if len(np.unique(n_boxes)) < 3:
        return {'dimension': np.nan, 'r_squared': 0, 'std_err': np.inf, 
                'valid': False, 'reason': 'insufficient_variation'}
    
    # Calculate logs
    log_inv_delta = np.log(1 / deltas)
    log_n_boxes = np.log(n_boxes)
    
    # Linear regression
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_inv_delta, log_n_boxes
        )
        r_squared = r_value ** 2
        
        return {
            'dimension': slope,
            'r_squared': r_squared,
            'std_err': std_err,
            'valid': True,
            'deltas': deltas,
            'n_boxes': n_boxes
        }
    except:
        return {'dimension': np.nan, 'r_squared': 0, 'std_err': np.inf, 
                'valid': False, 'reason': 'regression_failed'}


def objective_function(params, segments, domain_x, domain_y, target_dimension=1.0):
    """
    Objective function to minimize: deviation from target dimension.
    
    Parameters:
    -----------
    params : array [log(initial_delta), log(delta_factor), num_steps]
    """
    log_initial_delta, log_delta_factor, num_steps = params
    
    initial_delta = np.exp(log_initial_delta)
    delta_factor = np.exp(log_delta_factor)
    num_steps = int(np.round(num_steps))
    
    # Ensure reasonable bounds
    if num_steps < 5 or num_steps > 30:
        return 1e6
    
    result = calculate_fractal_dimension(
        segments, domain_x, domain_y, 
        initial_delta, delta_factor, num_steps
    )
    
    if not result['valid']:
        return 1e6
    
    # Objective: minimize deviation from target, penalize poor fits
    dimension_error = abs(result['dimension'] - target_dimension)
    r_squared_penalty = (1 - result['r_squared']) * 10  # Penalize poor fits
    
    return dimension_error + r_squared_penalty


def grid_search_optimizer(segments, domain_x, domain_y, target_dimension=1.0, 
                         verbose=True):
    """
    Grid search over parameter space.
    """
    domain_width = domain_x[1] - domain_x[0]
    domain_height = domain_y[1] - domain_y[0]
    min_domain = min(domain_width, domain_height)
    
    # Define parameter ranges
    initial_deltas = np.logspace(np.log10(min_domain/10), np.log10(min_domain/2), 8)
    delta_factors = np.linspace(1.3, 2.5, 10)
    num_steps_range = range(8, 21, 2)
    
    best_params = None
    best_score = np.inf
    best_result = None
    
    results_list = []
    
    total_combinations = len(initial_deltas) * len(delta_factors) * len(num_steps_range)
    count = 0
    
    if verbose:
        print(f"Testing {total_combinations} parameter combinations...")
    
    for initial_delta in initial_deltas:
        for delta_factor in delta_factors:
            for num_steps in num_steps_range:
                count += 1
                
                result = calculate_fractal_dimension(
                    segments, domain_x, domain_y,
                    initial_delta, delta_factor, num_steps
                )
                
                if result['valid']:
                    score = abs(result['dimension'] - target_dimension)
                    
                    results_list.append({
                        'initial_delta': initial_delta,
                        'delta_factor': delta_factor,
                        'num_steps': num_steps,
                        'dimension': result['dimension'],
                        'r_squared': result['r_squared'],
                        'std_err': result['std_err'],
                        'score': score
                    })
                    
                    if score < best_score:
                        best_score = score
                        best_params = (initial_delta, delta_factor, num_steps)
                        best_result = result
                
                if verbose and count % 100 == 0:
                    print(f"  Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)")
    
    if verbose:
        print(f"  Completed: {count}/{total_combinations} (100.0%)")
    
    return best_params, best_result, results_list


def optimize_parameters(filename, domain_x, domain_y, target_dimension=1.0,
                       method='grid', plot=True, save_plot=True, output_prefix=None):
    """
    Find optimal box-counting parameters.
    
    Parameters:
    -----------
    filename : str
        Path to segment file
    domain_x, domain_y : tuple
        Domain bounds
    target_dimension : float
        Target fractal dimension (1.0 for straight line)
    method : str
        'grid' for grid search, 'differential_evolution' for global optimization
    plot : bool
        Whether to visualize results
    save_plot : bool
        Whether to save the plot to file
    output_prefix : str or None
        Prefix for output files. If None, uses input filename
    """
    
    print("="*70)
    print("BOX-COUNTING PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"File: {filename}")
    print(f"Domain: X=[{domain_x[0]}, {domain_x[1]}], Y=[{domain_y[0]}, {domain_y[1]}]")
    print(f"Target dimension: {target_dimension}")
    print(f"Method: {method}")
    print("="*70 + "\n")
    
    # Load segments
    segments = parse_segment_file(filename)
    print(f"Loaded {len(segments)} segments\n")
    
    if method == 'grid':
        best_params, best_result, results_list = grid_search_optimizer(
            segments, domain_x, domain_y, target_dimension
        )
        
        initial_delta, delta_factor, num_steps = best_params
        
    elif method == 'differential_evolution':
        print("Running differential evolution optimization...")
        
        domain_width = domain_x[1] - domain_x[0]
        domain_height = domain_y[1] - domain_y[0]
        min_domain = min(domain_width, domain_height)
        
        # Bounds: [log(initial_delta), log(delta_factor), num_steps]
        bounds = [
            (np.log(min_domain/20), np.log(min_domain/2)),  # initial_delta
            (np.log(1.2), np.log(3.0)),                     # delta_factor
            (5, 25)                                          # num_steps
        ]
        
        result_opt = differential_evolution(
            objective_function,
            bounds,
            args=(segments, domain_x, domain_y, target_dimension),
            seed=42,
            maxiter=100,
            popsize=15,
            atol=1e-6,
            tol=1e-6
        )
        
        initial_delta = np.exp(result_opt.x[0])
        delta_factor = np.exp(result_opt.x[1])
        num_steps = int(np.round(result_opt.x[2]))
        
        best_result = calculate_fractal_dimension(
            segments, domain_x, domain_y,
            initial_delta, delta_factor, num_steps
        )
        
        results_list = []
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMAL PARAMETERS FOUND")
    print("="*70)
    print(f"Initial delta:    {initial_delta:.6f}")
    print(f"Delta factor:     {delta_factor:.6f}")
    print(f"Number of steps:  {num_steps}")
    print(f"Final delta:      {initial_delta / (delta_factor ** (num_steps - 1)):.6f}")
    print()
    print(f"Resulting dimension:  {best_result['dimension']:.8f}")
    print(f"Target dimension:     {target_dimension:.8f}")
    print(f"Error:                {abs(best_result['dimension'] - target_dimension):.8f}")
    print(f"R-squared:            {best_result['r_squared']:.10f}")
    print(f"Standard error:       {best_result['std_err']:.8f}")
    print("="*70)
    
    # Generate output filename prefix
    if output_prefix is None:
        import os
        output_prefix = os.path.splitext(filename)[0]
    
    # Save results to text file
    results_filename = f"{output_prefix}_optimization_results.txt"
    with open(results_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BOX-COUNTING PARAMETER OPTIMIZATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Input file: {filename}\n")
        f.write(f"Domain: X=[{domain_x[0]}, {domain_x[1]}], Y=[{domain_y[0]}, {domain_y[1]}]\n")
        f.write(f"Target dimension: {target_dimension}\n")
        f.write(f"Optimization method: {method}\n")
        f.write("\n")
        f.write("OPTIMAL PARAMETERS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Initial delta:    {initial_delta:.6f}\n")
        f.write(f"Delta factor:     {delta_factor:.6f}\n")
        f.write(f"Number of steps:  {num_steps}\n")
        f.write(f"Final delta:      {initial_delta / (delta_factor ** (num_steps - 1)):.6f}\n")
        f.write("\n")
        f.write("RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Resulting dimension:  {best_result['dimension']:.8f}\n")
        f.write(f"Target dimension:     {target_dimension:.8f}\n")
        f.write(f"Error:                {abs(best_result['dimension'] - target_dimension):.8f}\n")
        f.write(f"R-squared:            {best_result['r_squared']:.10f}\n")
        f.write(f"Standard error:       {best_result['std_err']:.8f}\n")
        f.write("="*70 + "\n")
        
        # Add command to reproduce
        f.write("\n")
        f.write("COMMAND TO USE THESE PARAMETERS:\n")
        f.write("-"*70 + "\n")
        f.write(f"python simple_analyzer.py {filename} \\\n")
        f.write(f"  --domain-x {domain_x[0]} {domain_x[1]} \\\n")
        f.write(f"  --domain-y {domain_y[0]} {domain_y[1]} \\\n")
        f.write(f"  --initial-delta {initial_delta:.6f} \\\n")
        f.write(f"  --delta-factor {delta_factor:.6f} \\\n")
        f.write(f"  --num-steps {num_steps}\n")
        
        # Add top results if available
        if method == 'grid' and results_list:
            f.write("\n")
            f.write("TOP 20 PARAMETER SETS:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Rank':<6} {'Init_δ':<12} {'Factor':<10} {'Steps':<8} {'Dimension':<12} {'R²':<12} {'Error':<10}\n")
            f.write("-"*70 + "\n")
            
            sorted_results = sorted(results_list, key=lambda x: x['score'])[:20]
            for i, r in enumerate(sorted_results):
                f.write(f"{i+1:<6} {r['initial_delta']:<12.6f} {r['delta_factor']:<10.4f} "
                       f"{r['num_steps']:<8} {r['dimension']:<12.8f} {r['r_squared']:<12.10f} "
                       f"{r['score']:<10.6f}\n")
    
    print(f"\nResults saved to: {results_filename}")
    
    # Visualization
    if plot and method == 'grid':
        fig = plt.figure(figsize=(16, 10))
        
        # Convert results to arrays for plotting
        results_array = np.array([
            [r['initial_delta'], r['delta_factor'], r['num_steps'], 
             r['dimension'], r['r_squared'], r['score']]
            for r in results_list
        ])
        
        # Plot 1: Dimension vs Initial Delta (for best delta_factor)
        ax1 = plt.subplot(2, 3, 1)
        for df in np.unique(results_array[:, 1]):
            mask = results_array[:, 1] == df
            data = results_array[mask]
            ax1.scatter(data[:, 0], data[:, 3], alpha=0.5, s=30, label=f'factor={df:.2f}')
        ax1.axhline(y=target_dimension, color='r', linestyle='--', linewidth=2, label='Target')
        ax1.set_xlabel('Initial Delta', fontweight='bold')
        ax1.set_ylabel('Dimension', fontweight='bold')
        ax1.set_title('Dimension vs Initial Delta')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, ncol=2)
        
        # Plot 2: Dimension vs Delta Factor
        ax2 = plt.subplot(2, 3, 2)
        for ns in np.unique(results_array[:, 2]):
            mask = results_array[:, 2] == ns
            data = results_array[mask]
            ax2.scatter(data[:, 1], data[:, 3], alpha=0.5, s=30, label=f'steps={int(ns)}')
        ax2.axhline(y=target_dimension, color='r', linestyle='--', linewidth=2, label='Target')
        ax2.set_xlabel('Delta Factor', fontweight='bold')
        ax2.set_ylabel('Dimension', fontweight='bold')
        ax2.set_title('Dimension vs Delta Factor')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, ncol=2)
        
        # Plot 3: Dimension vs Num Steps
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(results_array[:, 2], results_array[:, 3], alpha=0.5, s=30, c=results_array[:, 4], cmap='viridis')
        ax3.axhline(y=target_dimension, color='r', linestyle='--', linewidth=2, label='Target')
        ax3.set_xlabel('Number of Steps', fontweight='bold')
        ax3.set_ylabel('Dimension', fontweight='bold')
        ax3.set_title('Dimension vs Number of Steps')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: R² vs Dimension Error
        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(np.abs(results_array[:, 3] - target_dimension), results_array[:, 4], 
                   alpha=0.5, s=30, c=results_array[:, 2], cmap='plasma')
        ax4.set_xlabel('|Dimension - Target|', fontweight='bold')
        ax4.set_ylabel('R²', fontweight='bold')
        ax4.set_title('Fit Quality vs Accuracy')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Parameter space heatmap (initial_delta vs delta_factor)
        ax5 = plt.subplot(2, 3, 5)
        # Group by initial_delta and delta_factor, average over num_steps
        from scipy.interpolate import griddata
        points = results_array[:, :2]
        values = results_array[:, 5]  # score
        
        id_range = np.logspace(np.log10(results_array[:, 0].min()), 
                               np.log10(results_array[:, 0].max()), 50)
        df_range = np.linspace(results_array[:, 1].min(), results_array[:, 1].max(), 50)
        id_grid, df_grid = np.meshgrid(id_range, df_range)
        
        score_grid = griddata(points, values, (id_grid, df_grid), method='linear')
        
        im = ax5.contourf(id_grid, df_grid, score_grid, levels=20, cmap='RdYlGn_r')
        ax5.scatter(initial_delta, delta_factor, c='blue', s=200, marker='*', 
                   edgecolors='black', linewidths=2, label='Optimal', zorder=5)
        ax5.set_xlabel('Initial Delta', fontweight='bold')
        ax5.set_ylabel('Delta Factor', fontweight='bold')
        ax5.set_title('Parameter Space (Score)')
        ax5.set_xscale('log')
        plt.colorbar(im, ax=ax5, label='Score (lower=better)')
        ax5.legend()
        
        # Plot 6: Top 10 parameter sets
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Sort by score
        sorted_indices = np.argsort(results_array[:, 5])
        top_10 = results_array[sorted_indices[:10]]
        
        text = "Top 10 Parameter Sets:\n\n"
        text += f"{'Rank':<5} {'Init_δ':<10} {'Factor':<8} {'Steps':<6} {'Dim':<10} {'R²':<10}\n"
        text += "-" * 60 + "\n"
        
        for i, row in enumerate(top_10):
            text += f"{i+1:<5} {row[0]:<10.4f} {row[1]:<8.3f} {int(row[2]):<6} {row[3]:<10.6f} {row[4]:<10.8f}\n"
        
        ax6.text(0.1, 0.9, text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_plot:
            plot_filename = f"{output_prefix}_optimization_plots.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_filename}")
        
        plt.show()
    
    return {
        'initial_delta': initial_delta,
        'delta_factor': delta_factor,
        'num_steps': num_steps,
        'dimension': best_result['dimension'],
        'r_squared': best_result['r_squared'],
        'std_err': best_result['std_err'],
        'all_results': results_list if method == 'grid' else None
    }


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize box-counting parameters')
    parser.add_argument('filename', help='Input file with segment coordinates')
    parser.add_argument('--domain-x', type=float, nargs=2, required=True,
                       metavar=('XMIN', 'XMAX'), help='X domain bounds')
    parser.add_argument('--domain-y', type=float, nargs=2, required=True,
                       metavar=('YMIN', 'YMAX'), help='Y domain bounds')
    parser.add_argument('--target', type=float, default=1.0,
                       help='Target fractal dimension (default: 1.0)')
    parser.add_argument('--method', choices=['grid', 'differential_evolution'],
                       default='grid', help='Optimization method')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots/results to files')
    parser.add_argument('--output-prefix', type=str, default=None,
                       help='Prefix for output files (default: use input filename)')
    
    args = parser.parse_args()
    
    results = optimize_parameters(
        args.filename,
        domain_x=tuple(args.domain_x),
        domain_y=tuple(args.domain_y),
        target_dimension=args.target,
        method=args.method,
        plot=not args.no_plot,
        save_plot=not args.no_save,
        output_prefix=args.output_prefix
    )
