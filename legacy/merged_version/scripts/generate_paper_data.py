#!/usr/bin/env python3
"""
Generate all data needed for the fractal dimension paper.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from scipy import stats

# Import the fractal analyzer
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Import RT analyzer if available
try:
    from rt_analyzer import RTAnalyzer
    RT_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: RT Analyzer not available. RT analysis will be skipped.")
    RT_ANALYZER_AVAILABLE = False

# Base directory for all output
BASE_DIR = "./paper_data"

# Create a log file
LOG_FILE = "paper_data_generation.log"

def log_message(message):
    """Log a message to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, "a") as f:
        f.write(full_message + "\n")

def analyze_fractal_both_methods(fractal_type, output_dir, max_level=8):
    """
    Complete analysis for a single fractal using both basic box-counting 
    and sliding window approach for direct comparison.
    Optimized to minimize redundant calculations.
    """
    log_message(f"Starting analysis of {fractal_type} fractal (max level: {max_level})...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = FractalAnalyzer(fractal_type)
    analysis_tools = FractalAnalysisTools(analyzer)
    
    # Get theoretical dimension if available
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type)
    log_message(f"Theoretical dimension for {fractal_type}: {theoretical_dimension:.6f}")
    
    # Generate the fractal at specified level - only do this once
    log_message(f"Generating {fractal_type} fractal at level {max_level}...")
    _, segments = analyzer.generate_fractal(fractal_type, level=max_level)
    log_message(f"Generated fractal with {len(segments)} segments")
    
    # Calculate box counting data once and reuse it
    log_message("Performing box counting calculation (will be reused)...")
    box_sizes, box_counts, bounding_box = analyzer.box_counter.box_counting_optimized(
        segments, min_box_size=0.001, max_box_size=None, box_size_factor=1.5
    )
    
    # 1. Calculate fractal dimension using basic box-counting
    log_message("Calculating basic box-counting dimension...")
    basic_fd, basic_error, intercept = analyzer.box_counter.calculate_fractal_dimension(
        box_sizes, box_counts
    )
    
    log_message(f"Basic box-counting dimension: {basic_fd:.6f} ± {basic_error:.6f}")
    
    # Compute R² for basic method
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    _, _, r_value, _, _ = stats.linregress(log_sizes, log_counts)
    basic_r2 = r_value**2
    log_message(f"Basic box-counting R²: {basic_r2:.6f}")
    
    # Plot basic box-counting results
    analyzer.visualizer.plot_loglog(
        box_sizes, box_counts, basic_fd, basic_error, intercept,
        custom_filename=os.path.join(output_dir, f"{fractal_type}_basic_loglog.png")
    )
    
    # Visualize the fractal curve itself
    analyzer.visualizer.plot_fractal_curve(
        segments, bounding_box, plot_boxes=False,
        custom_filename=os.path.join(output_dir, f"{fractal_type}_curve_level_{max_level}.png"),
        level=max_level
    )
    
    # 2. Run sliding window optimal scaling region analysis
    # Pass the pre-computed box counting data to avoid redundant calculation
    log_message("Running sliding window optimal scaling region analysis...")
    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis_tools.analyze_linear_region(
        segments=segments, 
        box_sizes=box_sizes,
        box_counts=box_counts,
        bounding_box=bounding_box,
        fractal_type=fractal_type,
        plot_results=True,
        save_plots=True,
        output_dir=output_dir
    )
    
    log_message(f"Sliding window optimal dimension: {optimal_dimension:.6f} ± {errors[windows.index(optimal_window)]:.6f}")
    log_message(f"Optimal window: {optimal_window}, R²: {r_squared[windows.index(optimal_window)]:.6f}")
    
    # Create direct comparison plot between basic and sliding window
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot the box counting data
        plt.loglog(box_sizes, box_counts, 'bo', alpha=0.7, label='Box counting data')
        
        # Plot basic method fit
        log_sizes = np.log(box_sizes)
        basic_fit = np.exp(intercept + (-basic_fd) * log_sizes)
        plt.loglog(box_sizes, basic_fit, 'r-', linewidth=2, 
                 label=f'Basic method: D = {basic_fd:.6f} ± {basic_error:.6f}, R² = {basic_r2:.6f}')
        
        # Plot sliding window optimal fit
        # First get the optimal window range
        window_idx = windows.index(optimal_window)
        optimal_window_indices = None
        for start_idx in range(len(log_sizes) - optimal_window + 1):
            end_idx = start_idx + optimal_window
            window_log_sizes = log_sizes[start_idx:end_idx]
            window_log_counts = log_counts[start_idx:end_idx]
            slope, intercept_window, r_value, _, _ = stats.linregress(window_log_sizes, window_log_counts)
            r2 = r_value**2
            
            if abs(r2 - r_squared[window_idx]) < 0.001 and abs(-slope - dimensions[window_idx]) < 0.001:
                optimal_window_indices = (start_idx, end_idx)
                break
        
        if optimal_window_indices:
            start_idx, end_idx = optimal_window_indices
            # Highlight the optimal window range
            plt.loglog(box_sizes[start_idx:end_idx], box_counts[start_idx:end_idx], 'go', 
                     markersize=8, label='Optimal window range')
            
            # Plot sliding window fit
            window_log_sizes = log_sizes[start_idx:end_idx]
            window_log_counts = log_counts[start_idx:end_idx]
            slope, intercept_window, _, _, _ = stats.linregress(window_log_sizes, window_log_counts)
            window_fit = np.exp(intercept_window + slope * log_sizes)
            plt.loglog(box_sizes, window_fit, 'g--', linewidth=2, 
                     label=f'Sliding window: D = {-slope:.6f}, R² = {r_squared[window_idx]:.6f}')
        
        # Add theoretical dimension if available
        if theoretical_dimension:
            plt.axhline(y=0, color='k', linestyle='-', alpha=0)  # Invisible line for consistent legend
            plt.plot([], [], 'k--', label=f'Theoretical: D = {theoretical_dimension:.6f}')
            
            # Add relative errors as text
            basic_rel_error = abs(basic_fd - theoretical_dimension) / theoretical_dimension * 100
            window_rel_error = abs(optimal_dimension - theoretical_dimension) / theoretical_dimension * 100
            
            plt.figtext(0.5, 0.01, 
                      f"Relative errors - Basic: {basic_rel_error:.4f}%, Sliding Window: {window_rel_error:.4f}%", 
                      ha="center", fontsize=10, 
                      bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.xlabel('Box Size (r)')
        plt.ylabel('Box Count (N)')
        plt.title(f'{fractal_type.capitalize()} Fractal: Basic vs. Sliding Window Comparison (Level {max_level})')
        plt.grid(True, which='both', alpha=0.3)
        plt.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f"{fractal_type}_method_comparison.png"), dpi=300)
        plt.close()
        
        log_message(f"Created method comparison plot: {os.path.join(output_dir, f'{fractal_type}_method_comparison.png')}")
    except Exception as e:
        log_message(f"Error creating method comparison plot: {str(e)}")
    
    # 3. Run convergence analysis (iterations study) with both methods
    log_message(f"Running convergence analysis (iterations 1-{max_level}) with both methods...")
    
    # For storing results
    basic_results = {
        'levels': [],
        'dimensions': [],
        'errors': [],
        'r_squared': []
    }
    
    window_results = {
        'levels': [],
        'dimensions': [],
        'errors': [],
        'r_squared': [],
        'optimal_windows': []
    }
    
    # Analyze each level (minimize redundant calculations)
    for level in range(1, max_level + 1):
        try:
            log_message(f"Analyzing level {level}...")
            
            # Generate fractal at this level (cached if generated before)
            _, level_segments = analyzer.generate_fractal(fractal_type, level)
            
            # Run box counting once per level and share results
            level_box_sizes, level_box_counts, level_bbox = analyzer.box_counter.box_counting_optimized(
                level_segments, min_box_size=0.001, max_box_size=None, box_size_factor=1.5
            )
            
            # Basic box-counting (reuse the box counting results)
            basic_fd, basic_error, _ = analyzer.box_counter.calculate_fractal_dimension(
                level_box_sizes, level_box_counts
            )
            
            # Calculate R² for basic method
            level_log_sizes = np.log(level_box_sizes)
            level_log_counts = np.log(level_box_counts)
            _, _, r_value, _, _ = stats.linregress(level_log_sizes, level_log_counts)
            basic_r2 = r_value**2
            
            # Store basic results
            basic_results['levels'].append(level)
            basic_results['dimensions'].append(basic_fd)
            basic_results['errors'].append(basic_error)
            basic_results['r_squared'].append(basic_r2)
            
            # Sliding window analysis if enough data points
            if len(level_box_sizes) >= 5:  # Minimum needed for meaningful window analysis
                try:
                    # Use the pre-computed box counting data
                    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis_tools.analyze_linear_region(
                        segments=level_segments,
                        box_sizes=level_box_sizes,
                        box_counts=level_box_counts,
                        bounding_box=level_bbox,
                        fractal_type=fractal_type,
                        plot_results=False,  # Don't create plots for each level to save time
                        save_plots=False
                    )
                    
                    # Store sliding window results
                    window_results['levels'].append(level)
                    window_results['dimensions'].append(optimal_dimension)
                    window_results['errors'].append(errors[windows.index(optimal_window)])
                    window_results['r_squared'].append(r_squared[windows.index(optimal_window)])
                    window_results['optimal_windows'].append(optimal_window)
                except Exception as e:
                    log_message(f"Error in sliding window analysis for level {level}: {str(e)}")
                    # Use basic results as fallback if window analysis fails
                    window_results['levels'].append(level)
                    window_results['dimensions'].append(basic_fd)
                    window_results['errors'].append(basic_error)
                    window_results['r_squared'].append(basic_r2)
                    window_results['optimal_windows'].append(None)
            else:
                log_message(f"Not enough data points for sliding window analysis at level {level}")
                # Use basic results as fallback
                window_results['levels'].append(level)
                window_results['dimensions'].append(basic_fd)
                window_results['errors'].append(basic_error)
                window_results['r_squared'].append(basic_r2)
                window_results['optimal_windows'].append(None)
            
            log_message(f"Level {level} - Basic: D = {basic_fd:.6f} ± {basic_error:.6f}, "
                       f"Window: D = {window_results['dimensions'][-1]:.6f} ± {window_results['errors'][-1]:.6f}")
            
        except Exception as e:
            log_message(f"Error analyzing level {level}: {str(e)}")
    
    # Create combined convergence plot
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot basic results
        plt.errorbar(basic_results['levels'], basic_results['dimensions'], yerr=basic_results['errors'],
                   fmt='bo-', capsize=5, linewidth=1.5, label='Basic Box-Counting')
        
        # Plot sliding window results
        plt.errorbar(window_results['levels'], window_results['dimensions'], yerr=window_results['errors'],
                   fmt='rs-', capsize=5, linewidth=1.5, label='Sliding Window Approach')
        
        # Add theoretical dimension line if available
        if theoretical_dimension:
            plt.axhline(y=theoretical_dimension, color='k', linestyle='--', 
                      label=f'Theoretical Dimension ({theoretical_dimension:.6f})')
        
        # Add R² values as a second y-axis
        ax2 = plt.gca().twinx()
        ax2.plot(basic_results['levels'], basic_results['r_squared'], 'b--', marker='.', alpha=0.5, 
                label='Basic R²')
        ax2.plot(window_results['levels'], window_results['r_squared'], 'r--', marker='.', alpha=0.5,
                label='Window R²')
        ax2.set_ylabel('R²', color='gray')
        ax2.set_ylim([0.9, 1.01])
        
        # Add both legends
        lines1, labels1 = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.xlabel('Iteration Level')
        plt.ylabel('Fractal Dimension')
        plt.title(f'{fractal_type.capitalize()} Fractal: Dimension vs. Iteration Level Comparison')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fractal_type}_convergence_comparison.png"), dpi=300)
        plt.close()
        
        log_message(f"Created convergence comparison plot: {os.path.join(output_dir, f'{fractal_type}_convergence_comparison.png')}")
    except Exception as e:
        log_message(f"Error creating convergence comparison plot: {str(e)}")
    
    # Save iteration results to CSV
    try:
        # Combine both methods' results
        iterations_df = pd.DataFrame({
            'Level': basic_results['levels'],
            'Basic_Dimension': basic_results['dimensions'],
            'Basic_Error': basic_results['errors'],
            'Basic_R_squared': basic_results['r_squared'],
            'Window_Dimension': window_results['dimensions'],
            'Window_Error': window_results['errors'],
            'Window_R_squared': window_results['r_squared'],
            'Optimal_Window': window_results['optimal_windows']
        })
        
        # Add theoretical dimension and relative errors if available
        if theoretical_dimension:
            iterations_df['Theoretical'] = theoretical_dimension
            iterations_df['Basic_Rel_Error'] = abs(iterations_df['Basic_Dimension'] - theoretical_dimension) / theoretical_dimension * 100
            iterations_df['Window_Rel_Error'] = abs(iterations_df['Window_Dimension'] - theoretical_dimension) / theoretical_dimension * 100
        
        iterations_df.to_csv(os.path.join(output_dir, f"{fractal_type}_iterations_comparison.csv"), index=False)
        log_message(f"Saved iteration comparison results to {os.path.join(output_dir, f'{fractal_type}_iterations_comparison.csv')}")
    except Exception as e:
        log_message(f"Error saving iteration comparison results: {str(e)}")
    
    # Return the results
    return {
        'fractal_type': fractal_type,
        'theoretical_dimension': theoretical_dimension,
        'basic': {
            'dimension': basic_fd,
            'error': basic_error,
            'r_squared': basic_r2
        },
        'sliding_window': {
            'dimension': optimal_dimension,
            'error': errors[windows.index(optimal_window)],
            'r_squared': r_squared[windows.index(optimal_window)],
            'optimal_window': optimal_window
        },
        'convergence': {
            'basic': basic_results,
            'window': window_results
        }
    }

def analyze_parameter_sensitivity(fractal_type, output_dir):
    """Analyze sensitivity to different parameters with minimized redundant computation."""
    log_message(f"Starting parameter sensitivity analysis using {fractal_type} fractal...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer and tools
    analyzer = FractalAnalyzer(fractal_type)
    analysis_tools = FractalAnalysisTools(analyzer)
    
    # Generate a high-level fractal - do this only once
    level = 6  # Good balance between complexity and performance
    log_message(f"Generating {fractal_type} fractal at level {level} for sensitivity analysis...")
    _, segments = analyzer.generate_fractal(fractal_type, level=level)
    log_message(f"Generated fractal with {len(segments)} segments")
    
    # Get theoretical dimension
    theoretical_dimension = analyzer.base.THEORETICAL_DIMENSIONS.get(fractal_type)
    
    # Run box counting only once and reuse the data for all tests
    log_message("Performing box counting (will be reused for all parameter tests)...")
    box_sizes, box_counts, bounding_box = analyzer.box_counter.box_counting_optimized(
        segments, min_box_size=0.001, max_box_size=None, box_size_factor=1.5)
    
    # Convert to ln scale once for all analyses
    log_sizes = np.log(box_sizes)
    log_counts = np.log(box_counts)
    
    # 1. Analyze window width sensitivity
    log_message("Analyzing window width sensitivity...")
    window_results = []
    
    for width in [3, 5, 7, 9]:
        log_message(f"Testing window width = {width}")
        
        # Create directory for this width
        width_dir = os.path.join(output_dir, f"window_{width}")
        os.makedirs(width_dir, exist_ok=True)
        
        # For window sensitivity, use the same box counting data with different window widths
        best_r2 = -1
        best_dimension = None
        best_error = None
        best_start = None
        best_end = None
        
        # Try all possible starting points for this window size
        if width < len(log_sizes):
            for start_idx in range(len(log_sizes) - width + 1):
                end_idx = start_idx + width
                
                # Perform regression on this window
                window_log_sizes = log_sizes[start_idx:end_idx]
                window_log_counts = log_counts[start_idx:end_idx]
                slope, intercept, r_value, p_value, std_err = stats.linregress(window_log_sizes, window_log_counts)
                dimension = -slope
                
                # Store if this is the best fit for this window size
                if r_value**2 > best_r2:
                    best_r2 = r_value**2
                    best_dimension = dimension
                    best_error = std_err
                    best_start = start_idx
                    best_end = end_idx
            
            # Store the results
            window_results.append({
                'window_width': width,
                'dimension': best_dimension,
                'error': best_error,
                'r_squared': best_r2,
                'start_idx': best_start,
                'end_idx': best_end,
                'diff_from_theoretical': abs(best_dimension - theoretical_dimension) if theoretical_dimension else None
            })
            
            # Visualize the result for this window width
            if best_start is not None and best_end is not None:
                try:
                    # Create log-log plot with this window
                    plt.figure(figsize=(10, 8))
                    
                    # Plot all points
                    plt.loglog(box_sizes, box_counts, 'bo', alpha=0.5, label='All data points')
                    
                    # Highlight selected window
                    plt.loglog(box_sizes[best_start:best_end], box_counts[best_start:best_end], 
                             'ro', markersize=8, label=f'Window width = {width}')
                    
                    # Add fit line
                    window_fit = np.exp(intercept + (-best_dimension) * log_sizes)
                    plt.loglog(box_sizes, window_fit, 'r-', 
                             label=f'Fit: D = {best_dimension:.4f} ± {best_error:.4f}, R² = {best_r2:.4f}')
                    
                    plt.xlabel('Box Size (ε)')
                    plt.ylabel('Box Count N(ε)')
                    plt.title(f'Window Width = {width}: Optimal Region')
                    plt.grid(True, which='both', alpha=0.4)
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(width_dir, f"window_{width}_loglog.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    log_message(f"Error creating visualization for window width {width}: {str(e)}")
            
            log_message(f"Window width {width}: dimension = {best_dimension:.6f} ± {best_error:.6f}, R² = {best_r2:.6f}")
    
    # Save window width results to CSV
    window_df = pd.DataFrame(window_results)
    window_df.to_csv(os.path.join(output_dir, "window_width_sensitivity.csv"), index=False)
    
    # 2. Analyze threshold sensitivity
    log_message("Analyzing threshold sensitivity...")
    threshold_results = []
    
    for threshold in [0.03, 0.05, 0.08, 0.10]:
        log_message(f"Testing threshold = {threshold}")
        
        threshold_dir = os.path.join(output_dir, f"threshold_{threshold:.2f}")
        os.makedirs(threshold_dir, exist_ok=True)
        
        try:
            # Run linear region analysis with this threshold, reusing box counting data
            windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis_tools.analyze_linear_region(
                segments=None,  # Pass None since we're providing box counts directly 
                box_sizes=box_sizes,
                box_counts=box_counts,
                bounding_box=bounding_box,
                fractal_type=fractal_type,
                plot_results=True,
                save_plots=True,
                output_dir=threshold_dir,
                sigma_thresh=threshold  # Pass the threshold parameter
            )
            
            # Store the results
            threshold_results.append({
                'threshold': threshold,
                'optimal_window': optimal_window,
                'dimension': optimal_dimension,
                'error': errors[windows.index(optimal_window)],
                'r_squared': r_squared[windows.index(optimal_window)],
                'diff_from_theoretical': abs(optimal_dimension - theoretical_dimension) if theoretical_dimension else None
            })
            
            log_message(f"Threshold {threshold}: optimal window = {optimal_window}, "
                       f"dimension = {optimal_dimension:.6f} ± {errors[windows.index(optimal_window)]:.6f}, "
                       f"R² = {r_squared[windows.index(optimal_window)]:.6f}")
        except Exception as e:
            log_message(f"Error analyzing threshold {threshold}: {str(e)}")
    
    # Save threshold results to CSV
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv(os.path.join(output_dir, "threshold_sensitivity.csv"), index=False)
    
    # 3. Analyze minimum region length
    log_message("Analyzing minimum region length sensitivity...")
    length_results = []
    
    for min_length in [3, 5, 7, 9]:
        log_message(f"Testing min_length = {min_length}")
        
        length_dir = os.path.join(output_dir, f"min_length_{min_length}")
        os.makedirs(length_dir, exist_ok=True)
        
        try:
            # Run linear region analysis with this minimum length, reusing box counting data
            windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis_tools.analyze_linear_region(
                segments=None,  # Pass None since we're providing box counts directly
                box_sizes=box_sizes,
                box_counts=box_counts,
                bounding_box=bounding_box,
                fractal_type=fractal_type,
                plot_results=True,
                save_plots=True,
                output_dir=length_dir,
                L_min=min_length  # Pass the minimum length parameter
            )
            
            # Store the results
            length_results.append({
                'min_length': min_length,
                'optimal_window': optimal_window,
                'dimension': optimal_dimension,
                'error': errors[windows.index(optimal_window)],
                'r_squared': r_squared[windows.index(optimal_window)],
                'diff_from_theoretical': abs(optimal_dimension - theoretical_dimension) if theoretical_dimension else None
            })
            
            log_message(f"Min length {min_length}: optimal window = {optimal_window}, "
                       f"dimension = {optimal_dimension:.6f} ± {errors[windows.index(optimal_window)]:.6f}, "
                       f"R² = {r_squared[windows.index(optimal_window)]:.6f}")
        except Exception as e:
            log_message(f"Error analyzing min_length {min_length}: {str(e)}")
    
    # Save minimum length results to CSV
    length_df = pd.DataFrame(length_results)
    length_df.to_csv(os.path.join(output_dir, "min_length_sensitivity.csv"), index=False)
    
    # 4. Create combined plot of all sensitivity results
    try:
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Window width sensitivity
        if window_results:
            widths = [r['window_width'] for r in window_results]
            dims = [r['dimension'] for r in window_results]
            errs = [r['error'] for r in window_results]
            
            axs[0].errorbar(widths, dims, yerr=errs, fmt='o-', capsize=5)
            if theoretical_dimension:
                axs[0].axhline(y=theoretical_dimension, color='r', linestyle='--', 
                             label=f'Theoretical: {theoretical_dimension:.4f}')
            axs[0].set_xlabel('Window Width')
            axs[0].set_ylabel('Fractal Dimension')
            axs[0].set_title('Window Width Sensitivity')
            axs[0].grid(True)
            axs[0].legend()
        
        # Threshold sensitivity
        if threshold_results:
            thresholds = [r['threshold'] for r in threshold_results]
            dims = [r['dimension'] for r in threshold_results]
            errs = [r['error'] for r in threshold_results]
            
            axs[1].errorbar(thresholds, dims, yerr=errs, fmt='o-', capsize=5)
            if theoretical_dimension:
                axs[1].axhline(y=theoretical_dimension, color='r', linestyle='--', 
                             label=f'Theoretical: {theoretical_dimension:.4f}')
            axs[1].set_xlabel('Threshold')
            axs[1].set_ylabel('Fractal Dimension')
            axs[1].set_title('Threshold Sensitivity')
            axs[1].grid(True)
            axs[1].legend()
        
        # Minimum length sensitivity
        if length_results:
            lengths = [r['min_length'] for r in length_results]
            dims = [r['dimension'] for r in length_results]
            errs = [r['error'] for r in length_results]
            
            axs[2].errorbar(lengths, dims, yerr=errs, fmt='o-', capsize=5)
            if theoretical_dimension:
                axs[2].axhline(y=theoretical_dimension, color='r', linestyle='--', 
                             label=f'Theoretical: {theoretical_dimension:.4f}')
            axs[2].set_xlabel('Minimum Region Length')
            axs[2].set_ylabel('Fractal Dimension')
            axs[2].set_title('Minimum Region Length Sensitivity')
            axs[2].grid(True)
            axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_sensitivity_combined.png"), dpi=300)
        plt.close()
        
        log_message("Created combined sensitivity analysis plot")
    except Exception as e:
        log_message(f"Error creating combined sensitivity plot: {str(e)}")
    
    return {
        'window_results': window_results,
        'threshold_results': threshold_results,
        'length_results': length_results
    }

def analyze_rt_simulation(data_dir, output_dir):
    """Analyze RT simulation data with optimized computation."""
    log_message("Starting RT simulation analysis...")
    
    if not RT_ANALYZER_AVAILABLE:
        log_message("RT Analyzer not available. Skipping RT analysis.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = RTAnalyzer(output_dir)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        log_message(f"Warning: RT data directory {data_dir} does not exist.")
        return None
    
    # Find VTK files pattern
    pattern = os.path.join(data_dir, "RT*-*.vtk")
    vtk_files = sorted(glob.glob(pattern))
    
    if not vtk_files:
        log_message(f"Warning: No VTK files found matching pattern: {pattern}")
        return None
    
    log_message(f"Found {len(vtk_files)} VTK files for analysis.")
    
    # Determine resolution from filename
    resolution = None
    if vtk_files:
        # Extract resolution from filename (assuming format RTXXXxXXX-...)
        import re
        match = re.search(r'RT(\d+)x\d+-', os.path.basename(vtk_files[0]))
        if match:
            resolution = int(match.group(1))
            log_message(f"Detected resolution: {resolution}x{resolution}")
    
    # Create a data cache to avoid re-reading files
    vtk_data_cache = {}
    interface_cache = {}
    
    # Process a series of VTK files to analyze temporal evolution
    log_message("Processing VTK series for temporal evolution...")
    try:
        results = analyzer.process_vtk_series(
            pattern, 
            resolution=resolution, 
            analyze_linear=True
        )
        log_message("VTK series analysis complete.")
        
        # Create visualization of the temporal evolution
        log_message("Creating summary plots for RT evolution...")
        analyzer.create_summary_plots(results, os.path.join(output_dir, "evolution"))
        log_message("Summary plots created.")
    except Exception as e:
        log_message(f"Error in RT series analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        results = None
    
    # For selected time points, perform detailed analysis
    time_points = [1.0, 3.0, 5.0, 7.0, 9.0]
    snapshot_results = []
    
    log_message(f"Analyzing specific time points: {time_points}")
    
    # First, find files matching the time points to reduce file scanning
    time_point_files = {}
    for time in time_points:
        # Check multiple possible filename patterns
        for vtk_file in vtk_files:
            # Extract time from filename
            time_match = re.search(r'(\d+)\.vtk$', os.path.basename(vtk_file))
            if time_match:
                file_time = float(time_match.group(1))/1000.0
                if abs(file_time - time) < 0.05:  # Allow for small time differences
                    time_point_files[time] = vtk_file
                    break
    
    # Now analyze the found files
    for time in time_points:
        vtk_file = time_point_files.get(time)
        
        if vtk_file:
            log_message(f"Analyzing time point t={time} using file: {vtk_file}")
            try:
                # Check if we've already analyzed this file
                if vtk_file in vtk_data_cache:
                    log_message(f"Using cached data for {vtk_file}")
                    result = vtk_data_cache[vtk_file]
                else:
                    # Analyze the file and store in cache
                    result = analyzer.analyze_vtk_file(
                        vtk_file, 
                        f"snapshot_t{time}", 
                        analyze_linear=True
                    )
                    vtk_data_cache[vtk_file] = result
                
                snapshot_results.append(result)
                log_message(f"Time point t={time} analysis complete. "
                           f"Dimension: {result['fractal_dim']:.6f} ± {result['fd_error']:.6f}")
            except Exception as e:
                log_message(f"Error analyzing time point t={time}: {str(e)}")
        else:
            log_message(f"No file found for time point t={time}")
    
    # Save snapshot results to CSV
    if snapshot_results:
        snapshot_df = pd.DataFrame(snapshot_results)
        snapshot_df.to_csv(os.path.join(output_dir, "rt_snapshots.csv"), index=False)
        log_message(f"Saved snapshot results to {os.path.join(output_dir, 'rt_snapshots.csv')}")
    
    # Return combined results
    return {
        'temporal_results': results,
        'snapshot_results': snapshot_results,
        'output_dir': output_dir
    }

def create_paper_figures(fractal_results, sensitivity_results, rt_results, output_dir):
    """Create final figures for the paper."""
    log_message("Creating final figures for the paper...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create comparison table for all fractals, comparing both methods
    try:
        log_message("Creating fractal comparison table...")
        
        # Extract data
        data = []
        for ftype, result in fractal_results.items():
            data.append({
                'Fractal Type': ftype.capitalize(),
                'Theoretical': result['theoretical_dimension'],
                'Basic_Dimension': result['basic']['dimension'],
                'Basic_Error': result['basic']['error'],
                'Basic_R²': result['basic']['r_squared'],
                'Window_Dimension': result['sliding_window']['dimension'],
                'Window_Error': result['sliding_window']['error'],
                'Window_R²': result['sliding_window']['r_squared'],
                'Optimal_Window': result['sliding_window']['optimal_window']
            })
        
        # Add relative errors if theoretical dimension is available
        for item in data:
            if item['Theoretical']:
                item['Basic_Rel_Error'] = abs(item['Basic_Dimension'] - item['Theoretical']) / item['Theoretical'] * 100
                item['Window_Rel_Error'] = abs(item['Window_Dimension'] - item['Theoretical']) / item['Theoretical'] * 100
        
        # Create DataFrame and save to CSV
        comparison_df = pd.DataFrame(data)
        comparison_df.to_csv(os.path.join(output_dir, "fractal_method_comparison.csv"), index=False)
        log_message(f"Saved fractal method comparison to {os.path.join(output_dir, 'fractal_method_comparison.csv')}")
        
        # Create LaTeX formatted table
        try:
            latex_table = comparison_df.to_latex(index=False, float_format="%.6f")
            with open(os.path.join(output_dir, "fractal_method_comparison.tex"), "w") as f:
                f.write(latex_table)
            log_message(f"Saved LaTeX table to {os.path.join(output_dir, 'fractal_method_comparison.tex')}")
        except Exception as e:
            log_message(f"Error creating LaTeX table: {str(e)}")
    except Exception as e:
        log_message(f"Error creating comparison table: {str(e)}")
    
    # 2. Create dimension vs level plot for all fractals
    try:
        log_message("Creating dimension vs level plot for all fractals...")
        
        # Two separate plots for basic and window methods
        for method in ["basic", "window"]:
            plt.figure(figsize=(12, 8))
            
            # Colors and markers for different fractals
            colors = ['b', 'g', 'r', 'c', 'm']
            markers = ['o', 's', '^', 'D', 'v']
            
            for i, (ftype, result) in enumerate(fractal_results.items()):
                # Get convergence data for this method
                if method == "basic":
                    levels = result['convergence']['basic']['levels']
                    dimensions = result['convergence']['basic']['dimensions']
                    errors = result['convergence']['basic']['errors']
                    method_name = "Basic Box-Counting"
                else:
                    levels = result['convergence']['window']['levels']
                    dimensions = result['convergence']['window']['dimensions']
                    errors = result['convergence']['window']['errors']
                    method_name = "Sliding Window"
                
                if levels:
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    
                    plt.errorbar(
                        levels, 
                        dimensions, 
                        yerr=errors, 
                        fmt=f'{color}{marker}-', 
                        capsize=5,
                        label=f"{ftype.capitalize()}"
                    )
                    
                    # Add theoretical dimension line if available
                    if result['theoretical_dimension']:
                        plt.axhline(
                            y=result['theoretical_dimension'], 
                            color=color, 
                            linestyle='--', 
                            alpha=0.5
                        )
            
            plt.xlabel('Iteration Level')
            plt.ylabel('Fractal Dimension')
            plt.title(f'Fractal Dimension vs. Iteration Level ({method_name})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f"dimension_vs_level_{method}.png"), dpi=300)
            plt.close()
            
            log_message(f"Saved dimension vs level plot ({method}) to {os.path.join(output_dir, f'dimension_vs_level_{method}.png')}")
        
        # Create a combined plot showing improvement percentage
        plt.figure(figsize=(12, 8))
        
        for i, (ftype, result) in enumerate(fractal_results.items()):
            # Check if we have both methods and theoretical value
            if (result['convergence']['basic']['levels'] and 
                result['convergence']['window']['levels'] and 
                result['theoretical_dimension']):
                
                # Calculate improvement percentage
                basic_levels = result['convergence']['basic']['levels']
                basic_dims = result['convergence']['basic']['dimensions']
                window_levels = result['convergence']['window']['levels']
                window_dims = result['convergence']['window']['dimensions']
                theoretical = result['theoretical_dimension']
                
                # Find matching levels
                common_levels = []
                improvements = []
                
                for level in set(basic_levels).intersection(window_levels):
                    basic_idx = basic_levels.index(level)
                    window_idx = window_levels.index(level)
                    
                    basic_err = abs(basic_dims[basic_idx] - theoretical) / theoretical * 100
                    window_err = abs(window_dims[window_idx] - theoretical) / theoretical * 100
                    
                    if basic_err > 0:  # Avoid division by zero
                        improvement = (basic_err - window_err) / basic_err * 100
                        common_levels.append(level)
                        improvements.append(improvement)
                
                if common_levels:
                    color = colors[i % len(colors)]
                    marker = markers[i % len(markers)]
                    
                    plt.plot(
                        common_levels, 
                        improvements, 
                        f'{color}{marker}-', 
                        label=f"{ftype.capitalize()}"
                    )
        
        plt.xlabel('Iteration Level')
        plt.ylabel('Improvement %')
        plt.title('Accuracy Improvement: Sliding Window vs Basic Method')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "method_improvement.png"), dpi=300)
        plt.close()
        
        log_message(f"Saved method improvement plot to {os.path.join(output_dir, 'method_improvement.png')}")
        
    except Exception as e:
        log_message(f"Error creating dimension vs level plot: {str(e)}")
    
    # 3. Create RT fractal dimension evolution plot (if available)
    if rt_results and rt_results.get('temporal_results') is not None:
        try:
            log_message("Creating RT fractal dimension evolution plot...")
            
            # Copy RT evolution plots to the paper figures directory
            import shutil
            rt_evolution_dir = os.path.join(rt_results['output_dir'], "evolution")
            if os.path.exists(rt_evolution_dir):
                for filename in os.listdir(rt_evolution_dir):
                    if filename.endswith(".png"):
                        src = os.path.join(rt_evolution_dir, filename)
                        dst = os.path.join(output_dir, f"rt_{filename}")
                        shutil.copy(src, dst)
                        log_message(f"Copied {src} to {dst}")
        except Exception as e:
            log_message(f"Error processing RT evolution plots: {str(e)}")
    
    log_message("Paper figures creation complete.")

def create_paper_box_visualization(fractal_type, output_dir, level=5):
    """Create a simple box counting visualization for the paper with optimized computation."""
    log_message(f"Creating paper box visualization for {fractal_type} at level {level}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = FractalAnalyzer(fractal_type)
    
    # Generate the fractal - this will use the cache if available
    _, segments = analyzer.generate_fractal(fractal_type, level=level)
    
    # Find the bounding box - do this once and reuse
    min_x = min(min(s[0][0], s[1][0]) for s in segments)
    max_x = max(max(s[0][0], s[1][0]) for s in segments)
    min_y = min(min(s[0][1], s[1][1]) for s in segments)
    max_y = max(max(s[0][1], s[1][1]) for s in segments)
    
    # Add a small margin
    margin = max(max_x - min_x, max_y - min_y) * 0.05
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # Create figure for the fractal curve
    plt.figure(figsize=(8, 8))
    
    # Plot the fractal curve
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.axis('equal')
    plt.title(f'{fractal_type.capitalize()} Curve (Level {level})')
    
    # Save the fractal curve
    curve_file = os.path.join(output_dir, f"{fractal_type}_curve_level_{level}.png")
    plt.savefig(curve_file, dpi=300)
    plt.close()
    
    # Now create a figure with box overlay
    # First run box counting to get good box sizes - this will use the cache if available
    box_sizes, box_counts, bounding_box = analyzer.box_counter.box_counting_optimized(
        segments, min_box_size=0.001, max_box_size=None, box_size_factor=1.5)
    
    # Choose a good box size for visualization (not too small, not too large)
    # Try a box size around 1/4 of the way through the list (smaller boxes)
    box_size_idx = len(box_sizes) * 3 // 4
    box_size = box_sizes[box_size_idx]
    
    # Create new figure for box overlay
    plt.figure(figsize=(8, 8))
    
    # Plot the fractal again
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
    
    # Draw boxes directly - use a function to avoid code duplication
    def draw_boxes(ax, min_x, min_y, max_x, max_y, box_size, segments, zoom_region=None):
        """Draw boxes that intersect the fractal and optionally restrict to a zoom region."""
        num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
        
        count = 0
        box_positions = []
        
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
                
                # If zoom region specified, check if box is in the region
                if zoom_region:
                    zoom_min_x, zoom_max_x, zoom_min_y, zoom_max_y = zoom_region
                    if (box_xmax < zoom_min_x or box_xmin > zoom_max_x or 
                        box_ymax < zoom_min_y or box_ymin > zoom_max_y):
                        continue
                
                # Check if any segment intersects this box
                for (x1, y1), (x2, y2) in segments:
                    if analyzer.base.liang_barsky_line_box_intersection(
                            x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        # Store box position
                        box_positions.append((box_xmin, box_ymin))
                        count += 1
                        break
        
        # Draw all boxes at once for improved performance
        for box_xmin, box_ymin in box_positions:
            rect = plt.Rectangle((box_xmin, box_ymin), box_size, box_size,
                               facecolor='none', edgecolor='r', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
        
        return count
    
    # Draw boxes for the full view
    count = draw_boxes(plt.gca(), min_x, min_y, max_x, max_y, box_size, segments)
    log_message(f"Drew {count} boxes")
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.axis('equal')
    plt.title(f'{fractal_type.capitalize()} with Box Counting (Box Size: {box_size:.6f})')
    
    # Save the box overlay
    box_file = os.path.join(output_dir, f"{fractal_type}_box_overlay.png")
    plt.savefig(box_file, dpi=300)
    plt.close()
    
    # Create a zoomed view of a section
    plt.figure(figsize=(8, 8))
    
    # Plot the fractal again
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.5)
    
    # Calculate zoom region (e.g., top-left quarter)
    zoom_width = (max_x - min_x) / 3
    zoom_height = (max_y - min_y) / 3
    zoom_min_x = min_x
    zoom_max_x = min_x + zoom_width
    zoom_min_y = min_y
    zoom_max_y = min_y + zoom_height
    
    # Adjust for different fractals
    if fractal_type == 'koch':
        # For Koch curve, zoom to a specific area
        zoom_min_x = min_x + (max_x - min_x) * 0.3
        zoom_max_x = min_x + (max_x - min_x) * 0.7
        zoom_min_y = min_y
        zoom_max_y = min_y + (max_y - min_y) * 0.4
    
    # Draw boxes in the zoom region
    zoom_region = (zoom_min_x, zoom_max_x, zoom_min_y, zoom_max_y)
    zoom_count = draw_boxes(plt.gca(), min_x, min_y, max_x, max_y, box_size, segments, zoom_region)
    log_message(f"Drew {zoom_count} boxes in zoom region")
    
    plt.xlim(zoom_min_x, zoom_max_x)
    plt.ylim(zoom_min_y, zoom_max_y)
    plt.axis('equal')
    plt.title(f'{fractal_type.capitalize()} Box Counting (Zoomed View)')
    
    # Save the zoomed view
    zoom_file = os.path.join(output_dir, f"{fractal_type}_box_overlay_zoom.png")
    plt.savefig(zoom_file, dpi=300)
    plt.close()
    
    log_message(f"Created box visualizations for {fractal_type}")
    
    return curve_file, box_file, zoom_file

def main():
    """Main function to run all analyses with optimized computation."""
    # Clear log file
    with open(LOG_FILE, "w") as f:
        f.write("")
    
    log_message("Starting paper data generation...")
    
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Print cache statistics helper function
    def print_cache_stats():
        """Print statistics about cache usage."""
        try:
            # Get BoxCounter cache stats
            analyzer = FractalAnalyzer('koch')  # Use any fractal type
            box_stats = analyzer.box_counter.cache.stats()
            
            # Get fractal generator cache stats
            gen_stats = analyzer.generator_cache.stats()
            
            log_message("\n===== CACHE STATISTICS =====")
            log_message(f"Box counting cache: {box_stats['hits']} hits, {box_stats['misses']} misses, " 
                       f"{box_stats['entries']} entries, {box_stats['hit_rate']*100:.1f}% hit rate")
            log_message(f"Fractal generator cache: {gen_stats['hits']} hits, {gen_stats['misses']} misses, "
                       f"{gen_stats['entries']} entries, {gen_stats['hit_rate']*100:.1f}% hit rate")
        except Exception as e:
            log_message(f"Could not get cache statistics: {str(e)}")
    
    # 1. Analyze all mathematical fractals with both methods
    fractal_types = ['koch', 'sierpinski', 'hilbert', 'dragon', 'minkowski']
    fractal_results = {}
    
    for fractal_type in fractal_types:
        try:
            log_message(f"\n{'='*80}\nAnalyzing {fractal_type} fractal with both methods\n{'='*80}")
            # Different max levels based on fractal complexity
            max_level = 6 if fractal_type == 'hilbert' else 8
            
            result = analyze_fractal_both_methods(
                fractal_type,
                os.path.join(BASE_DIR, "mathematical_fractals", fractal_type),
                max_level=max_level
            )
            fractal_results[fractal_type] = result
        except Exception as e:
            log_message(f"Error analyzing {fractal_type} fractal: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print cache statistics after fractal analysis
    print_cache_stats()
    
    # 2. Analyze parameter sensitivity (using Koch curve)
    try:
        log_message(f"\n{'='*80}\nAnalyzing parameter sensitivity\n{'='*80}")
        sensitivity_results = analyze_parameter_sensitivity(
            'koch',
            os.path.join(BASE_DIR, "parameter_sensitivity")
        )
    except Exception as e:
        log_message(f"Error in parameter sensitivity analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sensitivity_results = None
    
    # Print cache statistics after sensitivity analysis
    print_cache_stats()
    
    # 3. Analyze RT simulation (if available)
    rt_results = None
    if RT_ANALYZER_AVAILABLE:
        try:
            log_message(f"\n{'='*80}\nAnalyzing RT simulation\n{'='*80}")
            
            # Prompt for RT data directory
            rt_data_dir = input("Enter path to RT simulation data directory (or press Enter to skip): ").strip()
            
            if rt_data_dir:
                rt_results = analyze_rt_simulation(
                    rt_data_dir,
                    os.path.join(BASE_DIR, "rt_analysis")
                )
        except Exception as e:
            log_message(f"Error in RT simulation analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 4. Create final figures for the paper
    try:
        log_message(f"\n{'='*80}\nCreating paper figures\n{'='*80}")
        create_paper_figures(
            fractal_results,
            sensitivity_results,
            rt_results,
            os.path.join(BASE_DIR, "figures")
        )
    except Exception as e:
        log_message(f"Error creating paper figures: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 5. Create box visualization for the paper - reuse cache for efficiency
    log_message(f"\n{'='*80}\nCreating box visualizations for paper\n{'='*80}")
    for fractal_type in fractal_types:
        try:
            curve_file, box_file, zoom_file = create_paper_box_visualization(
                fractal_type,
                os.path.join(BASE_DIR, "figures"),
                level=5 if fractal_type == 'hilbert' else 6
            )
        except Exception as e:
            log_message(f"Error creating box visualization for {fractal_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final cache statistics
    print_cache_stats()
    
    log_message("\nPaper data generation complete!")
    log_message(f"All results are saved in: {os.path.abspath(BASE_DIR)}")

if __name__ == "__main__":
    main()
