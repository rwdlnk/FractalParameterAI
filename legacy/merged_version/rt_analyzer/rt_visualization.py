# rt_visualization.py
"""Visualization utilities for RT analysis results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D

def plot_temporal_evolution(df, resolution, output_dir):
    """Create plots for a single resolution's temporal evolution."""
    # Plot fractal dimension vs time
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                fmt='o-', capsize=3, linewidth=2, markersize=5)
    plt.xlabel('Time')
    plt.ylabel('Fractal Dimension')
    plt.title(f'Fractal Dimension Evolution ({resolution}x{resolution})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'dimension_evolution.png'), dpi=300)
    plt.close()
    
    # Plot mixing layer thickness vs time
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'], df['h_total'], 'b-', label='Total', linewidth=2)
    plt.plot(df['time'], df['ht'], 'r--', label='Upper', linewidth=2)
    plt.plot(df['time'], df['hb'], 'g--', label='Lower', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Mixing Layer Thickness')
    plt.title(f'Mixing Layer Evolution ({resolution}x{resolution})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'mixing_evolution.png'), dpi=300)
    plt.close()
    
    # Combined plot of fractal dimension and mixing thickness
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Fractal dimension on left axis
    ax1.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
               fmt='bo-', capsize=3, label='Fractal Dimension')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Fractal Dimension', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Mixing layer on right axis
    ax2 = ax1.twinx()
    ax2.plot(df['time'], df['h_total'], 'r-', label='Mixing Thickness', linewidth=2)
    ax2.set_ylabel('Mixing Layer Thickness', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add both legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Fractal Dimension and Mixing Layer Evolution ({resolution}x{resolution})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'combined_evolution.png'), dpi=300)
    plt.close()

def plot_multi_resolution_evolution(all_results, resolutions, output_dir):
    """Create plots comparing temporal evolution across multiple resolutions."""
    # Create output directory
    multi_res_dir = os.path.join(output_dir, 'multi_resolution')
    os.makedirs(multi_res_dir, exist_ok=True)
    
    # Plot fractal dimension evolution
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.errorbar(df['time'], df['fractal_dim'], yerr=df['fd_error'],
                   fmt=f'{color}{marker}-', capsize=3, linewidth=1.5, 
                   label=f'{resolution}x{resolution}')
    
    plt.xlabel('Time')
    plt.ylabel('Fractal Dimension')
    plt.title('Fractal Dimension Evolution Across Resolutions')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(multi_res_dir, 'dimension_evolution_comparison.png'), dpi=300)
    plt.close()
    
    # Plot mixing layer evolution
    plt.figure(figsize=(12, 8))
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        
        plt.plot(df['time'], df['h_total'], f'{color}-', linewidth=2, 
                label=f'{resolution}x{resolution}')
    
    plt.xlabel('Time')
    plt.ylabel('Mixing Layer Thickness')
    plt.title('Mixing Layer Evolution Across Resolutions')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(multi_res_dir, 'mixing_evolution_comparison.png'), dpi=300)
    plt.close()
    
    # Phase plot: Fractal dimension vs. Mixing layer thickness
    plt.figure(figsize=(12, 8))
    
    for i, resolution in enumerate(resolutions):
        if resolution not in all_results:
            continue
        
        df = all_results[resolution]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        plt.plot(df['h_total'], df['fractal_dim'], f'{color}{marker}-', linewidth=1.5, 
               label=f'{resolution}x{resolution}')
        
        # Add time labels to selected points
        if len(df) > 4:
            # Add labels every nth point
            n = max(1, len(df) // 5)
            for j in range(0, len(df), n):
                time = df['time'].iloc[j]
                plt.annotate(f't={time:.1f}', 
                            (df['h_total'].iloc[j], df['fractal_dim'].iloc[j]),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
    
    plt.xlabel('Mixing Layer Thickness')
    plt.ylabel('Fractal Dimension')
    plt.title('Phase Portrait: Fractal Dimension vs. Mixing Layer Thickness')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(multi_res_dir, 'phase_portrait.png'), dpi=300)
    plt.close()

def plot_extrapolation(extrapolation_result, output_dir):
    """Create extrapolation plot for resolution dependence.
    
    Args:
        extrapolation_result: Result dictionary from extrapolate_to_infinite_resolution
        output_dir: Directory to save the plot
    """
    # Extract data from the result
    resolutions = extrapolation_result['resolutions']
    values = extrapolation_result['values']
    h_values = extrapolation_result['h_values']
    name = extrapolation_result['name']
    f_inf = extrapolation_result['value']
    f_inf_err = extrapolation_result['error']
    C = extrapolation_result['coefficient']
    p = extrapolation_result['exponent']
    
    # Define the model function
    def extrapolation_model(h, f_inf, C, p):
        return f_inf + C * h**p
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the data points
    plt.plot(resolutions, values, 'bo-', linewidth=2, markersize=10, 
            label=f'Measured values')
    
    # Add resolution labels to points
    for i, res in enumerate(resolutions):
        plt.annotate(f"{res}×{res}", (resolutions[i], values[i]), 
                    textcoords="offset points", xytext=(5,5), ha='left')
    
    # Create smooth curve for the model
    h_curve = np.linspace(0, h_values[0], 100)
    res_curve = 1.0 / h_curve
    # Filter out inf and nan values
    valid_idx = np.isfinite(res_curve)
    res_curve = res_curve[valid_idx]
    model_curve = extrapolation_model(h_curve, f_inf, C, p)[valid_idx]
    
    plt.plot(res_curve, model_curve, 'r--', linewidth=2,
            label=f'Extrapolation: {name}(∞) = {f_inf:.4f} ± {f_inf_err:.4f}')
    
    # Add horizontal line at extrapolated value
    plt.axhline(y=f_inf, color='k', linestyle=':')
    
    # Format the plot
    plt.xscale('log', base=2)
    plt.xlabel('Resolution (N)', fontsize=14)
    plt.ylabel(f'{name}', fontsize=14)
    plt.title(f'Resolution Convergence of {name}', fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Add text with extrapolation details
    plt.figtext(0.5, 0.01, 
               f"Extrapolation model: {name}(N) = {name}(∞) + C·N^(-p) = {f_inf:.4f} + ({C:.4f})·N^(-{p:.4f})", 
               ha="center", fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(output_dir, f'{name}_extrapolation.png'), dpi=300)
    plt.close()
