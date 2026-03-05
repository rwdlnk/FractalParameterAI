#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import RTAnalyzer
from rt_analyzer import RTAnalyzer

def analyze_temporal_evolution(output_dir, resolutions, base_pattern=None, specific_times=None):
    """Analyze fractal dimension evolution over time for different resolutions."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each resolution
    all_results = {}
    
    # Process each resolution
    for resolution in resolutions:
        print(f"\n=== Analyzing temporal evolution for {resolution}x{resolution} resolution ===\n")
        
        # Create analyzer instance for this resolution
        res_dir = os.path.join(output_dir, f'res_{resolution}')
        analyzer = RTAnalyzer(res_dir)
        
        # Create pattern for this resolution
        if base_pattern:
            pattern = base_pattern.format(resolution=resolution)
        else:
            pattern = f"./RT{resolution}x{resolution}/*.vtk"
        
        # Find all VTK files for this resolution
        vtk_files = sorted(glob.glob(pattern))
        
        if not vtk_files:
            print(f"Warning: No VTK files found matching pattern: {pattern}")
            continue
        
        print(f"Found {len(vtk_files)} VTK files for {resolution}x{resolution} resolution")
        
        # Results for this resolution
        results = []
        
        # Process each file
        for i, vtk_file in enumerate(vtk_files):
            print(f"Processing file {i+1}/{len(vtk_files)}: {vtk_file}")
            
            try:
                # Read the VTK file
                data = analyzer.read_vtk_file(vtk_file)
                
                # Skip if not in specific times (if provided)
                if specific_times and not any(abs(data['time'] - t) < 0.1 for t in specific_times):
                    print(f"  Skipping time {data['time']} (not in specified times)")
                    continue
                
                # Find initial interface position
                h0 = analyzer.find_initial_interface(data)
                
                # Calculate mixing thickness
                mixing = analyzer.compute_mixing_thickness(data, h0)
                
                # Calculate fractal dimension
                fd_results = analyzer.compute_fractal_dimension(data)
                
                # Store results
                results.append({
                    'time': data['time'],
                    'h0': h0,
                    'ht': mixing['ht'],
                    'hb': mixing['hb'],
                    'h_total': mixing['h_total'],
                    'fractal_dim': fd_results['dimension'],
                    'fd_error': fd_results['error'],
                    'fd_r_squared': fd_results['r_squared'],
                    'resolution': resolution
                })
                
                print(f"  Time: {data['time']:.2f}, Dimension: {fd_results['dimension']:.4f}, "
                      f"Mixing: {mixing['h_total']:.4f}")
                
            except Exception as e:
                print(f"Error processing {vtk_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Convert results to DataFrame and sort by time
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values('time')
            all_results[resolution] = df
            
            # Save results for this resolution
            os.makedirs(res_dir, exist_ok=True)
            df.to_csv(os.path.join(res_dir, 'temporal_evolution.csv'), index=False)
            
            # Create individual plots for this resolution
            plot_single_resolution_evolution(df, resolution, res_dir)
        else:
            print(f"No results for {resolution}x{resolution} resolution")
    
    # Create combined plots across resolutions
    if all_results:
        plot_multi_resolution_evolution(all_results, resolutions, output_dir)
    
    return all_results

def plot_single_resolution_evolution(df, resolution, output_dir):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze fractal dimension temporal evolution')
    parser.add_argument('--resolutions', '-r', type=int, nargs='+', required=True,
                      help='Resolutions to analyze (e.g., 100 200 400 800)')
    parser.add_argument('--output', '-o', default='./temporal_analysis',
                      help='Output directory')
    parser.add_argument('--pattern', default=None,
                      help='Pattern for VTK files with {resolution} placeholder')
    parser.add_argument('--times', type=float, nargs='*',
                      help='Specific time points to analyze (optional)')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_temporal_evolution(args.output, args.resolutions, args.pattern, args.times)
