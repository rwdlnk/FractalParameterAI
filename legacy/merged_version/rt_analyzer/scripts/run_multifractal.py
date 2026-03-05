#!/usr/bin/env python3
"""
RT Instability Multifractal Analysis
Provides options for single file analysis, temporal evolution, or resolution dependence.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Make sure current directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import your RTAnalyzer class
from rt_analyzer import RTAnalyzer

def run_single_analysis(data_dir, output_dir, time_point=9.0, resolution=800):
    """Run multifractal analysis on a single file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RTAnalyzer(output_dir)
    
    # Construct the VTK filename based on resolution and time
    filename = f"RT{resolution}x{resolution}-{int(time_point*1000):04d}.vtk"
    vtk_file = os.path.join(data_dir, filename)
    
    print(f"Reading VTK file: {vtk_file}")
    
    # Read the VTK file
    try:
        data = analyzer.read_vtk_file(vtk_file)
        print(f"VTK file read successfully. Time: {data['time']}")
    except Exception as e:
        print(f"Error reading VTK file: {str(e)}")
        return
    
    # Define q-values for multifractal analysis
    q_values = np.arange(-5, 5.1, 1.0)
    
    # Perform multifractal analysis
    print("Performing multifractal analysis...")
    try:
        result = analyzer.compute_multifractal_spectrum(
            data, 
            q_values=q_values, 
            output_dir=output_dir
        )
        print("Multifractal analysis completed successfully")
    except Exception as e:
        print(f"Error during multifractal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Analysis complete. Results have been saved to {output_dir}")
    return result

def run_temporal_analysis(data_dir, output_dir, resolution=800, time_points=None):
    """Run multifractal analysis across multiple time points."""
    if time_points is None:
        time_points = [1.0, 3.0, 5.0, 7.0, 9.0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RTAnalyzer(output_dir)
    
    # Create dictionary of time points to VTK files
    time_files = {}
    for t in time_points:
        filename = f"RT{resolution}x{resolution}-{int(t*1000):04d}.vtk"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            time_files[t] = filepath
        else:
            print(f"Warning: File {filepath} not found")
    
    if not time_files:
        print("No valid files found for temporal analysis")
        return
    
    print(f"Running temporal analysis on {len(time_files)} time points...")
    
    # Define q-values (using fewer points for faster analysis)
    q_values = np.arange(-5, 5.1, 1.0)
    
    # Run the multifractal evolution analysis
    try:
        results = analyzer.analyze_multifractal_evolution(
            time_files, 
            output_dir=output_dir,
            q_values=q_values
        )
        print("Temporal evolution analysis completed successfully")
    except Exception as e:
        print(f"Error during temporal analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Temporal analysis complete. Results have been saved to {output_dir}")
    return results

def run_resolution_analysis(data_dir, output_dir, time_point=9.0, resolutions=None):
    """Run multifractal analysis across multiple resolutions."""
    if resolutions is None:
        resolutions = [100, 200, 400, 800]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = RTAnalyzer(output_dir)
    
    # Create dictionary of resolutions to VTK files
    resolution_files = {}
    for res in resolutions:
        filename = f"RT{res}x{res}-{int(time_point*1000):04d}.vtk"
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            resolution_files[res] = filepath
        else:
            print(f"Warning: File {filepath} not found")
    
    if not resolution_files:
        print("No valid files found for resolution analysis")
        return
    
    print(f"Running resolution analysis on {len(resolution_files)} resolutions...")
    
    # Define q-values (using fewer points for faster analysis)
    q_values = np.arange(-5, 5.1, 1.0)
    
    # Run the multifractal evolution analysis
    try:
        results = analyzer.analyze_multifractal_evolution(
            resolution_files, 
            output_dir=output_dir,
            q_values=q_values
        )
        print("Resolution analysis completed successfully")
    except Exception as e:
        print(f"Error during resolution analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Resolution analysis complete. Results have been saved to {output_dir}")
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RT Instability Multifractal Analysis')
    
    # Base data directory
    parser.add_argument('--data-dir', '-d', default='..', 
                        help='Base directory containing RT data directories')
    
    # Analysis type
    parser.add_argument('--type', '-t', choices=['single', 'temporal', 'resolution'], 
                        default='single', help='Type of analysis to run')
    
    # Single analysis parameters
    parser.add_argument('--time', type=float, default=9.0, 
                        help='Time point for analysis (default: 9.0)')
    parser.add_argument('--resolution', '-r', type=int, default=800, 
                        help='Resolution for analysis (default: 800)')
    
    # Temporal analysis parameters
    parser.add_argument('--time-points', nargs='+', type=float, 
                        default=[1.0, 3.0, 5.0, 7.0, 9.0], 
                        help='Time points for temporal analysis')
    
    # Resolution analysis parameters
    parser.add_argument('--resolutions', nargs='+', type=int, 
                        default=[100, 200, 400, 800], 
                        help='Resolutions for resolution analysis')
    
    # Output directory
    parser.add_argument('--output-dir', '-o', default='./results', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up paths based on analysis type
    if args.type == 'single':
        # For single analysis, use the resolution directory
        data_dir = os.path.join(args.data_dir, f"{args.resolution}x{args.resolution}")
        output_dir = os.path.join(args.output_dir, f"multifractal_{args.resolution}_t{args.time}")
        run_single_analysis(data_dir, output_dir, args.time, args.resolution)
        
    elif args.type == 'temporal':
        # For temporal analysis, use the resolution directory
        data_dir = os.path.join(args.data_dir, f"{args.resolution}x{args.resolution}")
        output_dir = os.path.join(args.output_dir, f"multifractal_temporal_{args.resolution}")
        run_temporal_analysis(data_dir, output_dir, args.resolution, args.time_points)
        
    elif args.type == 'resolution':
        # For resolution analysis, use the base directory
        output_dir = os.path.join(args.output_dir, f"multifractal_resolution_t{args.time}")
        run_resolution_analysis(args.data_dir, output_dir, args.time, args.resolutions)

if __name__ == "__main__":
    main()
