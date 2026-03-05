#!/usr/bin/env python3
"""
RT Analyzer CLI - Command line interface for RT analysis tools.
"""

import argparse
import os
import sys
import numpy as np
import glob
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the enhanced RTAnalyzer
from .rt_analyzer import RTAnalyzer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Rayleigh-Taylor Instability Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single VTK file
  rt-analyzer single RT30x60-499.vtk
  
  # Process a series of VTK files
  rt-analyzer series "RT30x60-*.vtk" --resolution 30
  
  # Analyze resolution convergence at specific time
  rt-analyzer convergence RT100x100-9000.vtk RT200x200-9000.vtk RT400x400-9000.vtk --resolutions 100 200 400
  
  # Perform multifractal analysis
  rt-analyzer multifractal RT800x800-9000.vtk
    
  # Analyze multifractal temporal evolution
  rt-analyzer temporal --data-dir ./RT800x800 --time-points 1.0 3.0 5.0 7.0 9.0
    
  # Analyze multifractal resolution dependence
  rt-analyzer resolution --resolutions 100 200 400 800 --time 9.0
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis mode')
    
    # Single file analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single VTK file')
    single_parser.add_argument('vtk_file', help='VTK file to analyze')
    single_parser.add_argument('--output', '-o', default='./rt_analysis',
                              help='Output directory')
    single_parser.add_argument('--analyze-linear', action='store_true', default=True,
                              help='Use linear region analysis for dimension calculation')
    single_parser.add_argument('--trim-boundary', type=int, default=1,
                              help='Number of box sizes to trim from each end')    

    # Series analysis
    series_parser = subparsers.add_parser('series', help='Analyze a series of VTK files')
    series_parser.add_argument('vtk_pattern', help='VTK file pattern (e.g., "RT100x100-*.vtk")')
    series_parser.add_argument('--resolution', '-r', type=int, default=None,
                              help='Grid resolution')
    series_parser.add_argument('--output', '-o', default='./rt_analysis',
                              help='Output directory')
    series_parser.add_argument('--analyze-linear', action='store_true', default=True,
                              help='Use linear region analysis for dimension calculation')
    series_parser.add_argument('--trim-boundary', type=int, default=1,
                              help='Number of box counts to trim from each end')
    
    # Convergence analysis
    conv_parser = subparsers.add_parser('convergence', help='Analyze resolution convergence')
    conv_parser.add_argument('vtk_files', nargs='+', help='VTK files at specific time')
    conv_parser.add_argument('--resolutions', '-r', type=int, nargs='+', required=True,
                            help='Resolutions corresponding to each file')
    conv_parser.add_argument('--time', '-t', type=float, default=9.0,
                            help='Target time for convergence analysis')
    conv_parser.add_argument('--output', '-o', default='./rt_analysis',
                            help='Output directory')
    conv_parser.add_argument('--analyze-linear', action='store_true', default=True,
                              help='Use linear region analysis for dimension calculation')
    conv_parser.add_argument('--trim-boundary', type=int, default=1,
                              help='Number of box counts to trim from each end')    

    # Multifractal analysis
    mf_parser = subparsers.add_parser('multifractal', help='Analyze multifractal properties')
    mf_parser.add_argument('vtk_file', help='VTK file to analyze')
    mf_parser.add_argument('--output', '-o', default='./rt_analysis/multifractal',
                          help='Output directory')
    mf_parser.add_argument('--qmin', type=float, default=-5.0,
                          help='Minimum q value')
    mf_parser.add_argument('--qmax', type=float, default=5.0,
                          help='Maximum q value')
    mf_parser.add_argument('--qstep', type=float, default=0.5,
                          help='Step size for q values')
    
    # Temporal analysis
    temporal_parser = subparsers.add_parser('temporal', 
                                           help='Analyze multifractal temporal evolution')
    temporal_parser.add_argument('--data-dir', '-d', required=True,
                               help='Directory containing VTK files')
    temporal_parser.add_argument('--output', '-o', default='./rt_analysis/temporal',
                               help='Output directory')
    temporal_parser.add_argument('--resolution', '-r', type=int, default=800,
                               help='Resolution to analyze')
    temporal_parser.add_argument('--time-points', nargs='+', type=float,
                               default=[1.0, 3.0, 5.0, 7.0, 9.0],
                               help='Time points to analyze')
    temporal_parser.add_argument('--qmin', type=float, default=-5.0)
    temporal_parser.add_argument('--qmax', type=float, default=5.0)
    temporal_parser.add_argument('--qstep', type=float, default=1.0)
    
    # Resolution analysis
    res_parser = subparsers.add_parser('resolution', 
                                      help='Analyze multifractal resolution dependence')
    res_parser.add_argument('--data-dir', '-d', required=True,
                          help='Base directory containing resolution directories')
    res_parser.add_argument('--output', '-o', default='./rt_analysis/resolution',
                          help='Output directory')
    res_parser.add_argument('--resolutions', nargs='+', type=int, required=True,
                          help='Resolutions to analyze')
    res_parser.add_argument('--time', '-t', type=float, default=9.0,
                          help='Time point to analyze')
    res_parser.add_argument('--qmin', type=float, default=-5.0)
    res_parser.add_argument('--qmax', type=float, default=5.0)
    res_parser.add_argument('--qstep', type=float, default=1.0)
    
    return parser.parse_args()

def run_single_analysis(args):
    """Run analysis on a single VTK file."""
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Analyze file with linear region parameters
    result = analyzer.analyze_vtk_file(
        args.vtk_file,
        analyze_linear=args.analyze_linear,
        trim_boundary=args.trim_boundary
    )    

    return result

def run_series_analysis(args):
    """Run analysis on a series of VTK files."""
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Process VTK series with linear region parameters
    results = analyzer.process_vtk_series(
        args.vtk_pattern, 
        args.resolution,
        analyze_linear=args.analyze_linear,
        trim_boundary=args.trim_boundary
    )    

    return results

def run_convergence_analysis(args):
    """Run resolution convergence analysis."""
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Check if number of files matches number of resolutions
    if len(args.vtk_files) != len(args.resolutions):
        logger.error("Number of VTK files must match number of resolutions.")
        sys.exit(1)
    
    # Run convergence analysis with linear region parameters
    results = analyzer.analyze_resolution_convergence(
        args.vtk_files, 
        args.resolutions, 
        args.time,
        analyze_linear=args.analyze_linear,
        trim_boundary=args.trim_boundary
    )
     
    return results

def run_multifractal_analysis(args):
    """Run multifractal analysis on a single VTK file."""
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Create q values array
    q_values = np.arange(args.qmin, args.qmax + args.qstep/2, args.qstep)
    
    # Use the analyze_multifractal_single method
    results = analyzer.analyze_multifractal_single(
        args.vtk_file, 
        output_dir=args.output,
        q_values=q_values
    )
    
    return results

def run_temporal_analysis(args):
    """Run multifractal temporal evolution analysis."""
    from .rt_visualization import plot_multi_resolution_evolution
    
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Create q values array
    q_values = np.arange(args.qmin, args.qmax + args.qstep/2, args.qstep)
    
    # Create dictionary of time points to VTK files
    time_files = {}
    for t in args.time_points:
        filename = f"RT{args.resolution}x{args.resolution}-{int(t*1000):04d}.vtk"
        filepath = os.path.join(args.data_dir, filename)
        if os.path.exists(filepath):
            time_files[t] = filepath
        else:
            logger.warning(f"File {filepath} not found")
    
    if not time_files:
        logger.error("No valid files found for temporal analysis")
        sys.exit(1)
    
    # Run the temporal evolution analysis
    results = analyzer.analyze_temporal_evolution(
        time_files, 
        output_dir=args.output,
        specific_times=args.time_points
    )
    
    # Generate additional visualizations
    if results:
        plot_multi_resolution_evolution(results, [args.resolution], args.output)
    
    return results

def run_resolution_analysis(args):
    """Run multifractal resolution dependence analysis."""
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Create q values array
    q_values = np.arange(args.qmin, args.qmax + args.qstep/2, args.qstep)
    
    # Create dictionary of resolutions to VTK files
    resolution_files = {}
    for res in args.resolutions:
        filename = f"RT{res}x{res}-{int(args.time*1000):04d}.vtk"
        # Try different possible subdirectory structures
        possible_paths = [
            os.path.join(args.data_dir, filename),
            os.path.join(args.data_dir, f"{res}x{res}", filename),
            os.path.join(args.data_dir, f"res_{res}", filename)
        ]
        
        file_found = False
        for filepath in possible_paths:
            if os.path.exists(filepath):
                resolution_files[res] = filepath
                file_found = True
                break
        
        if not file_found:
            logger.warning(f"No file found for resolution {res}x{res}")
    
    if not resolution_files:
        logger.error("No valid files found for resolution analysis")
        sys.exit(1)
    
    # Use the extrapolate_to_infinite_resolution method for analysis
    results = {}
    
    # First analyze each resolution file individually
    individual_results = {}
    for res, filepath in resolution_files.items():
        logger.info(f"Analyzing resolution {res}x{res}: {filepath}")
        result = analyzer.analyze_multifractal_single(
            filepath, 
            output_dir=os.path.join(args.output, f"res_{res}"),
            q_values=q_values
        )
        individual_results[res] = result
    
    # Now perform extrapolation for key parameters
    if len(individual_results) >= 3:
        resolutions = sorted(individual_results.keys())
        
        # Extract values for key parameters
        d0_values = [individual_results[res]['D0'] for res in resolutions]
        d1_values = [individual_results[res]['D1'] for res in resolutions]
        d2_values = [individual_results[res]['D2'] for res in resolutions]
        
        # Perform extrapolation
        d0_extrap = analyzer.extrapolate_to_infinite_resolution(
            resolutions, d0_values, name="D0", output_dir=args.output
        )
        
        d1_extrap = analyzer.extrapolate_to_infinite_resolution(
            resolutions, d1_values, name="D1", output_dir=args.output
        )
        
        d2_extrap = analyzer.extrapolate_to_infinite_resolution(
            resolutions, d2_values, name="D2", output_dir=args.output
        )
        
        # Store extrapolation results
        results = {
            'individual_results': individual_results,
            'extrapolation': {
                'D0': d0_extrap,
                'D1': d1_extrap,
                'D2': d2_extrap
            }
        }
    else:
        results = {'individual_results': individual_results}
    
    return results

def main():
    """Main entry point for RT analysis CLI."""
    args = parse_args()
    
    # Execute the chosen command
    if args.command == 'single':
        run_single_analysis(args)
        
    elif args.command == 'series':
        run_series_analysis(args)
        
    elif args.command == 'convergence':
        run_convergence_analysis(args)
        
    elif args.command == 'multifractal':
        run_multifractal_analysis(args)
        
    elif args.command == 'temporal':
        run_temporal_analysis(args)
        
    elif args.command == 'resolution':
        run_resolution_analysis(args)
        
    else:
        print("Please specify a command. Use --help for options.")
        sys.exit(1)
    
    logger.info("Analysis completed successfully.")

if __name__ == "__main__":
    main()
