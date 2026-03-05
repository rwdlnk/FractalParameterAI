# cli.py
import argparse
import os
import sys
from rt_analyzer import RTAnalyzer
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description='Rayleigh-Taylor Instability Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single VTK file
  python -m rt_analyzer single RT30x60-499.vtk
  
  # Process a series of VTK files
  python -m rt_analyzer series "RT30x60-*.vtk" --resolution 30
  
  # Analyze resolution convergence at specific time
  python -m rt_analyzer convergence RT100x100-9000.vtk RT200x200-9000.vtk RT400x400-9000.vtk --resolutions 100 200 400
""")
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis mode')
    
    # Single file analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single VTK file')
    single_parser.add_argument('vtk_file', help='VTK file to analyze')
    single_parser.add_argument('--output', '-o', default='./rt_analysis',
                              help='Output directory')
    
    # Series analysis
    series_parser = subparsers.add_parser('series', help='Analyze a series of VTK files')
    series_parser.add_argument('vtk_pattern', help='VTK file pattern (e.g., "RT100x100-*.vtk")')
    series_parser.add_argument('--resolution', '-r', type=int, default=None,
                              help='Grid resolution')
    series_parser.add_argument('--output', '-o', default='./rt_analysis',
                              help='Output directory')
    
    # Convergence analysis
    conv_parser = subparsers.add_parser('convergence', help='Analyze resolution convergence')
    conv_parser.add_argument('vtk_files', nargs='+', help='VTK files at specific time')
    conv_parser.add_argument('--resolutions', '-r', type=int, nargs='+', required=True,
                            help='Resolutions corresponding to each file')
    conv_parser.add_argument('--time', '-t', type=float, default=9.0,
                            help='Target time for convergence analysis')
    conv_parser.add_argument('--output', '-o', default='./rt_analysis',
                            help='Output directory')
    
    # Multifractal analyis
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


    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return
    
    # Create analyzer
    analyzer = RTAnalyzer(args.output)
    
    # Execute the chosen command
    if args.command == 'single':
        analyzer.analyze_vtk_file(args.vtk_file)
        
    elif args.command == 'series':
        analyzer.process_vtk_series(args.vtk_pattern, args.resolution)
        
    elif args.command == 'convergence':
        if len(args.vtk_files) != len(args.resolutions):
            print("Error: Number of VTK files must match number of resolutions")
            return
        analyzer.analyze_resolution_convergence(args.vtk_files, args.resolutions, args.time)

      
    elif args.command == 'multifractal':
        # Create q values array
        q_values = np.arange(args.qmin, args.qmax + args.qstep/2, args.qstep)
        
        # Read data
        data = analyzer.read_vtk_file(args.vtk_file)
        
        # Perform multifractal analysis
        analyzer.compute_multifractal_spectrum(data, q_values=q_values, output_dir=args.output)

if __name__ == "__main__":
    main()
