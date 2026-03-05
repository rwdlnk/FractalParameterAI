# cli.py
import argparse
import time
import gc
import matplotlib.pyplot as plt
from .main import FractalAnalyzer

def clean_memory():
    """Force garbage collection to free memory."""
    plt.close('all')
    gc.collect()

def main():
    parser = argparse.ArgumentParser(
        description='Universal Fractal Dimension Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and analyze a Koch curve at level 5
  python -m fractal_analyzer --generate koch --level 5
  
  # Analyze a coastline file
  python -m fractal_analyzer --file coastline.txt
  
  # Analyze with custom box sizes
  python -m fractal_analyzer --file coastline.txt --min_box_size 0.0005
""")
    parser.add_argument('--file', help='Path to file containing line segments')
    parser.add_argument('--generate', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                        help='Generate a fractal curve of specified type')
    parser.add_argument('--level', type=int, default=5, help='Level for fractal generation')
    parser.add_argument('--min_box_size', type=float, default=0.001, 
                        help='Minimum box size for calculation')
    parser.add_argument('--max_box_size', type=float, default=None, 
                        help='Maximum box size for calculation (default: auto-determined)')
    parser.add_argument('--box_size_factor', type=float, default=2.0, 
                        help='Factor by which to reduce box size in each step')
    parser.add_argument('--no_plot', action='store_true', 
                        help='Disable plotting')
    parser.add_argument('--no_box_plot', action='store_true',
                        help='Disable box overlay in the curve plot')
    parser.add_argument('--analyze_iterations', action='store_true',
                       help='Analyze how iteration depth affects measured dimension')
    parser.add_argument('--min_level', type=int, default=1,
                       help='Minimum curve level for iteration analysis')
    parser.add_argument('--max_level', type=int, default=8,
                       help='Maximum curve level for iteration analysis')
    parser.add_argument('--analyze_linear_region', action='store_true',
                       help='Analyze how linear region selection affects dimension')
    parser.add_argument('--fractal_type', type=str, choices=['koch', 'sierpinski', 'minkowski', 'hilbert', 'dragon'],
                       help='Specify fractal type for analysis (needed when using --file)')
    parser.add_argument('--trim_boundary', type=int, default=0,
                       help='Number of box counts to trim from each end of the data')
    
    args = parser.parse_args()
    
    # Display version info
    print(f"Running Fractal Analyzer v24 (Modular)")
    print(f"-------------------------------")
    
    # Create analyzer instance
    fractal_type = args.generate if args.generate else args.fractal_type
    analyzer = FractalAnalyzer(fractal_type)
    
    # Clean memory before starting
    clean_memory()
    
    # Generate a fractal curve if requested
    if args.generate:
        _, segments = analyzer.generate_fractal(args.generate, args.level)
        filename = f'{args.generate}_segments_level_{args.level}.txt'
        analyzer.write_segments_to_file(segments, filename)
        print(f"{args.generate.capitalize()} curve saved to {filename}")
        
        # Use this curve for analysis if no file is specified
        if args.file is None:
            args.file = filename
            analyzer.fractal_type = args.generate
    
    # Read line segments from file or use generated segments
    if args.file:
        segments = analyzer.read_line_segments(args.file)
        print(f"Read {len(segments)} line segments from {args.file}")
        
        if not segments:
            print("No valid line segments found. Exiting.")
            return
        
        # Advanced analysis options
        if args.analyze_linear_region:
            # Create analysis tools if not already present
            if not hasattr(analyzer, 'analysis_tools'):
                from .analysis_tools import FractalAnalysisTools
                analyzer.analysis_tools = FractalAnalysisTools(analyzer)
            
            print("\n=== Starting Linear Region Analysis ===\n")
            analyzer.analysis_tools.analyze_linear_region(
                segments, analyzer.fractal_type, not args.no_plot, 
                not args.no_box_plot, trim_boundary=args.trim_boundary)
            print("\n=== Linear Region Analysis Complete ===\n")
        
        # Iteration analysis if requested
        elif args.analyze_iterations:
            if not analyzer.fractal_type:
                print("Error: Must specify --fractal_type or --generate for iteration analysis")
                return
                
            # Create analysis tools if not already present
            if not hasattr(analyzer, 'analysis_tools'):
                from .analysis_tools import FractalAnalysisTools
                analyzer.analysis_tools = FractalAnalysisTools(analyzer)
                
            print("\n=== Starting Iteration Analysis ===\n")
            # Call iteration analysis method (needs to be implemented in analysis_tools.py)
            analyzer.analysis_tools.analyze_iterations(
                args.min_level, args.max_level, analyzer.fractal_type, 
                no_plots=args.no_plot, no_box_plot=args.no_box_plot)
            print("\n=== Iteration Analysis Complete ===\n")
            
        # Standard dimension calculation if no special analysis is requested
        else:
            try:
                # Calculate fractal dimension
                fractal_dimension, error, box_sizes, box_counts, bounding_box, intercept = analyzer.calculate_fractal_dimension(
                    segments, args.min_box_size, args.max_box_size, args.box_size_factor)
                
                # Print results
                print(f"Results:")
                print(f"  Fractal Dimension: {fractal_dimension:.6f} ± {error:.6f}")
                
                if analyzer.fractal_type and analyzer.fractal_type in analyzer.base.THEORETICAL_DIMENSIONS:
                    theoretical = analyzer.base.THEORETICAL_DIMENSIONS[analyzer.fractal_type]
                    print(f"  Theoretical {analyzer.fractal_type} dimension: {theoretical:.6f}")
                    print(f"  Difference: {abs(fractal_dimension - theoretical):.6f}")
                
                # Plot if requested
                if not args.no_plot:
                    plot_filename = analyzer.plot_results(
                        segments, box_sizes, box_counts, fractal_dimension, error, 
                        bounding_box, plot_boxes=not args.no_box_plot)
                    print(f"Plot saved to {plot_filename}")
            
            except Exception as e:
                print(f"Error during calculation: {str(e)}")
                import traceback
                traceback.print_exc()
                return
    
    if not (args.file or args.generate):
        print("No input file specified and no curve generation requested.")
        print("Use --file to specify an input file or --generate to create a fractal curve.")
        parser.print_help()

if __name__ == "__main__":
    main()
