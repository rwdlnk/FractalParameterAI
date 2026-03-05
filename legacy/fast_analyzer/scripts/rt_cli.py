#!/usr/bin/env python3
"""
Command-Line Interface for RT Analyzer with FastFractalAnalyzer Integration

This provides a simple command-line way to run RT analysis with breakthrough
90× accuracy improvement on VTK files.

Usage Examples:
    python rt_cli.py analyze /path/to/file.vtk
    python rt_cli.py batch /path/to/data/*.vtk --output results/
    python rt_cli.py analyze RT160x200-10000.vtk --fractal-only --interface 0.5
"""

import sys
import os
import argparse
import glob
import time
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def setup_argument_parser():
    """Setup command line argument parser."""

    parser = argparse.ArgumentParser(
        description='RT Analyzer with FastFractalAnalyzer Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single VTK file
  python rt_cli.py analyze /path/to/RT160x200-10000.vtk

  # Fractal analysis only with custom interface
  python rt_cli.py analyze RT160x200-10000.vtk --fractal-only --interface 0.5

  # Include multifractal analysis
  python rt_cli.py analyze RT160x200-10000.vtk --fractal-only --multifractal --q-values -5 -3 -1 0 1 3 5

  # Batch process multiple files
  python rt_cli.py batch "/path/to/data/RT160x200-*.vtk" --output results/

  # Batch with multifractal analysis
  python rt_cli.py batch "RT160x200-*.vtk" --multifractal --max-files 3

  # Show analyzer capabilities
  python rt_cli.py info
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze single VTK file')
    analyze_parser.add_argument('vtk_file', help='Path to VTK file')
    analyze_parser.add_argument('--output', '-o', default='./rt_results',
                               help='Output directory (default: ./rt_results)')
    analyze_parser.add_argument('--interface', '-i', type=float, default=0.5,
                               help='Interface threshold value (default: 0.5)')
    analyze_parser.add_argument('--min-box-size', type=float,
                               help='Minimum box size for fractal analysis')
    analyze_parser.add_argument('--fractal-only', action='store_true',
                               help='Run fractal analysis only (skip mixing)')
    analyze_parser.add_argument('--mixing-only', action='store_true',
                               help='Run mixing analysis only (skip fractal)')
    analyze_parser.add_argument('--multifractal', action='store_true',
                               help='Enable multifractal analysis')
    analyze_parser.add_argument('--q-values', nargs='+', type=float,
                               help='Q values for multifractal analysis (e.g., --q-values -5 -3 -1 0 1 3 5)')
    analyze_parser.add_argument('--mf-output-dir',
                               help='Custom output directory for multifractal results')
    analyze_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Verbose output')
    analyze_parser.add_argument('--generate-plots', action='store_true',
                               help='Generate and save plots for analysis results')
    analyze_parser.add_argument('--use_conrec', action='store_true',
                               help='Force CONREC interface extraction (more segments)')
    analyze_parser.add_argument('--use_plic', action='store_true',
                               help='Force PLIC interface extraction')
    analyze_parser.add_argument('--power-spectrum', action='store_true',
                               help='Enable power spectrum analysis')
    analyze_parser.add_argument('--rms-evolution', action='store_true',
                               help='Enable RMS velocity evolution analysis')
    analyze_parser.add_argument('--power-spectrum-output-dir',
                               help='Custom output directory for power spectrum results')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process VTK files')
    batch_parser.add_argument('pattern', help='File pattern (e.g., "*.vtk" or "/path/to/RT*.vtk")')
    batch_parser.add_argument('--output', '-o', default='./rt_batch_results',
                             help='Output directory (default: ./rt_batch_results)')
    batch_parser.add_argument('--interface', '-i', type=float, default=0.5,
                             help='Interface threshold value (default: 0.5)')
    batch_parser.add_argument('--fractal-only', action='store_true',
                             help='Run fractal analysis only')
    batch_parser.add_argument('--multifractal', action='store_true',
                             help='Enable multifractal analysis for all files')
    batch_parser.add_argument('--q-values', nargs='+', type=float,
                             help='Q values for multifractal analysis')
    batch_parser.add_argument('--mf-output-dir',
                             help='Custom output directory for multifractal results')
    batch_parser.add_argument('--max-files', type=int,
                             help='Maximum number of files to process')
    batch_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    batch_parser.add_argument('--generate-plots', action='store_true',
                             help='Generate and save plots for analysis results')
    batch_parser.add_argument('--use_conrec', action='store_true',
                             help='Force CONREC interface extraction (more segments)')
    batch_parser.add_argument('--use_plic', action='store_true',
                             help='Force PLIC interface extraction')
    batch_parser.add_argument('--power-spectrum', action='store_true',
                             help='Enable power spectrum analysis')
    batch_parser.add_argument('--rms-evolution', action='store_true',
                             help='Enable RMS velocity evolution analysis')
    batch_parser.add_argument('--power-spectrum-output-dir',
                             help='Custom output directory for power spectrum results')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show analyzer information')

    return parser

def analyze_single_file(args):
    """Analyze a single VTK file."""

    print(f"🌊 RT Analyzer CLI - Single File Analysis")
    print(f"=" * 50)

    # Import RT analyzer
    from fractal_analyzer.core.rt_analyzer import RTAnalyzer

    # Check file exists
    if not os.path.exists(args.vtk_file):
        print(f"❌ Error: File not found: {args.vtk_file}")
        return False

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize analyzer
    print(f"🚀 Initializing RT Analyzer...")
    if args.verbose:
        print(f"   Output directory: {args.output}")
        print(f"   Interface threshold: {args.interface}")

    rt_analyzer = RTAnalyzer(
        output_dir=args.output,
        use_grid_optimization=True,
        no_titles=not args.verbose,
        use_conrec=getattr(args, 'use_conrec', False),
        use_plic=getattr(args, 'use_plic', False)
    )

    # Initialize power spectrum analyzer if requested
    if getattr(args, 'power_spectrum', False) or getattr(args, 'rms_evolution', False):
        try:
            from fractal_analyzer.core.power_spectrum_analyzer import EnhancedDalzielPowerSpectrumAnalyzer
            rt_analyzer.power_spectrum_analyzer = EnhancedDalzielPowerSpectrumAnalyzer()
            if args.verbose:
                print(f"   🌊 Power spectrum analyzer initialized")
        except ImportError as e:
            print(f"   ⚠️  Power spectrum analyzer not available: {e}")

    # Show capabilities
    if args.verbose:
        caps = rt_analyzer.fractal_analyzer.get_capabilities()
        print(f"🎯 Capabilities:")
        print(f"   Advanced framework: {caps['advanced_framework']}")
        print(f"   Expected improvement: {caps['expected_improvement']}")

    # Determine analysis types
    analysis_types = []
    if args.fractal_only:
        analysis_types = ['fractal_dim']
    elif args.mixing_only:
        analysis_types = ['mixing']
    else:
        analysis_types = ['fractal_dim', 'mixing']

    # Run analysis
    print(f"\n🔍 Analyzing: {os.path.basename(args.vtk_file)}")
    print(f"📊 Analysis types: {', '.join(analysis_types)}")

    try:
        start_time = time.time()

        result = rt_analyzer.analyze_vtk_file(
            args.vtk_file,
            analysis_types=analysis_types,
            y0=args.interface,
            min_box_size=args.min_box_size,
            enable_multifractal=args.multifractal,
            q_values=args.q_values,
            mf_output_dir=args.mf_output_dir
        )

        analysis_time = time.time() - start_time

        if result:
            print(f"✅ Analysis completed in {analysis_time:.1f}s")

            # Display results
            print(f"\n📊 Results:")

            if 'fractal_dimension' in result:
                print(f"   Fractal Dimension: {result['fractal_dimension']:.6f}")
                print(f"   R²: {result.get('fractal_r_squared', 'N/A'):.6f}")
                print(f"   Interface segments: {result.get('interface_point_count', 'N/A')}")

                # Check advanced framework usage
                if rt_analyzer.fractal_analyzer.use_advanced_framework:
                    print(f"   🌟 Advanced Box Counting Framework used!")

            if 'mixing_width' in result:
                print(f"   Mixing width: {result.get('mixing_width', 'N/A'):.6f}")
                print(f"   Thickness: {result.get('thickness', 'N/A'):.6f}")

            if 'multifractal' in result and result['multifractal']:
                mf = result['multifractal']
                print(f"   🌀 Multifractal Analysis:")
                print(f"   D₀ (capacity): {mf.get('D0', 'N/A'):.6f}")
                print(f"   D₁ (information): {mf.get('D1', 'N/A'):.6f}")
                print(f"   D₂ (correlation): {mf.get('D2', 'N/A'):.6f}")
                print(f"   Singularity width: {mf.get('singularity_width', 'N/A'):.6f}")

            # Perform power spectrum analysis if requested
            if getattr(args, 'power_spectrum', False) or getattr(args, 'rms_evolution', False):
                if hasattr(rt_analyzer, 'power_spectrum_analyzer') and rt_analyzer.power_spectrum_analyzer:

                    # RMS analysis
                    if getattr(args, 'rms_evolution', False):
                        print(f"\n🌊 Performing RMS velocity analysis...")
                        try:
                            # For single file, create a list with just this file
                            rms_results = rt_analyzer.power_spectrum_analyzer.analyze_rms_evolution(
                                [args.vtk_file],
                                output_dir=getattr(args, 'power_spectrum_output_dir', args.output)
                            )
                            if rms_results:
                                print(f"   ✅ RMS analysis completed")
                                print(f"   📊 u_rms: {rms_results['u_rms'][0]:.6f}")
                                print(f"   📊 v_rms: {rms_results['v_rms'][0]:.6f}")
                                print(f"   📊 total_rms: {rms_results['total_rms'][0]:.6f}")
                            else:
                                print(f"   ⚠️  RMS analysis returned no results")
                        except Exception as e:
                            print(f"   ❌ RMS analysis failed: {e}")

                    # Dual power spectrum analysis (velocity + volume fraction)
                    if getattr(args, 'power_spectrum', False):
                        print(f"\n📊 Performing dual power spectrum analysis (velocity + volume fraction)...")
                        try:
                            ps_output_dir = getattr(args, 'power_spectrum_output_dir', args.output)
                            os.makedirs(ps_output_dir, exist_ok=True)

                            # Analyze single file dual spectrum (Dalziel method)
                            dual_results = rt_analyzer.power_spectrum_analyzer.analyze_rt_dual_spectrum(
                                args.vtk_file,
                                output_dir=ps_output_dir
                            )

                            if dual_results:
                                print(f"   ✅ Dual spectrum analysis completed")

                                # Display velocity spectrum
                                if 'velocity_spectrum' in dual_results and dual_results['velocity_spectrum']:
                                    vel_spec = dual_results['velocity_spectrum']
                                    k_norm, P_norm = vel_spec[0], vel_spec[1]

                                    print(f"   🌊 Velocity Spectrum E_vel(k/k₀):")
                                    print(f"   {'k/k₀':<8} {'E_vel(k)':<12}")
                                    print(f"   {'-'*20}")
                                    for i in range(min(8, len(k_norm))):
                                        print(f"   {k_norm[i]:<8.3f} {P_norm[i]:<12.6e}")
                                    if len(k_norm) > 8:
                                        print(f"   ... ({len(k_norm)-8} more values)")

                                # Display volume fraction spectrum
                                if 'vof_spectrum' in dual_results and dual_results['vof_spectrum']:
                                    vof_spec = dual_results['vof_spectrum']
                                    k_norm_vof, P_norm_vof = vof_spec[0], vof_spec[1]

                                    print(f"   🧪 Volume Fraction Spectrum E_F(k/k₀):")
                                    print(f"   {'k/k₀':<8} {'E_F(k)':<12}")
                                    print(f"   {'-'*20}")
                                    for i in range(min(8, len(k_norm_vof))):
                                        print(f"   {k_norm_vof[i]:<8.3f} {P_norm_vof[i]:<12.6e}")
                                    if len(k_norm_vof) > 8:
                                        print(f"   ... ({len(k_norm_vof)-8} more values)")

                                print(f"   📁 Full dual spectrum saved to: {ps_output_dir}")
                            else:
                                print(f"   ⚠️  Dual spectrum analysis returned no results")
                        except Exception as e:
                            print(f"   ❌ Dual spectrum analysis failed: {e}")
                            if args.verbose:
                                import traceback
                                traceback.print_exc()
                else:
                    print(f"   ⚠️  Power spectrum analyzer not available")

            print(f"\n📁 Results saved to: {args.output}")
            return True

        else:
            print(f"❌ Analysis failed - no results returned")
            return False

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False

def batch_process_files(args):
    """Batch process multiple VTK files."""

    print(f"🌊 RT Analyzer CLI - Batch Processing")
    print(f"=" * 50)

    # Find files
    files = glob.glob(args.pattern)
    if not files:
        print(f"❌ No files found matching pattern: {args.pattern}")
        return False

    # Sort files by simulation time (extracted from filename)
    # RT160x200-xxxxx.vtk where xxxxx = 1000*time_in_seconds
    def extract_time_from_filename(filename):
        """Extract simulation time from RT filename format."""
        import re
        match = re.search(r'RT\d+x\d+-(\d+)\.vtk', filename)
        if match:
            return int(match.group(1))  # Return time index for sorting
        return 0  # Fallback for non-matching files

    files = sorted(files, key=extract_time_from_filename)
    if args.max_files:
        files = files[:args.max_files]

    print(f"📁 Found {len(files)} files to process")

    # Import RT analyzer
    from fractal_analyzer.core.rt_analyzer import RTAnalyzer

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize analyzer
    print(f"🚀 Initializing RT Analyzer...")
    rt_analyzer = RTAnalyzer(
        output_dir=args.output,
        use_grid_optimization=True,
        no_titles=True,
        use_conrec=getattr(args, 'use_conrec', False),
        use_plic=getattr(args, 'use_plic', False)
    )

    # Initialize power spectrum analyzer if requested
    if getattr(args, 'power_spectrum', False) or getattr(args, 'rms_evolution', False):
        try:
            from fractal_analyzer.core.power_spectrum_analyzer import EnhancedDalzielPowerSpectrumAnalyzer
            rt_analyzer.power_spectrum_analyzer = EnhancedDalzielPowerSpectrumAnalyzer()
            print(f"   🌊 Power spectrum analyzer initialized")
        except ImportError as e:
            print(f"   ⚠️  Power spectrum analyzer not available: {e}")

    # Analysis types
    analysis_types = ['fractal_dim'] if args.fractal_only else ['fractal_dim', 'mixing']

    # Process files
    results = []
    successful = 0

    print(f"\n🔍 Processing files...")
    for i, vtk_file in enumerate(files, 1):
        filename = os.path.basename(vtk_file)

        # Clear file separator
        print(f"\n{'=' * 60}")
        print(f"📁 [{i:3d}/{len(files)}] Processing: {filename}")
        print(f"{'=' * 60}")

        try:
            start_time = time.time()

            result = rt_analyzer.analyze_vtk_file(
                vtk_file,
                analysis_types=analysis_types,
                y0=args.interface,
                enable_multifractal=args.multifractal,
                q_values=args.q_values,
                mf_output_dir=args.mf_output_dir
            )

            analysis_time = time.time() - start_time

            if result and 'fractal_dimension' in result:
                dimension = result.get('fractal_dimension', float('nan'))
                r_squared = result.get('fractal_r_squared', float('nan'))

                print(f"\n✅ COMPLETED: {filename}")
                print(f"   📊 Fractal Dimension: {dimension:.6f}")
                print(f"   📈 R²: {r_squared:.6f}")
                print(f"   ⏱️  Processing Time: {analysis_time:.1f}s")

                # Store detailed results for the comprehensive CSV
                detailed_result = {
                    'filename': filename,
                    'file_path': vtk_file,
                    'time': result.get('time', 0.0),
                    'interface_extraction_time': result.get('interface_extraction_time', 0.0),
                    'interface_point_count': result.get('interface_point_count', 0),
                    'interface_bounds': str(result.get('interface_bounds', '')),
                    'grid_shape': str(result.get('grid_shape', '')),
                    'analysis_types_performed': str(result.get('analysis_types_performed', [])),
                    'fractal_dimension': dimension,
                    'fractal_error': result.get('fractal_error', ''),
                    'fractal_r_squared': r_squared,
                    'fractal_computation_time': result.get('fractal_computation_time', 0.0),
                    'total_analysis_time': analysis_time,
                    'nx': result.get('nx', 0),
                    'ny': result.get('ny', 0),
                    'grid_type': result.get('grid_type', ''),
                    'time_step': result.get('time_step', result.get('time', 0.0)),
                }

                # Add comprehensive mixing results if available
                if 'mixing_width' in result:
                    detailed_result['mixing_width'] = result.get('mixing_width', float('nan'))
                    detailed_result['thickness'] = result.get('thickness', float('nan'))
                    detailed_result['mixing_computation_time'] = result.get('mixing_computation_time', float('nan'))
                    detailed_result['mixing_method'] = result.get('mixing_method', '')
                    detailed_result['mixing_error'] = result.get('mixing_error', '')
                    detailed_result['dalziel_h'] = result.get('dalziel_h', float('nan'))
                    detailed_result['dalziel_h_normalized'] = result.get('dalziel_h_normalized', float('nan'))
                    detailed_result['integral_width'] = result.get('integral_width', float('nan'))
                    detailed_result['statistical_width'] = result.get('statistical_width', float('nan'))
                    detailed_result['mixing_efficiency'] = result.get('mixing_efficiency', float('nan'))
                else:
                    # Add NaN values for missing mixing data
                    detailed_result['mixing_width'] = float('nan')
                    detailed_result['thickness'] = float('nan')
                    detailed_result['mixing_computation_time'] = float('nan')
                    detailed_result['mixing_method'] = ''
                    detailed_result['mixing_error'] = ''
                    detailed_result['dalziel_h'] = float('nan')
                    detailed_result['dalziel_h_normalized'] = float('nan')
                    detailed_result['integral_width'] = float('nan')
                    detailed_result['statistical_width'] = float('nan')
                    detailed_result['mixing_efficiency'] = float('nan')

                # Add comprehensive multifractal results if available
                if 'multifractal' in result and result['multifractal']:
                    mf = result['multifractal']
                    detailed_result['multifractal_enabled'] = True
                    detailed_result['D0_capacity'] = mf.get('D0', float('nan'))
                    detailed_result['D1_information'] = mf.get('D1', float('nan'))
                    detailed_result['D2_correlation'] = mf.get('D2', float('nan'))
                    detailed_result['Dq_minus5'] = mf.get('D-5', float('nan'))
                    detailed_result['Dq_minus3'] = mf.get('D-3', float('nan'))
                    detailed_result['Dq_minus1'] = mf.get('D-1', float('nan'))
                    detailed_result['Dq_plus1'] = mf.get('D1', float('nan'))
                    detailed_result['Dq_plus3'] = mf.get('D3', float('nan'))
                    detailed_result['Dq_plus5'] = mf.get('D5', float('nan'))
                    detailed_result['singularity_width'] = mf.get('singularity_width', float('nan'))
                    detailed_result['singularity_strength'] = mf.get('singularity_strength', float('nan'))
                    detailed_result['multifractal_spectrum_range'] = str(mf.get('spectrum_range', ''))
                    detailed_result['multifractal_computation_time'] = mf.get('computation_time', float('nan'))
                    detailed_result['q_values_used'] = str(mf.get('q_values', []))
                    detailed_result['scaling_quality'] = mf.get('scaling_quality', float('nan'))
                    detailed_result['generalized_dimensions'] = str(mf.get('generalized_dimensions', {}))
                else:
                    # Add NaN values for missing multifractal data
                    detailed_result['multifractal_enabled'] = False
                    detailed_result['D0_capacity'] = float('nan')
                    detailed_result['D1_information'] = float('nan')
                    detailed_result['D2_correlation'] = float('nan')
                    detailed_result['Dq_minus5'] = float('nan')
                    detailed_result['Dq_minus3'] = float('nan')
                    detailed_result['Dq_minus1'] = float('nan')
                    detailed_result['Dq_plus1'] = float('nan')
                    detailed_result['Dq_plus3'] = float('nan')
                    detailed_result['Dq_plus5'] = float('nan')
                    detailed_result['singularity_width'] = float('nan')
                    detailed_result['singularity_strength'] = float('nan')
                    detailed_result['multifractal_spectrum_range'] = ''
                    detailed_result['multifractal_computation_time'] = float('nan')
                    detailed_result['q_values_used'] = ''
                    detailed_result['scaling_quality'] = float('nan')
                    detailed_result['generalized_dimensions'] = ''

                # Add RMS velocity values if power spectrum analyzer is enabled
                if hasattr(rt_analyzer, 'power_spectrum_analyzer') and rt_analyzer.power_spectrum_analyzer is not None:
                    try:
                        # Read VTK file to get velocity fields for RMS calculation
                        u_field, v_field = rt_analyzer.power_spectrum_analyzer._read_velocity_field(vtk_file)
                        if u_field is not None and v_field is not None:
                            rms_data = rt_analyzer.power_spectrum_analyzer.compute_rms_velocities(u_field, v_field)
                            if rms_data:
                                detailed_result['u_rms'] = rms_data['u_rms']
                                detailed_result['v_rms'] = rms_data['v_rms']
                                detailed_result['total_rms'] = rms_data['total_rms']
                            else:
                                detailed_result['u_rms'] = float('nan')
                                detailed_result['v_rms'] = float('nan')
                                detailed_result['total_rms'] = float('nan')
                        else:
                            detailed_result['u_rms'] = float('nan')
                            detailed_result['v_rms'] = float('nan')
                            detailed_result['total_rms'] = float('nan')
                    except Exception as e:
                        # If RMS calculation fails, add NaN values
                        detailed_result['u_rms'] = float('nan')
                        detailed_result['v_rms'] = float('nan')
                        detailed_result['total_rms'] = float('nan')
                else:
                    # No power spectrum analyzer, add NaN values
                    detailed_result['u_rms'] = float('nan')
                    detailed_result['v_rms'] = float('nan')
                    detailed_result['total_rms'] = float('nan')

                results.append(detailed_result)
                successful += 1
            else:
                print(f"\n❌ FAILED: {filename} - No results returned")

        except Exception as e:
            print(f"\n❌ ERROR: {filename} - {e}")
            continue

    # Power spectrum and RMS evolution analysis for batch processing
    if getattr(args, 'rms_evolution', False) and len(files) > 1:
        if hasattr(rt_analyzer, 'power_spectrum_analyzer') and rt_analyzer.power_spectrum_analyzer:
            print(f"\n🌊 Performing RMS velocity evolution analysis for {len(files)} files...")
            try:
                # Create power spectrum output directory
                power_spectrum_dir = getattr(args, 'power_spectrum_output_dir', args.output)
                os.makedirs(power_spectrum_dir, exist_ok=True)

                # Analyze RMS evolution across all files
                rms_results = rt_analyzer.power_spectrum_analyzer.analyze_rms_evolution(
                    files,
                    output_dir=power_spectrum_dir
                )
                if rms_results:
                    print(f"   ✅ RMS evolution analysis completed")
                    print(f"   📊 Average u_rms: {sum(rms_results['u_rms'])/len(rms_results['u_rms']):.6f}")
                    print(f"   📊 Average v_rms: {sum(rms_results['v_rms'])/len(rms_results['v_rms']):.6f}")
                    print(f"   📊 Average total_rms: {sum(rms_results['total_rms'])/len(rms_results['total_rms']):.6f}")

                    # Create evolution plots
                    rt_analyzer.power_spectrum_analyzer.plot_rms_evolution(
                        rms_results, power_spectrum_dir
                    )
                    print(f"   📈 RMS evolution plots saved to: {power_spectrum_dir}")
                else:
                    print(f"   ⚠️  RMS analysis returned no results")
            except Exception as e:
                print(f"   ❌ RMS analysis failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            print(f"   ⚠️  Power spectrum analyzer not available for RMS analysis")

    if getattr(args, 'power_spectrum', False) and len(files) > 1:
        if hasattr(rt_analyzer, 'power_spectrum_analyzer') and rt_analyzer.power_spectrum_analyzer:
            print(f"\n📊 Performing dual spectrum analysis (velocity + volume fraction) for {len(files)} files...")
            try:
                # Create power spectrum output directory
                power_spectrum_dir = getattr(args, 'power_spectrum_output_dir', args.output)
                os.makedirs(power_spectrum_dir, exist_ok=True)

                # Analyze dual spectra for all files
                dual_results_all = []
                for i, vtk_file in enumerate(files):
                    # Show progress every 10 files
                    if i % 10 == 0 or i == len(files) - 1:
                        print(f"   Processing dual spectrum: {i+1}/{len(files)} files...")

                    dual_result = rt_analyzer.power_spectrum_analyzer.analyze_rt_dual_spectrum(
                        vtk_file,
                        output_dir=power_spectrum_dir
                    )
                    if dual_result:
                        dual_results_all.append(dual_result)

                if dual_results_all:
                    print(f"   ✅ Dual spectrum analysis completed for {len(dual_results_all)} files")

                    # Display summary of first and last files
                    if len(dual_results_all) > 0:
                        first_result = dual_results_all[0]
                        last_result = dual_results_all[-1]

                        print(f"   🌊 Velocity spectrum range:")
                        if 'velocity_spectrum' in first_result and first_result['velocity_spectrum']:
                            first_vel = first_result['velocity_spectrum']
                            last_vel = last_result['velocity_spectrum']
                            print(f"     First file peak: k/k₀={first_vel[0][0]:.3f}, E_vel={first_vel[1][0]:.3e}")
                            print(f"     Last file peak:  k/k₀={last_vel[0][0]:.3f}, E_vel={last_vel[1][0]:.3e}")

                        print(f"   🧪 Volume fraction spectrum range:")
                        if 'vof_spectrum' in first_result and first_result['vof_spectrum']:
                            first_vof = first_result['vof_spectrum']
                            last_vof = last_result['vof_spectrum']
                            print(f"     First file peak: k/k₀={first_vof[0][0]:.3f}, E_F={first_vof[1][0]:.3e}")
                            print(f"     Last file peak:  k/k₀={last_vof[0][0]:.3f}, E_F={last_vof[1][0]:.3e}")

                    print(f"   📈 Dual spectrum data saved to: {power_spectrum_dir}")
                else:
                    print(f"   ⚠️  Dual spectrum analysis returned no results")
            except Exception as e:
                print(f"   ❌ Dual spectrum analysis failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            print(f"   ⚠️  Power spectrum analyzer not available")

    # Summary
    print(f"\n" + "=" * 50)
    print(f"📋 Batch Processing Summary")
    print(f"=" * 50)
    print(f"✅ Successfully processed: {successful}/{len(files)} files")

    if results:
        # Calculate statistics from detailed results
        dimensions = [r['fractal_dimension'] for r in results if not (r['fractal_dimension'] != r['fractal_dimension'])]  # Filter NaN
        r_squared_values = [r['fractal_r_squared'] for r in results if not (r['fractal_r_squared'] != r['fractal_r_squared'])]  # Filter NaN
        times = [r['total_analysis_time'] for r in results]

        if dimensions:
            avg_dimension = sum(dimensions) / len(dimensions)
            avg_r2 = sum(r_squared_values) / len(r_squared_values) if r_squared_values else 0
            avg_time = sum(times) / len(times)

            print(f"📊 Average Results:")
            print(f"   Mean fractal dimension: {avg_dimension:.6f}")
            print(f"   Mean R²: {avg_r2:.6f}")
            print(f"   Average processing time: {avg_time:.1f}s")

    # Save comprehensive results to single CSV file
    if results:
        import pandas as pd

        # Create comprehensive DataFrame with all details
        df = pd.DataFrame(results)

        # Save comprehensive CSV with all details
        comprehensive_csv = os.path.join(args.output, 'batch_results_comprehensive.csv')
        df.to_csv(comprehensive_csv, index=False)
        print(f"📊 Comprehensive results saved to: {comprehensive_csv}")

        # Generate plots if requested
        if args.generate_plots:
            print(f"\n📊 Generating time series plots...")
            _generate_batch_plots(args.output, files, rt_analyzer)

        # Perform RMS evolution analysis if requested (ideal for batch processing)
        if getattr(args, 'rms_evolution', False) and len(files) > 1:
            if hasattr(rt_analyzer, 'power_spectrum_analyzer') and rt_analyzer.power_spectrum_analyzer:
                print(f"\n🌊 Performing RMS velocity evolution analysis on {len(files)} files...")
                try:
                    rms_results = rt_analyzer.power_spectrum_analyzer.analyze_rms_evolution(
                        files,
                        output_dir=getattr(args, 'power_spectrum_output_dir', args.output)
                    )
                    if rms_results:
                        print(f"   ✅ RMS evolution analysis completed")
                        print(f"   📊 Average u_rms: {sum(rms_results['u_rms'])/len(rms_results['u_rms']):.6f}")
                        print(f"   📊 Average v_rms: {sum(rms_results['v_rms'])/len(rms_results['v_rms']):.6f}")
                        print(f"   📊 Average total_rms: {sum(rms_results['total_rms'])/len(rms_results['total_rms']):.6f}")

                        # Generate RMS evolution plots
                        rt_analyzer.power_spectrum_analyzer.plot_rms_evolution(
                            rms_results,
                            getattr(args, 'power_spectrum_output_dir', args.output)
                        )
                        print(f"   📈 RMS evolution plots generated")
                    else:
                        print(f"   ⚠️  RMS analysis returned no results")
                except Exception as e:
                    print(f"   ❌ RMS analysis failed: {e}")
            else:
                print(f"   ⚠️  Power spectrum analyzer not available for RMS analysis")
        elif getattr(args, 'rms_evolution', False) and len(files) == 1:
            print(f"   💡 RMS evolution requires multiple files (found {len(files)})")

    print(f"📁 Results saved to: {args.output}")
    return successful > 0

def _generate_batch_plots(output_dir, vtk_files, rt_analyzer):
    """Generate time series plots from batch results using RTAnalyzer's plotting methods."""

    try:
        if not vtk_files:
            print("   ⚠️  No files to process for plotting")
            return

        # For a single file, skip time series plotting
        if len(vtk_files) == 1:
            print(f"   📈 Single file processed - skipping time series plots")
            print(f"   💡 For time series plots, process multiple files with sequential time steps")
            return

        # For multiple files, extract time information and create a subset pattern
        print(f"   📈 Creating time series plots for {len(vtk_files)} files...")

        # Create a temporary file list for the specific files processed
        import tempfile
        temp_file_list = os.path.join(output_dir, 'processed_files.txt')
        with open(temp_file_list, 'w') as f:
            for vtk_file in vtk_files:
                f.write(f"{vtk_file}\n")

        # Read the comprehensive CSV results and create plots from that data
        comprehensive_csv = os.path.join(output_dir, 'batch_results_comprehensive.csv')
        if os.path.exists(comprehensive_csv):
            import pandas as pd
            df = pd.read_csv(comprehensive_csv)

            if len(df) >= 2:  # Need at least 2 points for a time series
                _create_simple_plots(output_dir, df, vtk_files)
            else:
                print(f"   💡 Need at least 2 time points for time series plots (found {len(df)})")
        else:
            print(f"   ⚠️  Comprehensive CSV not found: {comprehensive_csv}")

        # Clean up temp file
        if os.path.exists(temp_file_list):
            os.remove(temp_file_list)

    except Exception as e:
        print(f"   ❌ Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

def _create_simple_plots(output_dir, df, vtk_files):
    """Create simple plots from processed results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create plots directory
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Extract time values from filenames
        times = []
        for filename in df['filename']:
            # Extract time from filename like RT160x200-4999.vtk -> 49.99
            if '-' in filename and '.vtk' in filename:
                time_str = filename.split('-')[1].replace('.vtk', '')
                try:
                    time_val = float(time_str) / 100.0  # Convert 4999 -> 49.99
                    times.append(time_val)
                except ValueError:
                    times.append(len(times))  # Fallback to index
            else:
                times.append(len(times))

        # Plot fractal dimension evolution
        if 'fractal_dimension' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(times, df['fractal_dimension'], 'bo-', linewidth=2, markersize=6)
            plt.xlabel('Time')
            plt.ylabel('Fractal Dimension')
            plt.title('Fractal Dimension Evolution')
            plt.grid(True, alpha=0.3)

            # Add R² values as annotations for high-quality points
            # R² annotations removed for cleaner plots

            plt.tight_layout()
            fractal_plot = os.path.join(plots_dir, 'fractal_dimension_evolution.png')
            plt.savefig(fractal_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   📊 Fractal dimension plot: {fractal_plot}")

        print(f"   ✅ Basic plots created in: {plots_dir}")

    except Exception as e:
        print(f"   ❌ Plot creation failed: {e}")
        import traceback
        traceback.print_exc()

def show_info():
    """Show analyzer information and capabilities."""

    print(f"🌊 RT Analyzer with FastFractalAnalyzer Integration")
    print(f"=" * 60)

    try:
        from fractal_analyzer.core.rt_analyzer import RTAnalyzer

        # Initialize analyzer
        rt_analyzer = RTAnalyzer(output_dir='/tmp', no_titles=True)

        # Get capabilities
        caps = rt_analyzer.fractal_analyzer.get_capabilities()

        print(f"🎯 Fractal Analysis Capabilities:")
        for key, value in caps.items():
            print(f"   {key}: {value}")

        print(f"\n📊 Supported Analysis Types:")
        print(f"   • fractal_dim: Fractal dimension analysis")
        print(f"   • mixing: Mixing width and thickness analysis")

        print(f"\n🔧 Key Features:")
        print(f"   • 90× accuracy improvement for RT interfaces")
        print(f"   • Automatic method selection")
        print(f"   • R² values typically > 0.995")
        print(f"   • Grid optimization enabled")
        print(f"   • CONREC and PLIC interface extraction")

        return True

    except Exception as e:
        print(f"❌ Error getting analyzer info: {e}")
        return False

def main():
    """Main CLI entry point."""

    parser = setup_argument_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    success = False

    if args.command == 'analyze':
        success = analyze_single_file(args)
    elif args.command == 'batch':
        success = batch_process_files(args)
    elif args.command == 'info':
        success = show_info()
    else:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1

    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())