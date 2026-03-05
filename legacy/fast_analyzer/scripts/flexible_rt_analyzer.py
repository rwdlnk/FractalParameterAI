#!/usr/bin/env python3
"""
Flexible RT Analyzer with Wu-Enhanced Framework
Supports multiple input modes for VTK file processing.
"""

import os
import sys
import argparse
import glob
import re
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def parse_vtk_input(input_spec, base_directory=None):
    """
    Parse flexible VTK input specification.

    Args:
        input_spec: One of:
            - Directory path (processes all *.vtk files)
            - Comma-separated file list: "file1.vtk,file2.vtk,file3.vtk"
            - Pattern: "RT160x200-*.vtk" or "RT320x400-{0,1000,10000,18000}.vtk"
            - Time list: "0,1.0,10.0,18.0" (converts to VTK files)
        base_directory: Base directory for relative paths

    Returns:
        List of absolute VTK file paths
    """

    vtk_files = []

    # Set base directory
    if base_directory is None:
        base_directory = "/media/rod/ResearchII_III/svofRuns/Dalziel_1999/"

    # Case 1: Directory path
    if os.path.isdir(input_spec):
        pattern = os.path.join(input_spec, "*.vtk")
        vtk_files = glob.glob(pattern)
        print(f"📁 Directory mode: Found {len(vtk_files)} VTK files in {input_spec}")

    # Case 2: Comma-separated file list
    elif ',' in input_spec:
        file_list = [f.strip() for f in input_spec.split(',')]

        # Check if it's a time list (all numeric)
        if all(is_numeric(f) for f in file_list):
            print("🕐 Time list detected, converting to VTK files...")
            # Auto-detect resolution from base directory
            resolution = detect_resolution(base_directory)
            vtk_files = times_to_vtk_files(file_list, resolution, base_directory)
        else:
            # File list
            print(f"📄 File list mode: Processing {len(file_list)} specified files")
            for file_path in file_list:
                if not os.path.isabs(file_path):
                    file_path = os.path.join(base_directory, file_path)
                if os.path.exists(file_path):
                    vtk_files.append(file_path)
                else:
                    print(f"⚠️  File not found: {file_path}")

    # Case 3: Pattern matching
    elif '*' in input_spec or '{' in input_spec:
        if '{' in input_spec and '}' in input_spec:
            # Brace expansion: RT320x400-{0,1000,10000}.vtk
            vtk_files = expand_brace_pattern(input_spec, base_directory)
        else:
            # Glob pattern: RT160x200-*.vtk
            if not os.path.isabs(input_spec):
                pattern = os.path.join(base_directory, input_spec)
            else:
                pattern = input_spec
            vtk_files = glob.glob(pattern)
        print(f"🔍 Pattern mode: Found {len(vtk_files)} files matching pattern")

    # Case 4: Single file
    else:
        if not os.path.isabs(input_spec):
            input_spec = os.path.join(base_directory, input_spec)
        if os.path.exists(input_spec):
            vtk_files = [input_spec]
            print(f"📄 Single file mode: {input_spec}")
        else:
            print(f"❌ File not found: {input_spec}")

    # Sort files by time (extract numeric part from filename)
    vtk_files = sorted(vtk_files, key=extract_time_from_filename)

    print(f"✅ Total files to process: {len(vtk_files)}")
    if vtk_files:
        print(f"   First: {os.path.basename(vtk_files[0])}")
        print(f"   Last:  {os.path.basename(vtk_files[-1])}")

    return vtk_files

def is_numeric(s):
    """Check if string represents a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def detect_resolution(base_directory):
    """Auto-detect resolution from directory structure."""
    if "160x200" in base_directory:
        return "160x200"
    elif "320x400" in base_directory:
        return "320x400"
    else:
        # Try to detect from VTK files in directory
        vtk_files = glob.glob(os.path.join(base_directory, "**/*.vtk"), recursive=True)
        for vtk_file in vtk_files:
            if "160x200" in vtk_file:
                return "160x200"
            elif "320x400" in vtk_file:
                return "320x400"
    return "320x400"  # Default

def times_to_vtk_files(time_list, resolution, base_directory):
    """Convert time values to VTK file paths."""
    vtk_files = []

    # Find the appropriate subdirectory
    possible_dirs = [
        os.path.join(base_directory, f"{resolution}/slimMaster"),
        os.path.join(base_directory, "slimMaster"),
        base_directory
    ]

    vtk_dir = None
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            vtk_dir = dir_path
            break

    if not vtk_dir:
        print(f"❌ Could not find VTK directory for resolution {resolution}")
        return []

    for time_str in time_list:
        time_val = float(time_str)
        # Convert time to filename: t=1.0 → RT320x400-1000.vtk
        time_step = int(time_val * 1000)
        vtk_filename = f"RT{resolution}-{time_step}.vtk"
        vtk_path = os.path.join(vtk_dir, vtk_filename)

        if os.path.exists(vtk_path):
            vtk_files.append(vtk_path)
            print(f"  ✅ t={time_val} → {vtk_filename}")
        else:
            print(f"  ❌ t={time_val} → {vtk_filename} (not found)")

    return vtk_files

def expand_brace_pattern(pattern, base_directory):
    """Expand brace patterns like RT320x400-{0,1000,10000}.vtk"""
    # Extract the brace part
    brace_match = re.search(r'\{([^}]+)\}', pattern)
    if not brace_match:
        return []

    brace_content = brace_match.group(1)
    values = [v.strip() for v in brace_content.split(',')]

    vtk_files = []
    for value in values:
        expanded_pattern = pattern.replace(f"{{{brace_content}}}", value)
        if not os.path.isabs(expanded_pattern):
            expanded_pattern = os.path.join(base_directory, "**", expanded_pattern)
        matches = glob.glob(expanded_pattern, recursive=True)
        vtk_files.extend(matches)

    return vtk_files

def extract_time_from_filename(filepath):
    """Extract time value from VTK filename for sorting."""
    basename = os.path.basename(filepath)
    # Match patterns like RT320x400-1000.vtk → 1000
    match = re.search(r'-(\d+)\.vtk$', basename)
    if match:
        return int(match.group(1))
    return 0

def run_wu_enhanced_analysis(vtk_files, output_csv="wu_enhanced_results.csv"):
    """Run Wu-enhanced fractal analysis on VTK files."""
    try:
        from advanced_box_counting_prototype import AdvancedBoxCountingFramework
        from fractal_analyzer.io.vtk_reader import VTKReader
        from fractal_analyzer.core.conrec_extractor import CONRECExtractor
        import pandas as pd
        import time

        print(f"\n🚀 Starting Wu-enhanced analysis on {len(vtk_files)} files...")

        reader = VTKReader()
        extractor = CONRECExtractor()
        results = []

        for i, vtk_file in enumerate(vtk_files):
            print(f"\n📊 Processing {i+1}/{len(vtk_files)}: {os.path.basename(vtk_file)}")

            try:
                # Extract time from filename
                time_val = extract_time_from_filename(vtk_file) / 1000.0

                # Load VTK and extract interface
                start_time = time.time()
                vtk_data = reader.read_vtk_file(vtk_file)

                # Extract required data from VTK reader output
                volume_fraction = vtk_data['F']  # Volume fraction field
                x_grid = vtk_data['x']          # X coordinate grid
                y_grid = vtk_data['y']          # Y coordinate grid

                segment_list = extractor.extract_interface_conrec(volume_fraction, x_grid, y_grid)
                extraction_time = time.time() - start_time

                # Convert list of segments to SegmentArray
                from fractal_analyzer.core.data_types import SegmentArray
                segments = SegmentArray.from_list(segment_list)

                print(f"  ✅ Interface: {segments.n_segments} segments ({extraction_time:.1f}s)")

                # Wu-enhanced fractal analysis
                analysis_start = time.time()
                framework = AdvancedBoxCountingFramework(segments)
                analysis_results = framework.analyze_comprehensive(method="auto")
                analysis_time = time.time() - analysis_start

                # Find best result
                valid_results = [r for r in analysis_results if r.dimension and r.dimension == r.dimension]

                if valid_results:
                    best_result = max(valid_results, key=lambda x: x.r_squared)
                    print(f"  🏆 {best_result.method_name}: D = {best_result.dimension:.6f}, R² = {best_result.r_squared:.6f}")

                    results.append({
                        'filename': os.path.basename(vtk_file),
                        'time': time_val,
                        'n_segments': segments.n_segments,
                        'fractal_dimension': best_result.dimension,
                        'fractal_r_squared': best_result.r_squared,
                        'method_used': best_result.method_name,
                        'n_points_used': best_result.n_points,
                        'extraction_time': extraction_time,
                        'analysis_time': analysis_time
                    })
                else:
                    print(f"  ❌ All Wu methods failed")
                    results.append({
                        'filename': os.path.basename(vtk_file),
                        'time': time_val,
                        'n_segments': segments.n_segments,
                        'fractal_dimension': None,
                        'fractal_r_squared': None,
                        'method_used': 'FAILED',
                        'n_points_used': 0,
                        'extraction_time': extraction_time,
                        'analysis_time': analysis_time
                    })

            except Exception as e:
                print(f"  ❌ Error processing {vtk_file}: {e}")
                results.append({
                    'filename': os.path.basename(vtk_file),
                    'time': time_val if 'time_val' in locals() else 0,
                    'error': str(e)
                })

        # Save results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\n✅ Results saved to: {output_csv}")

            # Summary
            if 'fractal_dimension' in df.columns:
                valid_count = df['fractal_dimension'].notna().sum()
                print(f"\n📊 Analysis Summary:")
                print(f"   Total files: {len(vtk_files)}")
                print(f"   Successful: {valid_count}")
                print(f"   Failed: {len(vtk_files) - valid_count}")

                if valid_count > 0:
                    print(f"   Dimension range: {df['fractal_dimension'].min():.3f} - {df['fractal_dimension'].max():.3f}")
            else:
                print(f"\n📊 Analysis Summary:")
                print(f"   Total files: {len(vtk_files)}")
                print(f"   All analyses failed")

        return results

    except ImportError as e:
        print(f"❌ Wu-enhanced framework not available: {e}")
        print("   Please ensure you're on the wu-enhanced-framework branch")
        return []

def main():
    parser = argparse.ArgumentParser(description="Flexible RT Analyzer with Wu-Enhanced Framework")
    parser.add_argument("input", help="VTK input specification (see examples below)")
    parser.add_argument("--output", "-o", default="wu_enhanced_results.csv",
                       help="Output CSV file (default: wu_enhanced_results.csv)")
    parser.add_argument("--base-dir", "-d",
                       default="/media/rod/ResearchII_III/svofRuns/Dalziel_1999/",
                       help="Base directory for relative paths")

    parser.epilog = """
Examples:
  # Process all VTK files in directory
  python flexible_rt_analyzer.py /path/to/320x400/slimMaster/

  # Process specific time points
  python flexible_rt_analyzer.py "0,1.0,10.0,18.0"

  # Process specific files
  python flexible_rt_analyzer.py "RT320x400-0.vtk,RT320x400-1000.vtk,RT320x400-10000.vtk"

  # Use patterns
  python flexible_rt_analyzer.py "RT160x200-*.vtk"
  python flexible_rt_analyzer.py "RT320x400-{0,1000,10000,18000}.vtk"

  # Single file
  python flexible_rt_analyzer.py RT320x400-999.vtk
    """

    args = parser.parse_args()

    print("🌟 Wu-Enhanced RT Analyzer")
    print("=" * 50)

    # Parse input specification
    vtk_files = parse_vtk_input(args.input, args.base_dir)

    if not vtk_files:
        print("❌ No VTK files found matching the specification")
        return 1

    # Run analysis
    results = run_wu_enhanced_analysis(vtk_files, args.output)

    if results:
        print(f"\n🎉 Analysis complete! Results saved to {args.output}")
        return 0
    else:
        print(f"\n❌ Analysis failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())