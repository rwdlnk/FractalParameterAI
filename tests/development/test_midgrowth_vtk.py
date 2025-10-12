#!/usr/bin/env python3
"""
Test Advanced Characterization System on Mid-Growth RT VTK File
Extracts interface from RT160x200-5999.vtk and applies AI-enhanced characterization
"""

import sys
import os
import time
import numpy as np

# Add necessary paths
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')

def main():
    print("🧬 AI-Enhanced Mid-Growth RT Interface Analysis")
    print("=" * 60)

    vtk_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer/data/Dalziel_1999/slimMaster/RT160x200-5999.vtk"

    # Check if VTK file exists
    if not os.path.exists(vtk_file):
        print(f"❌ VTK file not found: {vtk_file}")
        return 1

    print(f"📁 Processing: {os.path.basename(vtk_file)}")
    print(f"📊 Mid-growth time: {5999/1000:.1f}s")

    try:
        # Step 1: Extract interface using CONREC
        print("\n🔍 Step 1: Interface Extraction using CONREC")
        start_time = time.time()

        from fractal_analyzer.io.vtk_reader import VTKReader
        from fractal_analyzer.core.conrec_extractor import CONRECExtractor
        from fractal_analyzer.core.data_types import SegmentArray

        # Read VTK file
        reader = VTKReader()
        print("   📖 Reading VTK file...")
        vtk_data = reader.read_vtk_file(vtk_file)

        # Extract required data fields
        volume_fraction = vtk_data['F']  # Volume fraction field
        x_grid = vtk_data['x']          # X coordinate grid
        y_grid = vtk_data['y']          # Y coordinate grid

        print(f"   📏 Grid dimensions: {volume_fraction.shape}")
        print(f"   📐 Domain: x={x_grid.min():.3f}-{x_grid.max():.3f}m, y={y_grid.min():.3f}-{y_grid.max():.3f}m")

        # Extract interface at 0.5 contour level
        extractor = CONRECExtractor()
        print("   🌊 Extracting 0.5 contour level...")
        segment_list = extractor.extract_interface_conrec(volume_fraction, x_grid, y_grid, 0.5)

        # Convert to SegmentArray
        segments = SegmentArray.from_list(segment_list)
        extraction_time = time.time() - start_time

        print(f"   ✅ Interface extracted: {segments.n_segments} segments ({extraction_time:.1f}s)")
        print(f"   📊 Interface bounds: x={segments.bbox.min_x:.3f}-{segments.bbox.max_x:.3f}m")
        print(f"                        y={segments.bbox.min_y:.3f}-{segments.bbox.max_y:.3f}m")

        # Step 2: Apply Advanced Characterization System
        print("\n🤖 Step 2: AI-Enhanced Advanced Characterization")

        # Import advanced characterization modules
        sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI/core')
        from advanced_characterization import AdvancedCharacterizationFramework

        # Initialize framework
        framework = AdvancedCharacterizationFramework(segments)

        # Compute domain scale (for parameter scaling)
        domain_x = segments.bbox.max_x - segments.bbox.min_x
        domain_y = segments.bbox.max_y - segments.bbox.min_y
        domain_scale = max(domain_x, domain_y)

        print(f"   📏 Domain scale: {domain_scale:.3f}m")

        # Run comprehensive characterization
        analysis_start = time.time()
        results = framework.analyze_comprehensive(domain_scale=domain_scale)
        analysis_time = time.time() - analysis_start

        print(f"\n📊 Advanced Characterization Results ({analysis_time:.1f}s):")
        print("   " + "="*50)

        # Display individual method results
        for method_name, result in results.items():
            if result['confidence'] > 0:
                print(f"   🔬 {method_name.title()}:")
                print(f"      Parameters: δ₀={result['parameters']['initial_delta']:.6f}, factor={result['parameters']['delta_factor']:.3f}, steps={result['parameters']['num_steps']}")
                print(f"      Confidence: {result['confidence']:.1f}%")
                print(f"      Features: {len(result['features'])} extracted")
            else:
                print(f"   ❌ {method_name.title()}: Failed")

        # Get consensus parameters
        consensus = framework.get_consensus_parameters()
        print(f"\n🎯 Consensus Parameters:")
        print(f"   δ₀ = {consensus['parameters']['initial_delta']:.6f}")
        print(f"   factor = {consensus['parameters']['delta_factor']:.3f}")
        print(f"   steps = {consensus['parameters']['num_steps']}")
        print(f"   Overall confidence: {consensus['overall_confidence']:.1f}%")

        # Step 3: Apply Box Counting with AI-suggested parameters
        print(f"\n📐 Step 3: Box Counting with AI Parameters")

        from fractal_analyzer.core.fractal_analyzer import FastFractalAnalyzer

        analyzer = FastFractalAnalyzer()

        # Use AI-suggested parameters
        ai_params = consensus['parameters']

        bc_start = time.time()
        result = analyzer.analyze_segments_with_params(
            segments,
            initial_delta=ai_params['initial_delta'],
            delta_factor=ai_params['delta_factor'],
            num_steps=int(ai_params['num_steps'])
        )
        bc_time = time.time() - bc_start

        if result and hasattr(result, 'dimension'):
            print(f"   ✅ Fractal dimension: {result.dimension:.6f}")
            print(f"   📊 R² = {result.r_squared:.6f}")
            print(f"   ⏱️  Analysis time: {bc_time:.1f}s")

            # Compare with user's visual estimate (1.2-1.3)
            visual_estimate_center = 1.25
            error = abs(result.dimension - visual_estimate_center)
            error_pct = error / visual_estimate_center * 100

            print(f"\n🎯 Comparison with Visual Estimate:")
            print(f"   Visual estimate: 1.2 - 1.3 (center: {visual_estimate_center})")
            print(f"   AI prediction: {result.dimension:.6f}")
            print(f"   Error: ±{error:.6f} ({error_pct:.2f}%)")

            if 1.2 <= result.dimension <= 1.3:
                print(f"   ✅ Result within visual estimate range!")
            else:
                print(f"   ⚠️  Result outside visual estimate range")
        else:
            print(f"   ❌ Box counting analysis failed")

        # Step 4: Save detailed results
        print(f"\n💾 Step 4: Saving Results")

        results_data = {
            'vtk_file': os.path.basename(vtk_file),
            'time_s': 5999/1000.0,
            'n_segments': segments.n_segments,
            'extraction_time': extraction_time,
            'analysis_time': analysis_time,
            'bc_time': bc_time,
            'domain_scale': domain_scale,
            'advanced_results': results,
            'consensus': consensus
        }

        if result and hasattr(result, 'dimension'):
            results_data['fractal_dimension'] = result.dimension
            results_data['r_squared'] = result.r_squared

        # Save to file
        import json
        output_file = "midgrowth_rt_advanced_results.json"
        with open(output_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Convert all numpy types
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)

            clean_data = recursive_convert(results_data)
            json.dump(clean_data, f, indent=2)

        print(f"   ✅ Results saved to: {output_file}")

        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 Analysis Complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Interface complexity: {segments.n_segments} segments")
        if result and hasattr(result, 'dimension'):
            print(f"   Fractal dimension: {result.dimension:.6f} ± {error:.6f}")
            print(f"   Quality: R² = {result.r_squared:.6f}")
        print(f"   AI confidence: {consensus['overall_confidence']:.1f}%")

        return 0

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())