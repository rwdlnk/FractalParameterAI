#!/usr/bin/env python3
"""
Standalone Test of Mid-Growth RT Interface with Advanced Characterization
Extracts interface from RT160x200-5999.vtk and applies AI-enhanced characterization
"""

import sys
import os
import time
import numpy as np

# Add necessary paths
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FastFractalAnalyzer')
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI')
sys.path.append('/media/rod/ResearchII_III/ResearchIII/githubRepos/FractalParameterAI/core')

def main():
    print("🧬 AI-Enhanced Mid-Growth RT Interface Analysis (Standalone)")
    print("=" * 65)

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

        # Step 2: Apply Advanced Characterization System (Standalone imports)
        print("\n🤖 Step 2: AI-Enhanced Advanced Characterization")

        # Import individual modules with absolute imports
        try:
            from power_spectrum_features import extract_power_spectrum_features, suggest_parameters_from_spectrum
            print("   ✅ Power spectrum module loaded")
            power_spectrum_available = True
        except ImportError as e:
            print(f"   ⚠️  Power spectrum module not available: {e}")
            power_spectrum_available = False

        try:
            from wavelet_features import extract_wavelet_features, suggest_parameters_from_wavelet
            print("   ✅ Wavelet module loaded")
            wavelet_available = True
        except ImportError as e:
            print(f"   ⚠️  Wavelet module not available: {e}")
            wavelet_available = False

        try:
            from curvature_features import extract_curvature_features, suggest_parameters_from_curvature
            print("   ✅ Curvature module loaded")
            curvature_available = True
        except ImportError as e:
            print(f"   ⚠️  Curvature module not available: {e}")
            curvature_available = False

        try:
            from self_similarity_features import extract_self_similarity_features, suggest_parameters_from_self_similarity
            print("   ✅ Self-similarity module loaded")
            self_similarity_available = True
        except ImportError as e:
            print(f"   ⚠️  Self-similarity module not available: {e}")
            self_similarity_available = False

        # Compute domain scale (for parameter scaling)
        domain_x = segments.bbox.max_x - segments.bbox.min_x
        domain_y = segments.bbox.max_y - segments.bbox.min_y
        domain_scale = max(domain_x, domain_y)

        print(f"   📏 Domain scale: {domain_scale:.3f}m")

        # Run individual characterization methods
        analysis_start = time.time()
        results = {}

        # Power Spectrum Analysis
        if power_spectrum_available:
            try:
                print("   🔬 Running power spectrum analysis...")
                ps_features = extract_power_spectrum_features(segment_list)
                ps_params = suggest_parameters_from_spectrum(ps_features, domain_scale)
                results['power_spectrum'] = {
                    'parameters': ps_params,
                    'features': ps_features,
                    'confidence': 70.0  # Based on previous results
                }
                print(f"      δ₀={ps_params['initial_delta']:.6f}, factor={ps_params['delta_factor']:.3f}, steps={ps_params['num_steps']}")
            except Exception as e:
                print(f"      ❌ Power spectrum failed: {e}")
                results['power_spectrum'] = {'confidence': 0}

        # Wavelet Analysis
        if wavelet_available:
            try:
                print("   🌊 Running wavelet analysis...")
                wav_features = extract_wavelet_features(segment_list)
                wav_params = suggest_parameters_from_wavelet(wav_features, domain_scale)
                results['wavelet'] = {
                    'parameters': wav_params,
                    'features': wav_features,
                    'confidence': 100.0  # Based on previous results
                }
                print(f"      δ₀={wav_params['initial_delta']:.6f}, factor={wav_params['delta_factor']:.3f}, steps={wav_params['num_steps']}")
            except Exception as e:
                print(f"      ❌ Wavelet failed: {e}")
                results['wavelet'] = {'confidence': 0}

        # Curvature Analysis
        if curvature_available:
            try:
                print("   📐 Running curvature analysis...")
                curv_features = extract_curvature_features(segment_list)
                curv_params = suggest_parameters_from_curvature(curv_features, domain_scale)
                results['curvature'] = {
                    'parameters': curv_params,
                    'features': curv_features,
                    'confidence': 100.0  # Based on previous results
                }
                print(f"      δ₀={curv_params['initial_delta']:.6f}, factor={curv_params['delta_factor']:.3f}, steps={curv_params['num_steps']}")
            except Exception as e:
                print(f"      ❌ Curvature failed: {e}")
                results['curvature'] = {'confidence': 0}

        # Self-Similarity Analysis
        if self_similarity_available:
            try:
                print("   📊 Running self-similarity analysis...")
                ss_features = extract_self_similarity_features(segment_list)
                ss_params = suggest_parameters_from_self_similarity(ss_features, domain_scale)
                results['self_similarity'] = {
                    'parameters': ss_params,
                    'features': ss_features,
                    'confidence': 16.2  # Based on previous results
                }
                print(f"      δ₀={ss_params['initial_delta']:.6f}, factor={ss_params['delta_factor']:.3f}, steps={ss_params['num_steps']}")
            except Exception as e:
                print(f"      ❌ Self-similarity failed: {e}")
                results['self_similarity'] = {'confidence': 0}

        analysis_time = time.time() - analysis_start

        # Calculate consensus parameters
        print(f"\n🎯 Consensus Parameters:")

        # Collect valid results
        valid_results = [(name, res) for name, res in results.items() if res.get('confidence', 0) > 0]

        if valid_results:
            # Weighted average based on confidence
            total_weight = sum(res['confidence'] for _, res in valid_results)

            consensus_delta = sum(res['parameters']['initial_delta'] * res['confidence']
                                for _, res in valid_results) / total_weight
            consensus_factor = sum(res['parameters']['delta_factor'] * res['confidence']
                                 for _, res in valid_results) / total_weight
            consensus_steps = sum(res['parameters']['num_steps'] * res['confidence']
                                for _, res in valid_results) / total_weight

            overall_confidence = total_weight / 4  # 4 methods maximum

            consensus = {
                'parameters': {
                    'initial_delta': consensus_delta,
                    'delta_factor': consensus_factor,
                    'num_steps': int(consensus_steps)
                },
                'overall_confidence': overall_confidence
            }
        else:
            # Fallback parameters
            consensus = {
                'parameters': {
                    'initial_delta': domain_scale / 100,
                    'delta_factor': 1.5,
                    'num_steps': 15
                },
                'overall_confidence': 0.0
            }

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

            # Compare with early growth result (1.215545)
            early_growth_dimension = 1.215545
            growth_comparison = result.dimension - early_growth_dimension

            print(f"\n📈 Comparison with Early Growth (t=3s):")
            print(f"   Early growth (t=3s): {early_growth_dimension:.6f}")
            print(f"   Mid-growth (t=6s): {result.dimension:.6f}")
            print(f"   Change: {growth_comparison:+.6f}")

            if growth_comparison > 0:
                print(f"   🔺 Interface became MORE complex with time")
            elif growth_comparison < 0:
                print(f"   🔻 Interface became LESS complex with time")
            else:
                print(f"   ➡️  Interface complexity remained stable")

        else:
            print(f"   ❌ Box counting analysis failed")

        # Summary
        total_time = time.time() - start_time
        print(f"\n🎉 Analysis Complete!")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Interface complexity: {segments.n_segments} segments")
        if result and hasattr(result, 'dimension'):
            print(f"   Fractal dimension: {result.dimension:.6f}")
            print(f"   Quality: R² = {result.r_squared:.6f}")
        print(f"   AI confidence: {consensus['overall_confidence']:.1f}%")
        print(f"   Methods available: {len(valid_results)}/4")

        return 0

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())