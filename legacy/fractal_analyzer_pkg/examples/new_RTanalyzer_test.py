#!/usr/bin/env python3

"""
Detailed test script for RT linear region analysis
"""

import traceback
import numpy as np
from fractal_analyzer.applications.rt import RTAnalyzer

print("\n==== RT Linear Region Analysis Test ====\n")

# Create analyzer
analyzer = RTAnalyzer("./output")

# Check analyzer setup
print("Checking RTAnalyzer setup:")
print(f"- fractal_analyzer exists: {hasattr(analyzer, 'fractal_analyzer')}")

if hasattr(analyzer, 'fractal_analyzer'):
    fa = analyzer.fractal_analyzer
    print(f"- fractal_analyzer has analysis_tools: {hasattr(fa, 'analysis_tools')}")
    
    if hasattr(fa, 'analysis_tools'):
        tools = fa.analysis_tools
        print(f"- analysis_tools has analyze_linear_region: {hasattr(tools, 'analyze_linear_region')}")
        
        if hasattr(tools, 'analyze_linear_region'):
            # Check analysis_tools attribute names
            print("\nChecking analysis_tools attribute names:")
            has_analyzer = hasattr(tools, 'analyzer')
            has_fractal_analyzer = hasattr(tools, 'fractal_analyzer')
            print(f"- Has 'analyzer' attribute: {has_analyzer}")
            print(f"- Has 'fractal_analyzer' attribute: {has_fractal_analyzer}")
            
            if has_analyzer:
                print("WARNING: 'analyzer' attribute still exists! It should be renamed to 'fractal_analyzer'.")
            
            if not has_fractal_analyzer:
                print("WARNING: 'fractal_analyzer' attribute does not exist! Did you rename 'analyzer' to 'fractal_analyzer'?")

# Specify VTK file
vtk_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/800x800/RT800x800-7999.vtk"

# First, just read the file
print("\nReading VTK file...")
try:
    data = analyzer.read_vtk_file(vtk_file)
    print(f"VTK file read successfully. Time: {data['time']}")
except Exception as e:
    print(f"Error reading VTK file: {str(e)}")
    traceback.print_exc()
    exit(1)

# Run direct tests on compute_fractal_dimension
print("\n==== Direct Method Comparison ====\n")

# Run with linear region analysis
print("Testing compute_fractal_dimension with linear region analysis...")
try:
    result_linear = analyzer.compute_fractal_dimension(data, analyze_linear=True, trim_boundary=1)
    print(f"Result with linear region: D = {result_linear['dimension']:.6f} ± {result_linear['error']:.6f} (R² = {result_linear['r_squared']:.6f})")
except Exception as e:
    print(f"Error during linear region analysis: {str(e)}")
    traceback.print_exc()
    result_linear = None

# Run with standard method
print("\nTesting compute_fractal_dimension with standard method...")
try:
    result_standard = analyzer.compute_fractal_dimension(data, analyze_linear=False)
    print(f"Result with standard method: D = {result_standard['dimension']:.6f} ± {result_standard['error']:.6f} (R² = {result_standard['r_squared']:.6f})")
except Exception as e:
    print(f"Error during standard method: {str(e)}")
    traceback.print_exc()
    result_standard = None

# Compare results
if result_linear and result_standard:
    diff = abs(result_linear['dimension'] - result_standard['dimension'])
    print(f"\nDifference in dimensions: {diff:.6f}")
    
    if diff < 1e-6:
        print("WARNING: Both methods produced identical results!")
        print("This suggests that linear region analysis is not being used correctly.")
        
        # Check for common issues
        print("\nPossible issues:")
        if not hasattr(analyzer.fractal_analyzer, 'analysis_tools'):
            print("- analysis_tools is not set on fractal_analyzer")
        elif hasattr(analyzer.fractal_analyzer.analysis_tools, 'analyzer'):
            print("- analysis_tools still uses 'analyzer' attribute instead of 'fractal_analyzer'")
    else:
        print("SUCCESS: Different results obtained from different methods!")
