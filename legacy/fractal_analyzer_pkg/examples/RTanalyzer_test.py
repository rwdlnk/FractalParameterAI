#!/usr/bin/env python3

"""
Test script for analysis_tools initialization
"""

import logging
from fractal_analyzer.applications.rt import RTAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Create analyzer
print("Creating RTAnalyzer...")
analyzer = RTAnalyzer("./output")

# Check if analysis_tools was initialized
print(f"fractal_analyzer exists: {hasattr(analyzer, 'fractal_analyzer')}")

if hasattr(analyzer, 'fractal_analyzer'):
    print(f"analysis_tools exists: {hasattr(analyzer.fractal_analyzer, 'analysis_tools')}")
    
    if hasattr(analyzer.fractal_analyzer, 'analysis_tools'):
        tools = analyzer.fractal_analyzer.analysis_tools
        print(f"analyze_linear_region exists: {hasattr(tools, 'analyze_linear_region')}")
        
        if hasattr(tools, 'fractal_analyzer'):
            print("Tools has correct 'fractal_analyzer' attribute")
        elif hasattr(tools, 'analyzer'):
            print("WARNING: Tools still has old 'analyzer' attribute!")
        else:
            print("WARNING: Tools missing both 'analyzer' and 'fractal_analyzer' attributes!")
    else:
        print("analysis_tools was not initialized!")
else:
    print("fractal_analyzer was not initialized!")

# Specify VTK file
vtk_file = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/800x800/RT800x800-7999.vtk"

# Test with linear region analysis
print("\nTesting with linear region analysis...")
try:
    result1 = analyzer.analyze_vtk_file(vtk_file, analyze_linear=True, trim_boundary=1)
    print(f"Fractal dimension (linear): {result1['fractal_dim']:.6f} ± {result1['fd_error']:.6f}")
except Exception as e:
    print(f"Error during linear analysis: {str(e)}")
    import traceback
    traceback.print_exc()

# Test with standard method
print("\nTesting with standard method...")
try:
    result2 = analyzer.analyze_vtk_file(vtk_file, analyze_linear=False)
    print(f"Fractal dimension (standard): {result2['fractal_dim']:.6f} ± {result2['fd_error']:.6f}")
except Exception as e:
    print(f"Error during standard analysis: {str(e)}")
    import traceback
    traceback.print_exc()

# Compare results
if 'fractal_dim' in result1 and 'fractal_dim' in result2:
    diff = abs(result1['fractal_dim'] - result2['fractal_dim'])
    print(f"\nDifference in dimensions: {diff:.6f}")
    
    if diff < 1e-6:
        print("WARNING: Both methods produced identical results!")
    else:
        print("SUCCESS: Different results from different methods!")
