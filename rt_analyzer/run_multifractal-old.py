#!/usr/bin/env python3
"""
Simple Multifractal Analysis
This script directly uses the RTAnalyzer class from the rt_analyzer.py file.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Try to import your RTAnalyzer class directly
try:
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Make sure current_dir is in the Python path
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Try to directly import the class from the file
    from rt_analyzer import RTAnalyzer
    print("Successfully imported RTAnalyzer")
except ImportError as e:
    print(f"Error importing RTAnalyzer: {e}")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir('.'))
    sys.exit(1)

# Configuration
DATA_DIR = "../../800x800"  # Path to your data directory
OUTPUT_DIR = "./results/multifractal_800"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the analyzer
analyzer = RTAnalyzer(OUTPUT_DIR)

# Define the VTK file path
vtk_file = os.path.join(DATA_DIR, "RT800x800-9000.vtk")
print(f"Reading VTK file: {vtk_file}")

# Read the VTK file
try:
    data = analyzer.read_vtk_file(vtk_file)
    print("VTK file read successfully")
except Exception as e:
    print(f"Error reading VTK file: {str(e)}")
    sys.exit(1)

# Define q-values for multifractal analysis
q_values = np.arange(-5, 5.1, 1.0)

# Perform multifractal analysis
print("Performing multifractal analysis...")
try:
    result = analyzer.compute_multifractal_spectrum(
        data, 
        q_values=q_values, 
        output_dir=OUTPUT_DIR
    )
    print("Multifractal analysis completed successfully")
except Exception as e:
    print(f"Error during multifractal analysis: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"Analysis complete. Results have been saved to {OUTPUT_DIR}")
