#!/usr/bin/env python3
"""
Fault and Fracture Data Downloader

Downloads geological fault/fracture data and converts to segment format
for fractal box-counting analysis.

Requirements:
    pip install geopandas shapely requests
"""

import os
import zipfile
import requests
import geopandas as gpd
import numpy as np
from pathlib import Path


def download_usgs_quaternary_faults(region='california'):
    """
    Download USGS Quaternary Fault data.
    
    Note: This uses a simplified approach. For full USGS data,
    visit: https://www.usgs.gov/programs/earthquake-hazards/faults
    
    For this example, we'll use a publicly available simplified dataset.
    """
    
    print("="*70)
    print("USGS QUATERNARY FAULT DATA")
    print("="*70)
    print("\nNote: For comprehensive USGS fault data, visit:")
    print("https://www.usgs.gov/programs/earthquake-hazards/faults")
    print("\nThis script provides example access to public fault datasets.")
    print("="*70)
    
    # Example: California faults (simplified version available publicly)
    # For production use, download from official USGS sources
    
    print(f"\nTo download California fault data:")
    print("1. Visit: https://maps.conservation.ca.gov/cgs/DataViewer/")
    print("2. Download 'Quaternary Faults' layer")
    print("3. Save as shapefile")
    print("\nOr use the USGS Quaternary Fault Database:")
    print("https://www.usgs.gov/natural-hazards/earthquake-hazards/faults")
    
    return None


def create_synthetic_fracture_network(num_fractures=100, domain_size=10.0, 
                                     output_file='synthetic_fractures.txt'):
    """
    Create a synthetic fracture network with realistic fractal properties.
    
    This generates a power-law distributed fracture network similar to 
    natural rock fractures.
    
    Parameters:
    -----------
    num_fractures : int
        Number of fractures to generate
    domain_size : float
        Size of the domain (square)
    output_file : str
        Output filename
    """
    
    print(f"\nGenerating synthetic fracture network...")
    print(f"Number of fractures: {num_fractures}")
    print(f"Domain: {domain_size} × {domain_size}")
    
    segments = []
    
    # Generate fractures with power-law length distribution
    # (characteristic of natural fracture networks)
    np.random.seed(42)
    
    # Power-law exponent for length distribution
    alpha = 2.5
    min_length = domain_size * 0.05
    max_length = domain_size * 0.5
    
    for i in range(num_fractures):
        # Power-law distributed lengths
        u = np.random.random()
        length = min_length * (1 - u + u * (max_length/min_length)**(1-alpha))**(1/(1-alpha))
        
        # Random position
        x_center = np.random.random() * domain_size
        y_center = np.random.random() * domain_size
        
        # Random orientation (with some preferred orientations)
        # Real fractures often have preferred orientations (joint sets)
        if np.random.random() < 0.3:
            # Preferential orientation 1 (e.g., N-S)
            angle = np.random.normal(0, 0.2)
        elif np.random.random() < 0.5:
            # Preferential orientation 2 (e.g., E-W)
            angle = np.random.normal(np.pi/2, 0.2)
        else:
            # Random orientation
            angle = np.random.random() * np.pi
        
        # Calculate endpoints
        dx = length * np.cos(angle) / 2
        dy = length * np.sin(angle) / 2
        
        x1 = x_center - dx
        y1 = y_center - dy
        x2 = x_center + dx
        y2 = y_center + dy
        
        # Keep only fractures within domain
        if (0 <= x1 <= domain_size and 0 <= x2 <= domain_size and
            0 <= y1 <= domain_size and 0 <= y2 <= domain_size):
            segments.append([x1, y1, x2, y2])
    
    segments = np.array(segments)
    
    # Calculate statistics
    lengths = np.sqrt((segments[:, 2] - segments[:, 0])**2 + 
                     (segments[:, 3] - segments[:, 1])**2)
    
    print(f"Generated {len(segments)} fractures")
    print(f"Mean length: {lengths.mean():.4f}")
    print(f"Length range: [{lengths.min():.4f}, {lengths.max():.4f}]")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("# Synthetic fracture network for box-counting analysis\n")
        f.write("# Generated with power-law length distribution\n")
        f.write(f"# Number of fractures: {len(segments)}\n")
        f.write(f"# Domain: [0, {domain_size}] × [0, {domain_size}]\n")
        f.write(f"# Expected fractal dimension: ~1.4-1.6\n")
        f.write("# Format: x1 y1 x2 y2\n")
        f.write("#\n")
        
        for seg in segments:
            f.write(f"{seg[0]:.6f} {seg[1]:.6f} {seg[2]:.6f} {seg[3]:.6f}\n")
    
    print(f"\nSaved to: {output_file}")
    
    metadata = {
        'num_segments': len(segments),
        'domain': (0, domain_size, 0, domain_size),
        'mean_length': lengths.mean(),
        'length_range': (lengths.min(), lengths.max())
    }
    
    return segments, metadata


def create_example_joint_set(spacing=1.0, domain_size=10.0, num_sets=2,
                             output_file='joint_pattern.txt'):
    """
    Create an example orthogonal joint set pattern.
    
    Common in sedimentary rocks - two perpendicular sets of parallel fractures.
    
    Parameters:
    -----------
    spacing : float
        Average spacing between parallel joints
    domain_size : float
        Size of domain
    num_sets : int
        Number of joint sets (typically 2 or 3)
    output_file : str
        Output filename
    """
    
    print(f"\nGenerating orthogonal joint set pattern...")
    print(f"Joint spacing: {spacing}")
    print(f"Domain: {domain_size} × {domain_size}")
    print(f"Number of joint sets: {num_sets}")
    
    segments = []
    
    # Add some randomness to spacing (natural variation)
    np.random.seed(42)
    
    angles = np.linspace(0, np.pi, num_sets, endpoint=False)
    
    for angle in angles:
        # Determine how many joints needed
        num_joints = int(domain_size * 1.5 / spacing)
        
        for i in range(num_joints):
            # Position with some randomness
            offset = (i - num_joints/2) * spacing * (1 + np.random.normal(0, 0.1))
            
            # Create joint line across domain
            if abs(np.cos(angle)) > abs(np.sin(angle)):
                # More horizontal
                x1 = 0
                y1 = domain_size/2 + offset * np.sin(angle)
                x2 = domain_size
                y2 = y1 + domain_size * np.tan(angle)
            else:
                # More vertical
                y1 = 0
                x1 = domain_size/2 + offset * np.cos(angle)
                y2 = domain_size
                x2 = x1 + domain_size / np.tan(angle) if np.tan(angle) != 0 else x1
            
            # Clip to domain
            if 0 <= y1 <= domain_size and 0 <= y2 <= domain_size:
                # Add some segment length variation
                length = np.random.uniform(0.6, 1.0)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                dx = (x2 - x1) * length / 2
                dy = (y2 - y1) * length / 2
                
                segments.append([cx - dx, cy - dy, cx + dx, cy + dy])
    
    segments = np.array(segments)
    
    # Filter to domain bounds
    mask = ((segments[:, 0] >= 0) & (segments[:, 0] <= domain_size) &
            (segments[:, 2] >= 0) & (segments[:, 2] <= domain_size) &
            (segments[:, 1] >= 0) & (segments[:, 1] <= domain_size) &
            (segments[:, 3] >= 0) & (segments[:, 3] <= domain_size))
    
    segments = segments[mask]
    
    print(f"Generated {len(segments)} joint traces")
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("# Orthogonal joint set pattern\n")
        f.write("# Simulates systematic jointing in sedimentary rocks\n")
        f.write(f"# Number of joint sets: {num_sets}\n")
        f.write(f"# Joint spacing: {spacing}\n")
        f.write(f"# Domain: [0, {domain_size}] × [0, {domain_size}]\n")
        f.write(f"# Expected dimension: ~1.0 (regular pattern)\n")
        f.write("# Format: x1 y1 x2 y2\n")
        f.write("#\n")
        
        for seg in segments:
            f.write(f"{seg[0]:.6f} {seg[1]:.6f} {seg[2]:.6f} {seg[3]:.6f}\n")
    
    print(f"Saved to: {output_file}")
    
    return segments


def main():
    """Main function with menu."""
    
    print("="*70)
    print("ROCK FRACTURE DATA GENERATOR")
    print("="*70)
    print("\nThis tool generates synthetic fracture patterns for testing")
    print("box-counting fractal analysis on geological structures.")
    print()
    print("Available datasets:")
    print("  1. Synthetic fracture network (power-law distributed)")
    print("  2. Orthogonal joint set (systematic jointing)")
    print("  3. Information about real fault data sources")
    print()
    
    choice = input("Choose option [1]: ").strip() or "1"
    
    if choice == "1":
        print("\n" + "="*70)
        print("SYNTHETIC FRACTURE NETWORK")
        print("="*70)
        
        num_fractures_input = input("Number of fractures [100]: ").strip()
        num_fractures = int(num_fractures_input) if num_fractures_input else 100
        
        domain_size_input = input("Domain size [10.0]: ").strip()
        domain_size = float(domain_size_input) if domain_size_input else 10.0
        
        segments, metadata = create_synthetic_fracture_network(
            num_fractures=num_fractures,
            domain_size=domain_size,
            output_file='synthetic_fractures.txt'
        )
        
        print("\n" + "="*70)
        print("SUGGESTED ANALYSIS:")
        print("="*70)
        print("\npython simple_analyzer.py synthetic_fractures.txt \\")
        print(f"  --domain-x 0 {domain_size} \\")
        print(f"  --domain-y 0 {domain_size} \\")
        print(f"  --initial-delta {domain_size/10:.2f} \\")
        print("  --delta-factor 1.5 \\")
        print("  --num-steps 12")
        print("\nExpected dimension: 1.4 - 1.6 (typical for fracture networks)")
        
    elif choice == "2":
        print("\n" + "="*70)
        print("ORTHOGONAL JOINT SET")
        print("="*70)
        
        spacing_input = input("Joint spacing [1.0]: ").strip()
        spacing = float(spacing_input) if spacing_input else 1.0
        
        domain_size_input = input("Domain size [10.0]: ").strip()
        domain_size = float(domain_size_input) if domain_size_input else 10.0
        
        num_sets_input = input("Number of joint sets [2]: ").strip()
        num_sets = int(num_sets_input) if num_sets_input else 2
        
        segments = create_example_joint_set(
            spacing=spacing,
            domain_size=domain_size,
            num_sets=num_sets,
            output_file='joint_pattern.txt'
        )
        
        print("\n" + "="*70)
        print("SUGGESTED ANALYSIS:")
        print("="*70)
        print("\npython simple_analyzer.py joint_pattern.txt \\")
        print(f"  --domain-x 0 {domain_size} \\")
        print(f"  --domain-y 0 {domain_size} \\")
        print(f"  --initial-delta {domain_size/10:.2f} \\")
        print("  --delta-factor 1.5 \\")
        print("  --num-steps 10")
        print("\nExpected dimension: ~1.0 (regular pattern)")
        print("Note: Regular joint patterns have low fractal dimension")
        
    elif choice == "3":
        print("\n" + "="*70)
        print("REAL FAULT DATA SOURCES")
        print("="*70)
        print("\n1. USGS Quaternary Fault Database:")
        print("   https://www.usgs.gov/natural-hazards/earthquake-hazards/faults")
        print("   - Download shapefiles")
        print("   - Contains fault traces for USA")
        print("   - Expected D: 1.3 - 1.6")
        
        print("\n2. California Geological Survey:")
        print("   https://maps.conservation.ca.gov/cgs/DataViewer/")
        print("   - High-resolution fault maps")
        print("   - San Andreas system")
        
        print("\n3. OpenTopography:")
        print("   https://opentopography.org/")
        print("   - LiDAR data of rock faces")
        print("   - Extract fracture traces from DEMs")
        
        print("\n4. British Geological Survey:")
        print("   https://www.bgs.ac.uk/datasets/")
        print("   - UK fault data")
        
        print("\n5. Research Repositories:")
        print("   - Zenodo: https://zenodo.org/ (search 'fracture network')")
        print("   - Pangaea: https://www.pangaea.de/")
        print("   - Dryad: https://datadryad.org/")
        
        print("\nTo convert shapefile to segment format:")
        print("Use the parse_segment_file() function in simple_analyzer.py")
        print("or geopandas to extract coordinates.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
