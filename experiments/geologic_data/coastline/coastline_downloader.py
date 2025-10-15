#!/usr/bin/env python3
"""
Coastline Fractal Data Downloader and Converter

Downloads coastline data from Natural Earth and converts it to 
segment format for box-counting fractal analysis.

Requirements:
    pip install geopandas shapely requests
"""

import os
import zipfile
import requests
import geopandas as gpd
import numpy as np
from pathlib import Path


def download_natural_earth_coastlines(resolution='50m'):
    """
    Download Natural Earth coastline data.
    
    Parameters:
    -----------
    resolution : str
        '10m' (high detail, ~50MB), '50m' (medium, ~5MB), or '110m' (low, ~1MB)
    
    Returns:
    --------
    str : Path to the downloaded shapefile
    """
    
    # URLs for different resolutions
    urls = {
        '10m': 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip',
        '50m': 'https://naciscdn.org/naturalearth/50m/physical/ne_50m_coastline.zip',
        '110m': 'https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip'
    }
    
    if resolution not in urls:
        raise ValueError(f"Resolution must be one of: {list(urls.keys())}")
    
    url = urls[resolution]
    zip_filename = f'ne_{resolution}_coastline.zip'
    
    print(f"Downloading {resolution} resolution coastline data...")
    print(f"URL: {url}")
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                progress = (downloaded / total_size) * 100
                print(f'\rProgress: {progress:.1f}%', end='')
    
    print(f'\nDownload complete: {zip_filename}')
    
    # Extract the zip file
    extract_dir = f'ne_{resolution}_coastline'
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print(f'Extracted to: {extract_dir}')
    
    # Return the path to the shapefile
    shapefile = os.path.join(extract_dir, f'ne_{resolution}_coastline.shp')
    return shapefile


def extract_segments_from_shapefile(shapefile_path, bbox=None, max_segments=2000):
    """
    Extract line segments from shapefile.
    
    Parameters:
    -----------
    shapefile_path : str
        Path to the shapefile
    bbox : tuple or None
        (min_lon, min_lat, max_lon, max_lat) to filter region
    max_segments : int
        Maximum number of segments to extract
    
    Returns:
    --------
    numpy.ndarray : Array of segments [x1, y1, x2, y2]
    dict : Metadata about the extraction
    """
    
    print(f"\nReading shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    print(f"Total features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    
    # Filter by bounding box if provided
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
        print(f"Filtered to bounding box: {len(gdf)} features")
    
    # Extract all segments
    all_segments = []
    total_points = 0
    
    for idx, geom in enumerate(gdf.geometry):
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            total_points += len(coords)
            
            for i in range(len(coords) - 1):
                x1, y1 = coords[i]
                x2, y2 = coords[i + 1]
                all_segments.append([x1, y1, x2, y2])
        
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                total_points += len(coords)
                
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    all_segments.append([x1, y1, x2, y2])
    
    segments = np.array(all_segments)
    
    print(f"\nExtracted {len(segments)} total segments from {total_points} points")
    
    # Subsample if too many segments
    if len(segments) > max_segments:
        print(f"Subsampling to {max_segments} segments...")
        indices = np.linspace(0, len(segments) - 1, max_segments, dtype=int)
        segments = segments[indices]
    
    # Calculate bounds
    x_coords = np.concatenate([segments[:, 0], segments[:, 2]])
    y_coords = np.concatenate([segments[:, 1], segments[:, 3]])
    
    metadata = {
        'num_segments': len(segments),
        'num_points': total_points,
        'bounds': {
            'x_min': x_coords.min(),
            'x_max': x_coords.max(),
            'y_min': y_coords.min(),
            'y_max': y_coords.max()
        },
        'total_length': np.sum(np.sqrt(
            (segments[:, 2] - segments[:, 0])**2 + 
            (segments[:, 3] - segments[:, 1])**2
        ))
    }
    
    return segments, metadata


def save_segments(segments, metadata, output_filename='coastline_segments.txt'):
    """
    Save segments to text file in box-counting format.
    
    Parameters:
    -----------
    segments : numpy.ndarray
        Array of segments [x1, y1, x2, y2]
    metadata : dict
        Metadata about the segments
    output_filename : str
        Output filename
    """
    
    with open(output_filename, 'w') as f:
        f.write("# Coastline segments for fractal box-counting analysis\n")
        f.write("# Format: x1 y1 x2 y2\n")
        f.write(f"# Source: Natural Earth coastline data\n")
        f.write(f"# Number of segments: {metadata['num_segments']}\n")
        f.write(f"# Total length: {metadata['total_length']:.2f} degrees\n")
        f.write(f"# Bounds: X=[{metadata['bounds']['x_min']:.4f}, {metadata['bounds']['x_max']:.4f}], ")
        f.write(f"Y=[{metadata['bounds']['y_min']:.4f}, {metadata['bounds']['y_max']:.4f}]\n")
        f.write("#\n")
        
        for seg in segments:
            f.write(f"{seg[0]:.6f} {seg[1]:.6f} {seg[2]:.6f} {seg[3]:.6f}\n")
    
    print(f"\nSegments saved to: {output_filename}")
    print(f"Number of segments: {metadata['num_segments']}")
    print(f"Total length: {metadata['total_length']:.2f} degrees")
    print(f"X range: [{metadata['bounds']['x_min']:.4f}, {metadata['bounds']['x_max']:.4f}]")
    print(f"Y range: [{metadata['bounds']['y_min']:.4f}, {metadata['bounds']['y_max']:.4f}]")


# Predefined regions with interesting coastlines
REGIONS = {
    'britain': {
        'name': 'Great Britain',
        'bbox': (-8, 50, 2, 59),
        'description': 'Classic fractal coastline studied by Mandelbrot'
    },
    'norway': {
        'name': 'Norway',
        'bbox': (4, 58, 31, 71),
        'description': 'Highly fractal fjord coastline'
    },
    'greece': {
        'name': 'Greece',
        'bbox': (19, 34, 29, 42),
        'description': 'Complex island coastline'
    },
    'japan': {
        'name': 'Japan',
        'bbox': (129, 30, 146, 46),
        'description': 'Island nation with varied coastline'
    },
    'california': {
        'name': 'California Coast',
        'bbox': (-125, 32, -117, 42),
        'description': 'Western US coastline'
    },
    'florida': {
        'name': 'Florida',
        'bbox': (-88, 24, -79, 31),
        'description': 'Peninsula with complex coastline'
    },
    'newzealand': {
        'name': 'New Zealand',
        'bbox': (166, -47, 179, -34),
        'description': 'Islands with fractal coastline'
    }
}


def main():
    """Main function with user interaction."""
    
    print("="*70)
    print("COASTLINE FRACTAL DATA DOWNLOADER")
    print("="*70)
    print("\nThis script downloads coastline data and converts it to")
    print("segment format for box-counting fractal analysis.\n")
    
    # Choose resolution
    print("Available resolutions:")
    print("  50m - Medium resolution (recommended, ~5MB)")
    print("  110m - Low resolution (fastest, ~1MB)")
    print("  10m - High resolution (detailed, ~50MB)")
    
    resolution = input("\nChoose resolution [50m]: ").strip() or '50m'
    
    # Download data
    try:
        shapefile = download_natural_earth_coastlines(resolution)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    # Choose region
    print("\n" + "="*70)
    print("Available regions:")
    for key, region in REGIONS.items():
        print(f"  {key:12s} - {region['name']:20s} - {region['description']}")
    print(f"  {'world':12s} - Entire world (may be large)")
    
    region_key = input("\nChoose region [britain]: ").strip() or 'britain'
    
    if region_key == 'world':
        bbox = None
        output_name = 'world_coastline_segments.txt'
    elif region_key in REGIONS:
        bbox = REGIONS[region_key]['bbox']
        output_name = f'{region_key}_coastline_segments.txt'
    else:
        print(f"Unknown region '{region_key}', using Britain")
        region_key = 'britain'
        bbox = REGIONS[region_key]['bbox']
        output_name = f'{region_key}_coastline_segments.txt'
    
    # Choose max segments
    max_segments_input = input("\nMaximum number of segments [2000]: ").strip()
    max_segments = int(max_segments_input) if max_segments_input else 2000
    
    # Extract segments
    print("\n" + "="*70)
    try:
        segments, metadata = extract_segments_from_shapefile(
            shapefile, 
            bbox=bbox, 
            max_segments=max_segments
        )
    except Exception as e:
        print(f"Error extracting segments: {e}")
        return
    
    # Save segments
    save_segments(segments, metadata, output_name)
    
    # Print analysis suggestions
    print("\n" + "="*70)
    print("SUGGESTED ANALYSIS PARAMETERS:")
    print("="*70)
    
    x_range = metadata['bounds']['x_max'] - metadata['bounds']['x_min']
    y_range = metadata['bounds']['y_max'] - metadata['bounds']['y_min']
    
    suggested_delta = max(x_range, y_range) / 20
    
    print(f"\nFor box-counting analysis, try:")
    print(f"  python simple_analyzer.py {output_name} \\")
    print(f"    --domain-x {metadata['bounds']['x_min']:.2f} {metadata['bounds']['x_max']:.2f} \\")
    print(f"    --domain-y {metadata['bounds']['y_min']:.2f} {metadata['bounds']['y_max']:.2f} \\")
    print(f"    --initial-delta {suggested_delta:.4f} \\")
    print(f"    --delta-factor 1.5 \\")
    print(f"    --num-steps 12")
    
    print(f"\nExpected fractal dimension: ~1.2 - 1.3 (typical for coastlines)")
    print(f"Mandelbrot's Britain result: ~1.25")
    print("="*70)


if __name__ == "__main__":
    main()
