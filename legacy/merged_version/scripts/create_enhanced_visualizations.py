#!/usr/bin/env python3
"""
Create enhanced visualizations for the paper using existing data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import re
import argparse

# Import the fractal analyzer
from fractal_analyzer import FractalAnalyzer
from fractal_analyzer.analysis_tools import FractalAnalysisTools

# Try to import RT analyzer
try:
    from rt_analyzer import RTAnalyzer
    RT_ANALYZER_AVAILABLE = True
except ImportError:
    print("Warning: RT Analyzer not available. RT analysis will be skipped.")
    RT_ANALYZER_AVAILABLE = False

# Base directory for output
parser = argparse.ArgumentParser(description='Create enhanced visualizations for fractal analysis')
parser.add_argument('--input-dir', default='./paper_data', 
                   help='Directory containing original analysis results')
parser.add_argument('--output-dir', default='./paper_data/enhanced_visualizations',
                   help='Directory to save enhanced visualizations')
args = parser.parse_args()

BASE_DIR = args.output_dir
ORIGINAL_DIR = args.input_dir

# Create log file
LOG_FILE = "enhanced_visualizations.log"

def log_message(message):
    """Log a message to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(LOG_FILE, "a") as f:
        f.write(full_message + "\n")

# Define the enhanced box visualization function
def create_paper_box_visualization(fractal_type, output_dir, level=5, box_size_percentile=75, 
                                 box_line_width=0.7, box_alpha=0.8, box_color='r', zoom_region=None):
    """
    Create improved box counting visualization for the paper.
    
    Args:
        fractal_type: Type of fractal to visualize
        output_dir: Directory to save output figures
        level: Iteration level for the fractal
        box_size_percentile: Percentile of box sizes to use (higher = smaller boxes)
        box_line_width: Line width for boxes
        box_alpha: Transparency for boxes
        box_color: Color for boxes ('r', 'b', 'g', etc.)
        zoom_region: Custom zoom region as (xmin, ymin, xmax, ymax) or None for auto
    """
    log_message(f"Creating enhanced box visualization for {fractal_type} at level {level}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create analyzer
    analyzer = FractalAnalyzer(fractal_type)
    
    # Generate the fractal - this will use cache if available
    _, segments = analyzer.generate_fractal(fractal_type, level=level)
    log_message(f"Generated {fractal_type} with {len(segments)} segments")
    
    # Find the bounding box
    min_x = min(min(s[0][0], s[1][0]) for s in segments)
    max_x = max(max(s[0][0], s[1][0]) for s in segments)
    min_y = min(min(s[0][1], s[1][1]) for s in segments)
    max_y = max(max(s[0][1], s[1][1]) for s in segments)
    
    # Add a small margin
    margin = max(max_x - min_x, max_y - min_y) * 0.05
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    
    # Create figure for the fractal curve
    plt.figure(figsize=(8, 8))
    
    # Plot the fractal curve
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.7)
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.axis('equal')
    plt.title(f'{fractal_type.capitalize()} Curve (Level {level})')
    
    # Save the fractal curve
    curve_file = os.path.join(output_dir, f"{fractal_type}_curve_level_{level}.png")
    plt.savefig(curve_file, dpi=300)
    plt.close()
    
    # Calculate extent for box sizes
    extent = max(max_x - min_x, max_y - min_y)
    max_box_size = extent / 2.0  # Explicit casting to float
    min_box_size = 0.001  # Explicit setting
    
    log_message(f"Using box size range: {min_box_size} to {max_box_size}")
    
    # Now create a figure with box overlay
    # First run box counting to get good box sizes
    try:
        # For safety, verify our parameters are valid
        if min_box_size <= 0:
            raise ValueError(f"Invalid min_box_size parameter: {min_box_size}")
        
        # This will use cached results if available
        box_sizes, box_counts, bounding_box = analyzer.box_counter.box_counting_optimized(
            segments, min_box_size=min_box_size, max_box_size=max_box_size, box_size_factor=1.5)            
        
        log_message(f"Box counting returned {len(box_sizes)} box sizes")
    except Exception as e:
        log_message(f"Error in box counting: {str(e)}")
        # Generate box sizes manually
        log_message("Generating box sizes manually")
        box_sizes = []
        box_counts = []
        current_size = max_box_size
        ratio = 1.5
        
        # Generate at least 10 box sizes
        while current_size >= min_box_size and len(box_sizes) < 10:
            box_sizes.append(current_size)
            # Just use a placeholder count that decreases with box size
            box_counts.append(int(100 / current_size))
            current_size /= ratio
        
        log_message(f"Generated {len(box_sizes)} box sizes manually")
    
    # Choose a box size for visualization
    if len(box_sizes) > 0:
        # Choose a good box size based on percentile
        box_size_idx = min(len(box_sizes) - 1, max(0, int(len(box_sizes) * box_size_percentile / 100)))
        box_size = box_sizes[box_size_idx]
    else:
        # Fallback
        box_size = extent / 50
    
    log_message(f"Using box size {box_size} for visualization")
    
    # Create new figure for box overlay
    plt.figure(figsize=(8, 8))
    
    # Plot the fractal again
    for (x1, y1), (x2, y2) in segments:
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.7)
    
    # Calculate boxes
    num_boxes_x = int(np.ceil((max_x - min_x) / box_size))
    num_boxes_y = int(np.ceil((max_y - min_y) / box_size))
    
    log_message(f"Drawing grid with {num_boxes_x}x{num_boxes_y} boxes (size: {box_size:.6f})")
    
    # Use draw_boxes helper function to avoid checking each box individually repeatedly
    def draw_boxes(ax, region_filter=None):
        """Draw boxes that intersect the fractal, optionally filtered by region."""
        occupied_boxes = []
        count = 0
        # Check all boxes for intersection with any segment
        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = min_x + i * box_size
                box_ymin = min_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size
                
                # Apply region filter if provided
                if region_filter:
                    reg_xmin, reg_ymin, reg_xmax, reg_ymax = region_filter
                    if (box_xmax < reg_xmin or box_xmin > reg_xmax or 
                        box_ymax < reg_ymin or box_ymin > reg_ymax):
                        continue
                
                # Check if any segment intersects this box
                for (x1, y1), (x2, y2) in segments:
                    if analyzer.base.liang_barsky_line_box_intersection(x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        occupied_boxes.append((box_xmin, box_ymin))
                        count += 1
                        break
        
        # Draw the occupied boxes
        for box_xmin, box_ymin in occupied_boxes:
            rect = plt.Rectangle((box_xmin, box_ymin), box_size, box_size,
                              facecolor='none', edgecolor=box_color, 
                              linewidth=box_line_width, alpha=box_alpha)
            ax.add_patch(rect)
        
        return count, occupied_boxes
    
    # Draw boxes on main figure
    count, occupied_boxes = draw_boxes(plt.gca())
    log_message(f"Found {count} occupied boxes")
    
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.axis('equal')
    plt.grid(False)
    plt.title(f'{fractal_type.capitalize()} with Box Counting (Box Size: {box_size:.6f})')
    
    # Save the box overlay
    box_file = os.path.join(output_dir, f"{fractal_type}_box_overlay.png")
    plt.savefig(box_file, dpi=300)
    plt.close()
    
    # Define zoom regions for different fractals
    if zoom_region is None:
        # Default zoom regions for each fractal type
        if fractal_type == 'koch':
            # For Koch curve, zoom to the left part
            zoom_min_x = min_x + (max_x - min_x) * 0.05
            zoom_max_x = min_x + (max_x - min_x) * 0.35
            zoom_min_y = min_y
            zoom_max_y = min_y + (max_y - min_y) * 0.5
        elif fractal_type == 'sierpinski':
            # For Sierpinski, zoom to the top corner
            zoom_min_x = min_x + (max_x - min_x) * 0.4
            zoom_max_x = min_x + (max_x - min_x) * 0.6
            zoom_min_y = min_y + (max_y - min_y) * 0.8
            zoom_max_y = max_y
        elif fractal_type == 'dragon':
            # For Dragon curve, zoom to a detailed area
            zoom_min_x = min_x + (max_x - min_x) * 0.2
            zoom_max_x = min_x + (max_x - min_x) * 0.5
            zoom_min_y = min_y + (max_y - min_y) * 0.5
            zoom_max_y = min_y + (max_y - min_y) * 0.8
        elif fractal_type == 'hilbert':
            # For Hilbert curve, zoom to bottom-left corner
            zoom_min_x = min_x 
            zoom_max_x = min_x + (max_x - min_x) * 0.3
            zoom_min_y = min_y
            zoom_max_y = min_y + (max_y - min_y) * 0.3
        elif fractal_type == 'minkowski':
            # For Minkowski curve, zoom to center
            zoom_min_x = min_x + (max_x - min_x) * 0.4
            zoom_max_x = min_x + (max_x - min_x) * 0.6
            zoom_min_y = min_y + (max_y - min_y) * 0.4
            zoom_max_y = min_y + (max_y - min_y) * 0.6
        else:
            # Default zoom to top-left quarter
            zoom_min_x = min_x
            zoom_max_x = min_x + (max_x - min_x) * 0.25
            zoom_min_y = min_y
            zoom_max_y = min_y + (max_y - min_y) * 0.25
    else:
        # Use custom zoom region
        zoom_min_x, zoom_min_y, zoom_max_x, zoom_max_y = zoom_region
    
    # Create zoom view
    plt.figure(figsize=(8, 8))
    
    # Plot segments in zoom region (more efficient than plotting all)
    for (x1, y1), (x2, y2) in segments:
        # Only plot segments that might be in the zoom region
        if (x1 >= zoom_min_x and x1 <= zoom_max_x and y1 >= zoom_min_y and y1 <= zoom_max_y) or \
           (x2 >= zoom_min_x and x2 <= zoom_max_x and y2 >= zoom_min_y and y2 <= zoom_max_y):
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.7)
    
    # Draw boxes in the zoom region
    zoom_count, _ = draw_boxes(plt.gca(), (zoom_min_x, zoom_min_y, zoom_max_x, zoom_max_y))
    log_message(f"Drew {zoom_count} boxes in zoom region")
    
    plt.xlim(zoom_min_x, zoom_max_x)
    plt.ylim(zoom_min_y, zoom_max_y)
    plt.axis('equal')
    plt.grid(False)
    plt.title(f'{fractal_type.capitalize()} Box Counting (Zoomed View)')
    
    # Save the zoomed view
    zoom_file = os.path.join(output_dir, f"{fractal_type}_box_overlay_zoom.png")
    plt.savefig(zoom_file, dpi=300)
    plt.close()
    
    # Create additional closeup for extra detail
    # Use an even smaller area of interest
    detail_width = (zoom_max_x - zoom_min_x) * 0.4
    detail_height = (zoom_max_y - zoom_min_y) * 0.4
    detail_min_x = zoom_min_x + (zoom_max_x - zoom_min_x) * 0.3
    detail_max_x = detail_min_x + detail_width
    detail_min_y = zoom_min_y + (zoom_max_y - zoom_min_y) * 0.3
    detail_max_y = detail_min_y + detail_height
    
    # Create extreme closeup
    plt.figure(figsize=(8, 8))
    
    # Plot segments in detail region
    for (x1, y1), (x2, y2) in segments:
        # Only plot segments that might be in the detail region
        if (x1 >= detail_min_x and x1 <= detail_max_x and y1 >= detail_min_y and y1 <= detail_max_y) or \
           (x2 >= detail_min_x and x2 <= detail_max_x and y2 >= detail_min_y and y2 <= detail_max_y):
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=0.9)  # Slightly thicker lines for detail view
    
    # Draw boxes in detail region
    detail_count, _ = draw_boxes(plt.gca(), (detail_min_x, detail_min_y, detail_max_x, detail_max_y))
    log_message(f"Drew {detail_count} boxes in detail region")
    
    plt.xlim(detail_min_x, detail_max_x)
    plt.ylim(detail_min_y, detail_max_y)
    plt.axis('equal')
    plt.grid(False)
    plt.title(f'{fractal_type.capitalize()} Box Counting (Detail View)')
    
    # Save the detail view
    detail_file = os.path.join(output_dir, f"{fractal_type}_box_overlay_detail.png")
    plt.savefig(detail_file, dpi=300)
    plt.close()
    
    log_message(f"Created enhanced box visualizations for {fractal_type}")
    
    return curve_file, box_file, zoom_file, detail_file

def create_comparison_figures(output_dir):
    """Create comparison figures summarizing the results for all fractals."""
    log_message("Creating comparison figures for all fractals...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dimension data from CSV files
    fractal_types = ['koch', 'sierpinski', 'hilbert', 'dragon', 'minkowski']
    results = []
    
    for fractal_type in fractal_types:
        csv_file = os.path.join(ORIGINAL_DIR, "mathematical_fractals", 
                              fractal_type, f"{fractal_type}_iterations_comparison.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            # Get the highest level result
            max_level_row = df.iloc[-1]
            
            # Store results
            results.append({
                'Fractal Type': fractal_type.capitalize(),
                'Theoretical': max_level_row.get('Theoretical', None),
                'Basic_Dimension': max_level_row['Basic_Dimension'],
                'Basic_Error': max_level_row['Basic_Error'],
                'Basic_R_squared': max_level_row['Basic_R_squared'],
                'Window_Dimension': max_level_row['Window_Dimension'],
                'Window_Error': max_level_row['Window_Error'],
                'Window_R_squared': max_level_row['Window_R_squared'],
                'Optimal_Window': max_level_row['Optimal_Window'],
                'Level': max_level_row['Level']
            })
    
    # Create comparison DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate relative errors where theoretical value exists
        df['Basic_Rel_Error'] = np.nan
        df['Window_Rel_Error'] = np.nan
        
        for i, row in df.iterrows():
            if pd.notnull(row['Theoretical']) and row['Theoretical'] > 0:
                df.loc[i, 'Basic_Rel_Error'] = abs(row['Basic_Dimension'] - row['Theoretical']) / row['Theoretical'] * 100
                df.loc[i, 'Window_Rel_Error'] = abs(row['Window_Dimension'] - row['Theoretical']) / row['Theoretical'] * 100
        
        # Save the comparison to CSV
        df.to_csv(os.path.join(output_dir, "fractal_comparison.csv"), index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Bar positions
        x = np.arange(len(fractal_types))
        width = 0.35
        
        # Create bar chart for relative errors
        basic_errors = df['Basic_Rel_Error'].fillna(0)
        window_errors = df['Window_Rel_Error'].fillna(0)
        
        plt.bar(x - width/2, basic_errors, width, label='Basic Method')
        plt.bar(x + width/2, window_errors, width, label='Sliding Window')
        
        plt.xlabel('Fractal Type')
        plt.ylabel('Relative Error (%)')
        plt.title('Fractal Dimension Calculation Error Comparison')
        plt.xticks(x, df['Fractal Type'])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add theoretical dimension values as text
        for i, row in df.iterrows():
            if pd.notnull(row['Theoretical']):
                plt.text(i, -2, f"Theoretical: {row['Theoretical']:.4f}", 
                       ha='center', rotation=0, fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_comparison.png"), dpi=300)
        plt.close()
        
        # Create accuracy improvement plot
        plt.figure(figsize=(12, 6))
        
        # Calculate improvement percentage
        improvement = []
        for i, row in df.iterrows():
            if pd.notnull(row['Basic_Rel_Error']) and row['Basic_Rel_Error'] > 0:
                imp = (row['Basic_Rel_Error'] - row['Window_Rel_Error']) / row['Basic_Rel_Error'] * 100
                improvement.append(imp)
            else:
                improvement.append(0)
        
        plt.bar(x, improvement, width=0.5)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.xlabel('Fractal Type')
        plt.ylabel('Accuracy Improvement (%)')
        plt.title('Sliding Window Method: Accuracy Improvement Over Basic Method')
        plt.xticks(x, df['Fractal Type'])
        plt.grid(True, alpha=0.3, axis='y')
        
        for i, imp in enumerate(improvement):
            if imp > 0:
                plt.text(i, imp + 2, f"{imp:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_improvement.png"), dpi=300)
        plt.close()
        
        log_message(f"Created comparison figures in {output_dir}")
        
    else:
        log_message("No data found for comparison figures")

def main():
    """Main function to run enhanced visualizations."""
    # Clear log file
    with open(LOG_FILE, "w") as f:
        f.write("")
    
    log_message("Starting enhanced visualization generation...")
    
    # Create base directory
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # Print cache statistics if available
    try:
        # Get cache stats
        analyzer = FractalAnalyzer('koch')
        if hasattr(analyzer.box_counter, 'cache') and hasattr(analyzer.box_counter.cache, 'stats'):
            box_stats = analyzer.box_counter.cache.stats()
            log_message("\n===== CACHE STATISTICS =====")
            log_message(f"Box counting cache: {box_stats['hits']} hits, {box_stats['misses']} misses, " 
                       f"{box_stats['hit_rate']*100:.1f}% hit rate")
    except Exception as e:
        log_message(f"Could not get cache statistics: {str(e)}")
    
    # 1. Create enhanced box visualizations for each fractal
    fractal_types = ['koch', 'sierpinski', 'hilbert', 'dragon', 'minkowski']
    
    for fractal_type in fractal_types:
        try:
            log_message(f"Creating enhanced box visualization for {fractal_type}")
            
            # Set appropriate level for each fractal
            level = 5 if fractal_type == 'hilbert' else 6
            
            # Create the visualizations
            curve_file, box_file, zoom_file, detail_file = create_paper_box_visualization(
                fractal_type,
                os.path.join(BASE_DIR, fractal_type),
                level=level,
                box_line_width=0.7,
                box_alpha=0.8
            )
            
            log_message(f"Created visualizations for {fractal_type}:")
            log_message(f"  Curve: {curve_file}")
            log_message(f"  Box overlay: {box_file}")
            log_message(f"  Zoom view: {zoom_file}")
            log_message(f"  Detail view: {detail_file}")
            
        except Exception as e:
            log_message(f"Error creating visualization for {fractal_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 2. Create comparison figures
    try:
        create_comparison_figures(os.path.join(BASE_DIR, "comparison"))
    except Exception as e:
        log_message(f"Error creating comparison figures: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Final cache statistics
    try:
        if hasattr(analyzer.box_counter, 'cache') and hasattr(analyzer.box_counter.cache, 'stats'):
            box_stats = analyzer.box_counter.cache.stats()
            log_message("\n===== FINAL CACHE STATISTICS =====")
            log_message(f"Box counting cache: {box_stats['hits']} hits, {box_stats['misses']} misses, " 
                       f"{box_stats['hit_rate']*100:.1f}% hit rate")
    except Exception:
        pass
    
    log_message("\nEnhanced visualization generation complete!")
    log_message(f"All results saved to: {os.path.abspath(BASE_DIR)}")

if __name__ == "__main__":
    main()
