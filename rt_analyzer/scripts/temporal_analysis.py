#!/usr/bin/env python3
"""
RT Instability Multifractal Temporal Evolution Analysis
This script analyzes how multifractal properties evolve over time.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from rt_analyzer import RTAnalyzer

# Configuration - Edit these paths to match your file locations
BASE_DIR = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/800x800"  # Change this to your data directory
OUTPUT_DIR = "media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/analysis/multifractal_analysis/results/multifractal_time_evolution_800x800"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define time points and corresponding files (adjust for your file naming convention)
TIME_FILES = {
    1.0: os.path.join(BASE_DIR, "RT800x800-1000.vtk"),
    3.0: os.path.join(BASE_DIR, "RT800x800-3000.vtk"),
    5.0: os.path.join(BASE_DIR, "RT800x800-5000.vtk"),
    7.0: os.path.join(BASE_DIR, "RT800x800-7000.vtk"),
    9.0: os.path.join(BASE_DIR, "RT800x800-9000.vtk")
}

# Initialize analyzer
analyzer = RTAnalyzer(OUTPUT_DIR)

# Define q-values (can use fewer points for faster computation)
q_values = np.arange(-5, 5.1, 1.0)  # Coarser q-range for faster results

# Run the multifractal evolution analysis
print(f"Starting multifractal temporal evolution analysis...")
results = analyzer.analyze_multifractal_evolution(
    TIME_FILES, 
    output_dir=OUTPUT_DIR,
    q_values=q_values
)

# At this point, the analyze_multifractal_evolution method has already created:
# 1. Individual multifractal analyses for each time point
# 2. Basic evolution plots (dimensions_evolution.png, multifractal_params_evolution.png)
# 3. CSV files with results

# Now let's create additional, more detailed visualizations

# 1. Create overlaid f(α) spectrum plot with time color gradient
print("Creating overlaid f(α) spectrum plot...")
plt.figure(figsize=(12, 8))
cmap = get_cmap('viridis')
times = sorted([res['time'] for res in results])
time_min, time_max = min(times), max(times)

for i, result in enumerate(sorted(results, key=lambda x: x['time'])):
    time = result['time']
    color = cmap((time - time_min) / (time_max - time_min))
    
    # Extract valid alpha and f_alpha values
    valid = ~np.isnan(result['alpha']) & ~np.isnan(result['f_alpha'])
    alpha = result['alpha'][valid]
    f_alpha = result['f_alpha'][valid]
    
    # Sort by alpha for proper line connection
    sort_idx = np.argsort(alpha)
    alpha = alpha[sort_idx]
    f_alpha = f_alpha[sort_idx]
    
    plt.plot(alpha, f_alpha, '-', color=color, linewidth=2, 
             label=f"t = {time:.1f}")

plt.xlabel('α', fontsize=14)
plt.ylabel('f(α)', fontsize=14)
plt.title('Evolution of Multifractal Spectrum f(α) Over Time', fontsize=16)
plt.grid(True)
plt.legend(loc='best')
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(time_min, time_max)), 
             label='Time')
plt.savefig(os.path.join(OUTPUT_DIR, "f_alpha_spectrum_evolution.png"), dpi=300)
plt.close()

# 2. Create D(q) curves overlaid with time color gradient
print("Creating overlaid D(q) curves plot...")
plt.figure(figsize=(12, 8))

for i, result in enumerate(sorted(results, key=lambda x: x['time'])):
    time = result['time']
    color = cmap((time - time_min) / (time_max - time_min))
    
    # Extract valid q and D(q) values
    valid = ~np.isnan(result['Dq'])
    q = result['q_values'][valid]
    Dq = result['Dq'][valid]
    
    plt.plot(q, Dq, '-', color=color, linewidth=2, 
             label=f"t = {time:.1f}")

plt.xlabel('q', fontsize=14)
plt.ylabel('D(q)', fontsize=14)
plt.title('Evolution of Generalized Dimensions D(q) Over Time', fontsize=16)
plt.grid(True)
plt.legend(loc='best')
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                                  norm=plt.Normalize(time_min, time_max)), 
             label='Time')
plt.savefig(os.path.join(OUTPUT_DIR, "Dq_curves_evolution.png"), dpi=300)
plt.close()

# 3. Create a detailed evolution of key multifractal parameters
print("Creating detailed parameter evolution plot...")
times = [res['time'] for res in results]
D0_values = [res['D0'] for res in results]
D1_values = [res['D1'] for res in results]
D2_values = [res['D2'] for res in results]
alpha_widths = [res['alpha_width'] for res in results]
mf_degrees = [res['degree_multifractality'] for res in results]

# First plot: Evolution of D0, D1, D2
plt.figure(figsize=(12, 8))
plt.plot(times, D0_values, 'bo-', linewidth=2, markersize=8, label='D(0) - Capacity Dimension')
plt.plot(times, D1_values, 'ro-', linewidth=2, markersize=8, label='D(1) - Information Dimension')
plt.plot(times, D2_values, 'go-', linewidth=2, markersize=8, label='D(2) - Correlation Dimension')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Dimension Value', fontsize=14)
plt.title('Evolution of Generalized Dimensions Over Time', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)

# Add text annotations at each point
for i, time in enumerate(times):
    plt.annotate(f"{D0_values[i]:.4f}", (time, D0_values[i]), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{D1_values[i]:.4f}", (time, D1_values[i]), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{D2_values[i]:.4f}", (time, D2_values[i]), 
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "detailed_dimensions_evolution.png"), dpi=300)
plt.close()

# Second plot: Evolution of multifractality measures
plt.figure(figsize=(12, 8))
plt.plot(times, alpha_widths, 'ms-', linewidth=2, markersize=8, label='α width')
plt.plot(times, mf_degrees, 'cd-', linewidth=2, markersize=8, label='Degree of multifractality')

plt.xlabel('Time', fontsize=14)
plt.ylabel('Parameter Value', fontsize=14)
plt.title('Evolution of Multifractality Measures Over Time', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)

# Add text annotations
for i, time in enumerate(times):
    plt.annotate(f"{alpha_widths[i]:.4f}", (time, alpha_widths[i]), 
                textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f"{mf_degrees[i]:.4f}", (time, mf_degrees[i]), 
                textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "detailed_multifractality_evolution.png"), dpi=300)
plt.close()

# 4. Create phase portrait: Fractal dimension vs. Mixing layer thickness
print("Creating phase portrait plot...")
# First get mixing layer thicknesses from data
mixing_thicknesses = []
for time, vtk_file in sorted(TIME_FILES.items()):
    data = analyzer.read_vtk_file(vtk_file)
    h0 = analyzer.find_initial_interface(data)
    mixing = analyzer.compute_mixing_thickness(data, h0)
    mixing_thicknesses.append(mixing['h_total'])

plt.figure(figsize=(12, 8))
plt.plot(mixing_thicknesses, D0_values, 'bo-', linewidth=2, markersize=8, label='D(0)')
plt.plot(mixing_thicknesses, D1_values, 'ro-', linewidth=2, markersize=8, label='D(1)')

# Add time labels to the points
for i, time in enumerate(times):
    plt.annotate(f"t={time:.1f}", (mixing_thicknesses[i], D0_values[i]), 
                textcoords="offset points", xytext=(5,5), ha='left')

plt.xlabel('Mixing Layer Thickness', fontsize=14)
plt.ylabel('Fractal Dimension', fontsize=14)
plt.title('Phase Portrait: Fractal Dimension vs. Mixing Layer Thickness', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "phase_portrait_dimension_vs_thickness.png"), dpi=300)
plt.close()

# 5. Create detailed summary table
print("Creating summary table...")
summary_df = pd.DataFrame({
    'Time': times,
    'Mixing_Thickness': mixing_thicknesses,
    'D0_Capacity_Dimension': D0_values,
    'D1_Information_Dimension': D1_values, 
    'D2_Correlation_Dimension': D2_values,
    'Alpha_Width': alpha_widths,
    'Degree_of_Multifractality': mf_degrees,
    'D1_minus_D0': [d1-d0 for d1, d0 in zip(D1_values, D0_values)],
    'D0_minus_D2': [d0-d2 for d0, d2 in zip(D0_values, D2_values)]
})

# Save to CSV
summary_df.to_csv(os.path.join(OUTPUT_DIR, "temporal_evolution_summary.csv"), index=False)

# Create HTML table for easy viewing
html_table = """
<html>
<head>
<style>
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: center; padding: 8px; border: 1px solid #ddd; }
    th { background-color: #f2f2f2; }
    tr:nth-child(even) { background-color: #f9f9f9; }
</style>
</head>
<body>
<h2>Multifractal Temporal Evolution Summary</h2>
<table>
  <tr>
    <th>Time</th>
    <th>Mixing Thickness</th>
    <th>D(0)</th>
    <th>D(1)</th>
    <th>D(2)</th>
    <th>α width</th>
    <th>Multifractality Degree</th>
    <th>D(1)-D(0)</th>
    <th>D(0)-D(2)</th>
  </tr>
"""

for i in range(len(times)):
    html_table += f"""
  <tr>
    <td>{times[i]:.1f}</td>
    <td>{mixing_thicknesses[i]:.4f}</td>
    <td>{D0_values[i]:.4f}</td>
    <td>{D1_values[i]:.4f}</td>
    <td>{D2_values[i]:.4f}</td>
    <td>{alpha_widths[i]:.4f}</td>
    <td>{mf_degrees[i]:.4f}</td>
    <td>{D1_values[i]-D0_values[i]:.4f}</td>
    <td>{D0_values[i]-D2_values[i]:.4f}</td>
  </tr>"""

html_table += """
</table>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "temporal_evolution_summary.html"), 'w') as f:
    f.write(html_table)

print(f"Temporal evolution analysis complete. Results saved to {OUTPUT_DIR}")
