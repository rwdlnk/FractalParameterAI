#!/usr/bin/env python3
"""
RT Instability Multifractal Resolution Dependence Analysis
This script analyzes how multifractal properties depend on grid resolution.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from rt_analyzer import RTAnalyzer

# Configuration - Edit these paths to match your file locations
BASE_DIR = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/800x800"  # Change this to your data directory
OUTPUT_DIR = "/media/rod/ResearchII_III/ResearchIII/githubRepos/svof/build-Release/test_data/fracConv/results/multifractal_resolution_dependence"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define resolutions and corresponding files
RESOLUTION_FILES = {
    100: os.path.join(BASE_DIR, "RT100x100-9000.vtk"),
    200: os.path.join(BASE_DIR, "RT200x200-9000.vtk"),
    400: os.path.join(BASE_DIR, "RT400x400-9000.vtk"),
    800: os.path.join(BASE_DIR, "RT800x800-9000.vtk")
}

# Initialize analyzer
analyzer = RTAnalyzer(OUTPUT_DIR)

# Define q-values
q_values = np.arange(-5, 5.1, 1.0)

# Run the multifractal evolution analysis
print(f"Starting multifractal resolution dependence analysis...")
results = analyzer.analyze_multifractal_evolution(
    RESOLUTION_FILES, 
    output_dir=OUTPUT_DIR,
    q_values=q_values
)

# Extract resolution-dependent properties
resolutions = np.array(sorted([res['resolution'] for res in results]))
D0_values = [res['D0'] for res in sorted(results, key=lambda x: x['resolution'])]
D1_values = [res['D1'] for res in sorted(results, key=lambda x: x['resolution'])]
D2_values = [res['D2'] for res in sorted(results, key=lambda x: x['resolution'])]
alpha_widths = [res['alpha_width'] for res in sorted(results, key=lambda x: x['resolution'])]
mf_degrees = [res['degree_multifractality'] for res in sorted(results, key=lambda x: x['resolution'])]

# Create 1/N values for extrapolation
h_values = 1.0 / resolutions

# Define extrapolation model function
def extrapolation_model(h, f_inf, C, p):
    """Model for Richardson extrapolation: f(h) = f_inf + C*h^p"""
    return f_inf + C * h**p

# Function to perform extrapolation and create plots
def extrapolate_and_plot(values, name, ylabel, output_dir):
    """Extrapolate values to infinite resolution and create plots."""
    try:
        # Perform the curve fitting
        params, pcov = curve_fit(extrapolation_model, h_values, values, 
                                p0=[values[-1] + 0.05, -0.5, 1.0])
        f_inf, C, p = params
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        f_inf_err, C_err, p_err = perr
        
        # Create extrapolation plot
        plt.figure(figsize=(12, 8))
        
        # Plot the data points
        plt.plot(resolutions, values, 'bo-', linewidth=2, markersize=10, 
                label=f'Measured values')
        
        # Add resolution labels to points
        for i, res in enumerate(resolutions):
            plt.annotate(f"{res}×{res}", (resolutions[i], values[i]), 
                        textcoords="offset points", xytext=(5,5), ha='left')
        
        # Create smooth curve for the model
        h_curve = np.linspace(0, h_values[0], 100)
        res_curve = 1.0 / h_curve
        plt.plot(res_curve, extrapolation_model(h_curve, f_inf, C, p), 'r--', linewidth=2,
                label=f'Extrapolation: {name}(∞) = {f_inf:.4f} ± {f_inf_err:.4f}')
        
        # Add horizontal line at extrapolated value
        plt.axhline(y=f_inf, color='k', linestyle=':')
        
        # Format the plot
        plt.xscale('log', base=2)
        plt.xlabel('Resolution (N)', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(f'Resolution Convergence of {name}', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        
        # Add text with extrapolation details
        plt.figtext(0.5, 0.01, 
                   f"Extrapolation model: {name}(N) = {name}(∞) + C·N^(-p) = {f_inf:.4f} + ({C:.4f})·N^(-{p:.4f})", 
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(os.path.join(output_dir, f'{name}_extrapolation.png'), dpi=300)
        plt.close()
        
        return f_inf, f_inf_err, C, p
    
    except Exception as e:
        print(f"Error in extrapolation of {name}: {str(e)}")
        return None, None, None, None

# Perform extrapolation for each property
print("Performing extrapolation analysis...")
D0_inf, D0_err, C_D0, p_D0 = extrapolate_and_plot(D0_values, "D0", "D(0) - Capacity Dimension", OUTPUT_DIR)
D1_inf, D1_err, C_D1, p_D1 = extrapolate_and_plot(D1_values, "D1", "D(1) - Information Dimension", OUTPUT_DIR)
D2_inf, D2_err, C_D2, p_D2 = extrapolate_and_plot(D2_values, "D2", "D(2) - Correlation Dimension", OUTPUT_DIR)
width_inf, width_err, C_width, p_width = extrapolate_and_plot(alpha_widths, "Width", "α Spectrum Width", OUTPUT_DIR)

# Create overlay of D(q) curves for different resolutions
print("Creating D(q) overlay plot...")
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r', 'm']
markers = ['o', 's', '^', 'D']

for i, res in enumerate(sorted(results, key=lambda x: x['resolution'])):
    resolution = res['resolution']
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    # Extract valid q and D(q) values
    valid = ~np.isnan(res['Dq'])
    q = res['q_values'][valid]
    Dq = res['Dq'][valid]
    
    plt.plot(q, Dq, color=color, marker=marker, linestyle='-', linewidth=2, 
             markersize=8, markevery=2, label=f"{resolution}×{resolution}")

plt.xlabel('q', fontsize=14)
plt.ylabel('D(q)', fontsize=14)
plt.title('Resolution Dependence of D(q) Spectrum', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "Dq_resolution_overlay.png"), dpi=300)
plt.close()

# Create overlay of f(α) spectra for different resolutions
print("Creating f(α) overlay plot...")
plt.figure(figsize=(12, 8))

for i, res in enumerate(sorted(results, key=lambda x: x['resolution'])):
    resolution = res['resolution']
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    # Extract valid alpha and f_alpha values
    valid = ~np.isnan(res['alpha']) & ~np.isnan(res['f_alpha'])
    alpha = res['alpha'][valid]
    f_alpha = res['f_alpha'][valid]
    
    # Sort by alpha for proper line connection
    sort_idx = np.argsort(alpha)
    alpha = alpha[sort_idx]
    f_alpha = f_alpha[sort_idx]
    
    plt.plot(alpha, f_alpha, color=color, marker=marker, linestyle='-', 
             linewidth=2, markersize=8, markevery=2, label=f"{resolution}×{resolution}")

plt.xlabel('α', fontsize=14)
plt.ylabel('f(α)', fontsize=14)
plt.title('Resolution Dependence of Multifractal Spectrum f(α)', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "f_alpha_resolution_overlay.png"), dpi=300)
plt.close()

# Create comparison of specific D(q) values across resolutions
print("Creating specific D(q) values comparison...")
q_to_extract = [-5, -2, 0, 1, 2, 5]  # Specific q values to compare
q_extracted_values = {q: [] for q in q_to_extract}

for res in sorted(results, key=lambda x: x['resolution']):
    for q_target in q_to_extract:
        # Find the closest q value in the results
        idx = np.argmin(np.abs(res['q_values'] - q_target))
        q_extracted_values[q_target].append(res['Dq'][idx])

# Plot the comparison
plt.figure(figsize=(12, 8))

for i, q in enumerate(q_to_extract):
    plt.plot(resolutions, q_extracted_values[q], f'{colors[i % len(colors)]}{markers[i % len(markers)]}-', 
             linewidth=2, markersize=8, label=f'D({q})')

plt.xscale('log', base=2)
plt.xlabel('Resolution (N)', fontsize=14)
plt.ylabel('D(q) Value', fontsize=14)
plt.title('Resolution Dependence of Specific D(q) Values', fontsize=16)
plt.grid(True)
plt.legend(loc='best', fontsize=12)
plt.savefig(os.path.join(OUTPUT_DIR, "specific_Dq_resolution_dependence.png"), dpi=300)
plt.close()

# Create summary table with extrapolation results
print("Creating summary table with extrapolation results...")
summary_df = pd.DataFrame({
    'Resolution': np.append(resolutions, ['∞']),
    'D0_Capacity_Dimension': np.append(D0_values, [D0_inf]),
    'D1_Information_Dimension': np.append(D1_values, [D1_inf]), 
    'D2_Correlation_Dimension': np.append(D2_values, [D2_inf]),
    'Alpha_Width': np.append(alpha_widths, [width_inf]),
    'Degree_of_Multifractality': np.append(mf_degrees, [None]),
    'D1_minus_D0': np.append([d1-d0 for d1, d0 in zip(D1_values, D0_values)], [D1_inf-D0_inf]),
    'D0_minus_D2': np.append([d0-d2 for d0, d2 in zip(D0_values, D2_values)], [D0_inf-D2_inf])
})

# Save to CSV
summary_df.to_csv(os.path.join(OUTPUT_DIR, "resolution_dependence_summary.csv"), index=False)

# Create HTML table for easy viewing
html_table = """
<html>
<head>
<style>
    table { border-collapse: collapse; width: 100%; }
    th, td { text-align: center; padding: 8px; border: 1px solid #ddd; }
    th { background-color: #f2f2f2; }
    tr:nth-child(even) { background-color: #f9f9f9; }
    .infinite { font-weight: bold; background-color: #ffffcc; }
</style>
</head>
<body>
<h2>Multifractal Resolution Dependence Summary</h2>
<table>
  <tr>
    <th>Resolution</th>
    <th>D(0)</th>
    <th>D(1)</th>
    <th>D(2)</th>
    <th>α width</th>
    <th>Multifractality Degree</th>
    <th>D(1)-D(0)</th>
    <th>D(0)-D(2)</th>
  </tr>
"""

for i in range(len(resolutions)):
    html_table += f"""
  <tr>
    <td>{resolutions[i]}×{resolutions[i]}</td>
    <td>{D0_values[i]:.4f}</td>
    <td>{D1_values[i]:.4f}</td>
    <td>{D2_values[i]:.4f}</td>
    <td>{alpha_widths[i]:.4f}</td>
    <td>{mf_degrees[i]:.4f}</td>
    <td>{D1_values[i]-D0_values[i]:.4f}</td>
    <td>{D0_values[i]-D2_values[i]:.4f}</td>
  </tr>"""

# Add infinite resolution extrapolation row
html_table += f"""
  <tr class="infinite">
    <td>∞ (Extrapolated)</td>
    <td>{D0_inf:.4f} ± {D0_err:.4f}</td>
    <td>{D1_inf:.4f} ± {D1_err:.4f}</td>
    <td>{D2_inf:.4f} ± {D2_err:.4f}</td>
    <td>{width_inf:.4f} ± {width_err:.4f}</td>
    <td>N/A</td>
    <td>{D1_inf-D0_inf:.4f}</td>
    <td>{D0_inf-D2_inf:.4f}</td>
  </tr>"""

html_table += """
</table>
</body>
</html>
"""

with open(os.path.join(OUTPUT_DIR, "resolution_dependence_summary.html"), 'w') as f:
    f.write(html_table)

# Create extrapolation model summary
extrapolation_summary = pd.DataFrame({
    'Parameter': ['D0', 'D1', 'D2', 'Alpha Width'],
    'Extrapolated_Value': [D0_inf, D1_inf, D2_inf, width_inf],
    'Error': [D0_err, D1_err, D2_err, width_err],
    'Coefficient_C': [C_D0, C_D1, C_D2, C_width],
    'Exponent_p': [p_D0, p_D1, p_D2, p_width]
})

extrapolation_summary.to_csv(os.path.join(OUTPUT_DIR, "extrapolation_model_summary.csv"), index=False)

print(f"Resolution dependence analysis complete. Results saved to {OUTPUT_DIR}")
