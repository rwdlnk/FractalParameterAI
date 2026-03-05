#!/usr/bin/env python3
"""
Hilbert Curve Scaling Range Analysis
====================================

This example demonstrates scaling range analysis using the Hilbert curve, which is a
space-filling curve that approaches the theoretical maximum dimension (D = 2.0) and
tests algorithm behavior at extreme scaling conditions.

Key Features Demonstrated:
- Space-filling curve analysis (D = 2.0 exactly)
- Standard box counting without optimization (baseline method)
- Scaling range optimization and box size selection
- Algorithm performance at extreme dimensions
- Simple, straightforward fractal dimension calculation

Hilbert Curve Properties:
- Theoretical dimension: D = 2.0 (exact, space-filling)
- Self-similar space-filling pattern
- Tests algorithm limits at boundary between 1D curves and 2D areas
- Excellent for scaling range analysis and parameter optimization
- Demonstrates basic fractal analysis workflow

Usage:
    python hilbert_scaling_analysis.py
    python hilbert_scaling_analysis.py --no_optimization --basic_analysis
    python hilbert_scaling_analysis.py --scaling_study --max_level 6
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Import the fractal analyzer
try:
    from fractal_analyzer import FractalAnalyzer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from fractal_analyzer import FractalAnalyzer


class HilbertScalingAnalyzer:
    """
    Scaling range analysis using Hilbert curve as space-filling test case.
    """
    
    def __init__(self, eps_plots=False, no_titles=False):
        self.analyzer = FractalAnalyzer(
            fractal_type='hilbert',
            eps_plots=eps_plots,
            no_titles=no_titles
        )
        self.theoretical_dimension = 2.0  # Exact for space-filling curve
        self.results = {}
    
    def basic_dimension_calculation(self, level=4, use_optimization=True, create_plots=True):
        """
        Basic fractal dimension calculation demonstrating standard workflow.
        
        Args:
            level: Hilbert curve iteration level
            use_optimization: Whether to use grid optimization
            create_plots: Whether to create analysis plots
        
        Returns:
            Dictionary with basic calculation results
        """
        print(f"\n{'='*70}")
        print(f"HILBERT CURVE BASIC DIMENSION CALCULATION")
        print(f"Level {level}, Optimization: {'ON' if use_optimization else 'OFF'}")
        print(f"{'='*70}")
        
        # Generate Hilbert curve
        print(f"1. Generating Hilbert curve at level {level}...")
        start_time = time.time()
        points, segments = self.analyzer.generate_fractal('hilbert', level)
        generation_time = time.time() - start_time
        
        print(f"   ✓ Generated {len(segments)} segments")
        print(f"   ✓ Generation time: {generation_time:.3f} seconds")
        print(f"   ✓ Theoretical dimension: {self.theoretical_dimension:.1f} (exact)")
        
        # Basic fractal dimension analysis
        method_name = "Grid Optimization" if use_optimization else "Standard Method"
        print(f"\n2. Fractal dimension analysis ({method_name})...")
        
        analysis_start = time.time()
        
        # Use analyze_linear_region for comprehensive analysis
        results = self.analyzer.analyze_linear_region(
            segments=segments,
            fractal_type='hilbert',
            plot_results=create_plots,
            plot_boxes=(level <= 3),  # Only show boxes for small levels
            use_grid_optimization=use_optimization,
            box_size_factor=1.5,
            trim_boundary=1,
            return_box_data=True
        )
        
        analysis_time = time.time() - analysis_start
        total_time = time.time() - start_time
        
        # Extract results
        windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, optimal_intercept, box_sizes, box_counts, bounding_box = results
        
        measurement_error = errors[windows.index(optimal_window)]
        quality_r2 = r_squared[windows.index(optimal_window)]
        theoretical_error = abs(optimal_dimension - self.theoretical_dimension)
        
        # Display results
        print(f"3. Results Summary:")
        print(f"   Measured dimension: {optimal_dimension:.6f} ± {measurement_error:.6f}")
        print(f"   Theoretical dimension: {self.theoretical_dimension:.1f}")
        print(f"   Error from theoretical: {theoretical_error:.6f}")
        print(f"   Relative error: {(theoretical_error/self.theoretical_dimension)*100:.3f}%")
        print(f"   R-squared: {quality_r2:.6f}")
        print(f"   Optimal window size: {optimal_window} points")
        print(f"   Analysis time: {analysis_time:.3f} seconds")
        print(f"   Total time: {total_time:.3f} seconds")
        
        # Quality assessment
        print(f"\n4. Quality Assessment:")
        if theoretical_error < 0.01 and quality_r2 > 0.99:
            assessment = "EXCELLENT - High accuracy and quality"
        elif theoretical_error < 0.05 and quality_r2 > 0.98:
            assessment = "GOOD - Acceptable accuracy and quality"
        elif theoretical_error < 0.1 and quality_r2 > 0.95:
            assessment = "ACCEPTABLE - Reasonable results"
        else:
            assessment = "POOR - Check parameters or implementation"
        
        print(f"   Assessment: {assessment}")
        
        # Scaling range analysis
        log_box_sizes = np.log(box_sizes)
        scaling_range = max(log_box_sizes) - min(log_box_sizes)
        scaling_decades = scaling_range / np.log(10)
        
        print(f"   Scaling range: {scaling_decades:.2f} decades")
        if scaling_decades < 1.5:
            print("   ⚠ Limited scaling range - consider adjusting box size parameters")
        elif scaling_decades > 3.0:
            print("   ✓ Excellent scaling range for reliable analysis")
        else:
            print("   ✓ Good scaling range for analysis")
        
        basic_results = {
            'level': level,
            'segments': len(segments),
            'use_optimization': use_optimization,
            'measured_dimension': optimal_dimension,
            'measurement_error': measurement_error,
            'theoretical_error': theoretical_error,
            'relative_error_percent': (theoretical_error/self.theoretical_dimension)*100,
            'r_squared': quality_r2,
            'optimal_window': optimal_window,
            'generation_time': generation_time,
            'analysis_time': analysis_time,
            'total_time': total_time,
            'scaling_decades': scaling_decades,
            'assessment': assessment,
            'box_sizes': box_sizes,
            'box_counts': box_counts
        }
        
        self.results['basic_calculation'] = basic_results
        return basic_results
    
    def optimization_comparison(self, level=4, create_plots=True):
        """
        Compare standard vs optimized methods for Hilbert curve analysis.
        
        Args:
            level: Hilbert curve level to analyze
            create_plots: Whether to create comparison plots
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*70}")
        print(f"HILBERT CURVE: STANDARD vs OPTIMIZED COMPARISON")
        print(f"Comparing baseline method with grid optimization")
        print(f"{'='*70}")
        
        # Generate Hilbert curve once
        points, segments = self.analyzer.generate_fractal('hilbert', level)
        print(f"Generated Hilbert curve with {len(segments)} segments")
        
        # Standard method (no optimization)
        print(f"\n1. Standard Method Analysis (No Optimization)...")
        standard_results = self.basic_dimension_calculation(
            level=level, 
            use_optimization=False, 
            create_plots=False
        )
        
        # Optimized method
        print(f"\n2. Optimized Method Analysis (Grid Optimization)...")
        optimized_results = self.basic_dimension_calculation(
            level=level, 
            use_optimization=True, 
            create_plots=False
        )
        
        # Comparison analysis
        print(f"\n3. Method Comparison:")
        print(f"   Standard Method:")
        print(f"     Dimension: {standard_results['measured_dimension']:.6f} ± {standard_results['measurement_error']:.6f}")
        print(f"     Error: {standard_results['theoretical_error']:.6f} ({standard_results['relative_error_percent']:.3f}%)")
        print(f"     R²: {standard_results['r_squared']:.6f}")
        print(f"     Time: {standard_results['total_time']:.3f}s")
        
        print(f"   Optimized Method:")
        print(f"     Dimension: {optimized_results['measured_dimension']:.6f} ± {optimized_results['measurement_error']:.6f}")
        print(f"     Error: {optimized_results['theoretical_error']:.6f} ({optimized_results['relative_error_percent']:.3f}%)")
        print(f"     R²: {optimized_results['r_squared']:.6f}")
        print(f"     Time: {optimized_results['total_time']:.3f}s")
        
        # Calculate improvements
        accuracy_improvement = standard_results['theoretical_error'] - optimized_results['theoretical_error']
        r2_improvement = optimized_results['r_squared'] - standard_results['r_squared']
        time_overhead = optimized_results['total_time'] - standard_results['total_time']
        time_overhead_percent = (time_overhead / standard_results['total_time']) * 100
        
        print(f"\n4. Optimization Benefits:")
        print(f"   Accuracy improvement: {accuracy_improvement:.6f}")
        print(f"   R² improvement: {r2_improvement:.6f}")
        print(f"   Time overhead: {time_overhead:.3f}s ({time_overhead_percent:.1f}%)")
        
        # Assessment
        if accuracy_improvement > 0.001 and r2_improvement > 0.001:
            benefit_assessment = "SIGNIFICANT - Clear benefits from optimization"
        elif accuracy_improvement > 0 and r2_improvement > 0:
            benefit_assessment = "MODEST - Some improvement from optimization"
        elif accuracy_improvement >= 0:
            benefit_assessment = "MINIMAL - Little benefit from optimization"
        else:
            benefit_assessment = "NONE - No clear benefit from optimization"
        
        print(f"   Overall benefit: {benefit_assessment}")
        
        # Create comparison plots
        if create_plots:
            self._create_comparison_plots(standard_results, optimized_results, 
                                        accuracy_improvement, r2_improvement, time_overhead_percent)
        
        comparison_results = {
            'level': level,
            'standard_results': standard_results,
            'optimized_results': optimized_results,
            'accuracy_improvement': accuracy_improvement,
            'r2_improvement': r2_improvement,
            'time_overhead': time_overhead,
            'time_overhead_percent': time_overhead_percent,
            'benefit_assessment': benefit_assessment
        }
        
        self.results['optimization_comparison'] = comparison_results
        return comparison_results
    
    def scaling_range_study(self, levels=[2, 3, 4, 5], create_plots=True):
        """
        Study scaling behavior across different Hilbert curve complexity levels.
        
        Args:
            levels: List of Hilbert curve levels to analyze
            create_plots: Whether to create scaling analysis plots
        
        Returns:
            Dictionary with scaling study results
        """
        print(f"\n{'='*70}")
        print(f"HILBERT CURVE SCALING RANGE STUDY")
        print(f"Testing scaling behavior across levels {levels}")
        print(f"{'='*70}")
        
        scaling_results = {
            'levels': [],
            'segment_counts': [],
            'dimensions': [],
            'errors': [],
            'theoretical_errors': [],
            'r_squared': [],
            'scaling_decades': [],
            'analysis_times': []
        }
        
        for level in levels:
            print(f"\n--- Analyzing Level {level} ---")
            
            try:
                # Basic calculation for this level
                level_results = self.basic_dimension_calculation(
                    level=level, 
                    use_optimization=True,  # Use optimization for best results
                    create_plots=False
                )
                
                # Store results
                scaling_results['levels'].append(level)
                scaling_results['segment_counts'].append(level_results['segments'])
                scaling_results['dimensions'].append(level_results['measured_dimension'])
                scaling_results['errors'].append(level_results['measurement_error'])
                scaling_results['theoretical_errors'].append(level_results['theoretical_error'])
                scaling_results['r_squared'].append(level_results['r_squared'])
                scaling_results['scaling_decades'].append(level_results['scaling_decades'])
                scaling_results['analysis_times'].append(level_results['analysis_time'])
                
                print(f"   Level {level}: D={level_results['measured_dimension']:.6f}, "
                      f"Error={level_results['theoretical_error']:.6f}, "
                      f"R²={level_results['r_squared']:.6f}")
                
            except Exception as e:
                print(f"   Level {level}: FAILED - {e}")
                continue
        
        # Scaling analysis
        if len(scaling_results['levels']) >= 3:
            print(f"\n5. Scaling Analysis Summary:")
            
            levels_array = np.array(scaling_results['levels'])
            dimensions_array = np.array(scaling_results['dimensions'])
            errors_array = np.array(scaling_results['theoretical_errors'])
            r2_array = np.array(scaling_results['r_squared'])
            
            # Consistency analysis
            dimension_std = np.std(dimensions_array)
            mean_dimension = np.mean(dimensions_array)
            cv_percent = (dimension_std / mean_dimension) * 100
            
            print(f"   Level range: {min(levels_array)} to {max(levels_array)}")
            print(f"   Dimension consistency: {mean_dimension:.6f} ± {dimension_std:.6f}")
            print(f"   Coefficient of variation: {cv_percent:.2f}%")
            print(f"   Mean error from theoretical: {np.mean(errors_array):.6f}")
            print(f"   Mean R²: {np.mean(r2_array):.6f}")
            
            # Quality assessment
            if cv_percent < 1.0 and np.mean(errors_array) < 0.02:
                consistency_assessment = "EXCELLENT - Very consistent across levels"
            elif cv_percent < 2.0 and np.mean(errors_array) < 0.05:
                consistency_assessment = "GOOD - Reasonably consistent"
            elif cv_percent < 5.0:
                consistency_assessment = "ACCEPTABLE - Some variation across levels"
            else:
                consistency_assessment = "POOR - High variation, check implementation"
            
            print(f"   Consistency assessment: {consistency_assessment}")
            
            # Best performing level
            best_idx = np.argmin(errors_array)
            best_level = levels_array[best_idx]
            best_error = errors_array[best_idx]
            
            print(f"   Best performing level: {best_level} (error: {best_error:.6f})")
        
        # Create scaling plots
        if create_plots and scaling_results['levels']:
            self._create_scaling_plots(scaling_results)
        
        study_results = {
            'scaling_data': scaling_results,
            'consistency_assessment': consistency_assessment if len(scaling_results['levels']) >= 3 else None,
            'best_level': best_level if len(scaling_results['levels']) >= 3 else None,
            'dimension_consistency': cv_percent if len(scaling_results['levels']) >= 3 else None
        }
        
        self.results['scaling_study'] = study_results
        return study_results
    
    def simple_workflow_demo(self, level=3):
        """
        Demonstrate the simplest possible fractal analysis workflow.
        
        Args:
            level: Hilbert curve level (kept low for simplicity)
        
        Returns:
            Dictionary with simple workflow results
        """
        print(f"\n{'='*70}")
        print(f"SIMPLE FRACTAL ANALYSIS WORKFLOW DEMONSTRATION")
        print(f"Step-by-step basic analysis for beginners")
        print(f"{'='*70}")
        
        print(f"Step 1: Generate fractal curve")
        points, segments = self.analyzer.generate_fractal('hilbert', level)
        print(f"   ✓ Generated Hilbert curve with {len(segments)} segments")
        
        print(f"\nStep 2: Set up analysis parameters")
        # Use default parameters - no optimization for simplicity
        print(f"   ✓ Using standard box counting (no optimization)")
        print(f"   ✓ Default box size factor: 1.5")
        print(f"   ✓ Default boundary trimming: 1 point")
        
        print(f"\nStep 3: Calculate fractal dimension")
        start_time = time.time()
        
        # Simple analysis call
        results = self.analyzer.analyze_linear_region(
            segments=segments,
            fractal_type='hilbert',
            plot_results=True,  # Show plots for educational value
            use_grid_optimization=False,  # Keep it simple
            return_box_data=False
        )
        
        analysis_time = time.time() - start_time
        optimal_dimension = results[5]
        
        print(f"   ✓ Analysis completed in {analysis_time:.3f} seconds")
        
        print(f"\nStep 4: Interpret results")
        theoretical_error = abs(optimal_dimension - self.theoretical_dimension)
        relative_error = (theoretical_error / self.theoretical_dimension) * 100
        
        print(f"   Measured dimension: {optimal_dimension:.6f}")
        print(f"   Expected dimension: {self.theoretical_dimension:.1f} (space-filling curve)")
        print(f"   Error: {theoretical_error:.6f} ({relative_error:.2f}%)")
        
        if relative_error < 5.0:
            print(f"   ✓ Good result! Error is within acceptable range.")
        else:
            print(f"   ⚠ High error - may need parameter adjustment or higher level.")
        
        print(f"\nStep 5: Summary")
        print(f"   This demonstrates the basic workflow for fractal analysis:")
        print(f"   1. Generate or load fractal curve data")
        print(f"   2. Choose analysis parameters")
        print(f"   3. Run box counting analysis")
        print(f"   4. Interpret dimension results")
        print(f"   5. Validate against known values when possible")
        
        simple_results = {
            'level': level,
            'segments': len(segments),
            'dimension': optimal_dimension,
            'theoretical_error': theoretical_error,
            'relative_error_percent': relative_error,
            'analysis_time': analysis_time
        }
        
        self.results['simple_workflow'] = simple_results
        return simple_results
    
    def _create_comparison_plots(self, standard_results, optimized_results, 
                               accuracy_improvement, r2_improvement, time_overhead_percent):
        """Create method comparison plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Dimension comparison
        methods = ['Standard\nMethod', 'Grid\nOptimization']
        dimensions = [standard_results['measured_dimension'], optimized_results['measured_dimension']]
        errors = [standard_results['measurement_error'], optimized_results['measurement_error']]
        
        bars = ax1.bar(methods, dimensions, yerr=errors, capsize=5, 
                      color=['blue', 'red'], alpha=0.7)
        ax1.axhline(y=self.theoretical_dimension, color='green', linestyle='--', linewidth=2,
                   label=f'Theoretical D = {self.theoretical_dimension:.1f}')
        ax1.set_ylabel('Fractal Dimension')
        ax1.set_title('Hilbert Curve: Method Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add dimension values on bars
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{dim:.6f}', ha='center', va='bottom')
        
        # Plot 2: Error comparison
        theoretical_errors = [standard_results['theoretical_error'], optimized_results['theoretical_error']]
        ax2.bar(methods, theoretical_errors, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('|Error from Theoretical|')
        ax2.set_title('Accuracy Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R-squared comparison
        r2_values = [standard_results['r_squared'], optimized_results['r_squared']]
        ax3.bar(methods, r2_values, color=['blue', 'red'], alpha=0.7)
        ax3.set_ylabel('R-squared Value')
        ax3.set_title('Analysis Quality Comparison')
        ax3.set_ylim(0.99, 1.001)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time comparison
        times = [standard_results['total_time'], optimized_results['total_time']]
        ax4.bar(methods, times, color=['blue', 'red'], alpha=0.7)
        ax4.set_ylabel('Computation Time (seconds)')
        ax4.set_title('Performance Comparison')
        ax4.grid(True, alpha=0.3)
        
        # Add overhead annotation
        ax4.text(0.5, max(times) * 0.8, 
                f'Overhead:\n{time_overhead_percent:.1f}%',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        self.analyzer._save_plot('hilbert_method_comparison')
        plt.close()
    
    def _create_scaling_plots(self, scaling_results):
        """Create scaling analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        levels = scaling_results['levels']
        
        # Plot 1: Dimension vs Level
        ax1.errorbar(levels, scaling_results['dimensions'], 
                    yerr=scaling_results['errors'], fmt='bo-', capsize=5, markersize=8)
        ax1.axhline(y=self.theoretical_dimension, color='red', linestyle='--',
                   label=f'Theoretical D = {self.theoretical_dimension:.1f}')
        ax1.set_xlabel('Hilbert Curve Level')
        ax1.set_ylabel('Measured Fractal Dimension')
        ax1.set_title('Dimension Consistency Across Levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error from theoretical vs Level
        ax2.semilogy(levels, scaling_results['theoretical_errors'], 'ro-', markersize=8)
        ax2.set_xlabel('Hilbert Curve Level')
        ax2.set_ylabel('|Error from Theoretical| (log scale)')
        ax2.set_title('Accuracy vs Complexity Level')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R-squared vs Level
        ax3.plot(levels, scaling_results['r_squared'], 'go-', markersize=8)
        ax3.set_xlabel('Hilbert Curve Level')
        ax3.set_ylabel('R-squared Value')
        ax3.set_title('Analysis Quality vs Level')
        ax3.set_ylim(0.98, 1.001)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scaling decades vs Level
        ax4.plot(levels, scaling_results['scaling_decades'], 'mo-', markersize=8)
        ax4.set_xlabel('Hilbert Curve Level')
        ax4.set_ylabel('Scaling Range (decades)')
        ax4.set_title('Scaling Range vs Level')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.analyzer._save_plot('hilbert_scaling_analysis')
        plt.close()


def main():
    """
    Main function for Hilbert curve scaling analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Hilbert Curve Scaling Range Analysis')
    parser.add_argument('--basic_analysis', action='store_true',
                       help='Run basic dimension calculation')
    parser.add_argument('--no_optimization', action='store_true',
                       help='Use standard method without grid optimization')
    parser.add_argument('--optimization_comparison', action='store_true',
                       help='Compare standard vs optimized methods')
    parser.add_argument('--scaling_study', action='store_true',
                       help='Study scaling behavior across complexity levels')
    parser.add_argument('--simple_workflow', action='store_true',
                       help='Demonstrate simple analysis workflow')
    parser.add_argument('--level', type=int, default=4,
                       help='Hilbert curve level for basic analysis (default: 4)')
    parser.add_argument('--max_level', type=int, default=5,
                       help='Maximum level for scaling study (default: 5)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--eps_plots', action='store_true',
                       help='Generate EPS plots for publication')
    parser.add_argument('--no_titles', action='store_true',
                       help='Disable plot titles for journal submission')
    
    args = parser.parse_args()
    
    # Create scaling analyzer
    analyzer = HilbertScalingAnalyzer(
        eps_plots=args.eps_plots,
        no_titles=args.no_titles
    )
    
    print("Hilbert Curve Scaling Range Analysis")
    print("=" * 60)
    print(f"Theoretical Hilbert dimension: {analyzer.theoretical_dimension:.1f} (exact, space-filling)")
    
    # Determine optimization setting
    use_optimization = not args.no_optimization
    
    # Run requested analyses
    if args.simple_workflow:
        print("\nRunning simple workflow demonstration...")
        simple_results = analyzer.simple_workflow_demo(level=3)
    
    elif args.basic_analysis or (not any([args.optimization_comparison, args.scaling_study, args.simple_workflow])):
        print(f"\nRunning basic dimension calculation...")
        basic_results = analyzer.basic_dimension_calculation(
            level=args.level, 
            use_optimization=use_optimization,
            create_plots=not args.no_plots
        )
    
    if args.optimization_comparison:
        print("\nRunning optimization comparison...")
        comparison_results = analyzer.optimization_comparison(
            level=args.level, 
            create_plots=not args.no_plots
        )
    
    if args.scaling_study:
        print("\nRunning scaling range study...")
        scaling_results = analyzer.scaling_range_study(
            levels=list(range(2, args.max_level + 1)), 
            create_plots=not args.no_plots
        )
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("HILBERT CURVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if 'basic_calculation' in analyzer.results:
        basic = analyzer.results['basic_calculation']
        print(f"Basic Analysis:")
        print(f"  Level: {basic['level']}")
        print(f"  Dimension: {basic['measured_dimension']:.6f}")
        print(f"  Error: {basic['theoretical_error']:.6f} ({basic['relative_error_percent']:.2f}%)")
        print(f"  Assessment: {basic['assessment']}")
    
    if 'optimization_comparison' in analyzer.results:
        comparison = analyzer.results['optimization_comparison']
        print(f"Optimization Comparison:")
        print(f"  Accuracy improvement: {comparison['accuracy_improvement']:.6f}")
        print(f"  Time overhead: {comparison['time_overhead_percent']:.1f}%")
        print(f"  Assessment: {comparison['benefit_assessment']}")
    
    if 'scaling_study' in analyzer.results:
        scaling = analyzer.results['scaling_study']
        print(f"Scaling Study:")
        if scaling['best_level']:
            print(f"  Best performing level: {scaling['best_level']}")
        if scaling['dimension_consistency']:
            print(f"  Dimension consistency: {scaling['dimension_consistency']:.2f}% CV")
        if scaling['consistency_assessment']:
            print(f"  Overall assessment: {scaling['consistency_assessment']}")
    
    print(f"\n💡 Hilbert curve demonstrates:")
    print(f"   • Space-filling curve analysis (theoretical maximum D = 2.0)")
    print(f"   • Standard vs optimized method comparison")
    print(f"   • Algorithm performance at extreme dimensions")
    print(f"   • Simple workflow for basic fractal analysis")
    print(f"   • Scaling range effects on measurement accuracy")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
