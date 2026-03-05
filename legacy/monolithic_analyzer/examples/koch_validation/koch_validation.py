#!/usr/bin/env python3
"""
Koch Curve Fractal Dimension Validation Example
===============================================

This example demonstrates rigorous validation of the fractal dimension algorithm
using the Koch curve, which has a known theoretical dimension of log(4)/log(3) ≈ 1.2619.

Key Features Demonstrated:
- Convergence analysis across iteration levels
- Grid optimization vs standard method comparison
- Statistical validation and error analysis
- Publication-quality validation plots
- Method accuracy assessment

Theoretical Background:
The Koch curve is a mathematical fractal with exact known properties:
- Theoretical dimension: D = log(4)/log(3) ≈ 1.26186
- Self-similar at all scales
- Infinite perimeter in finite area
- Perfect test case for fractal dimension algorithms

Usage:
    python koch_validation.py
    python koch_validation.py --max_level 8 --eps_plots
    python koch_validation.py --comparison_analysis --no_plots
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the fractal analyzer
try:
    from fractal_analyzer import FractalAnalyzer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from fractal_analyzer import FractalAnalyzer


class KochValidationSuite:
    """
    Comprehensive validation suite for fractal dimension analysis using Koch curves.
    """
    
    def __init__(self, eps_plots=False, no_titles=False):
        self.analyzer = FractalAnalyzer(
            fractal_type='koch',
            eps_plots=eps_plots,
            no_titles=no_titles
#            log_level='INFO'
        )
        self.theoretical_dimension = np.log(4) / np.log(3)  # ≈ 1.26186
        self.results = {}
    
    def validate_single_level(self, level=5, use_optimization=True, create_plots=True):
        """
        Validate fractal dimension calculation for a single Koch curve level.
        
        Args:
            level: Koch curve iteration level
            use_optimization: Whether to use grid optimization
            create_plots: Whether to generate plots
        
        Returns:
            Dictionary with validation results
        """
        print(f"\n{'='*60}")
        print(f"KOCH CURVE VALIDATION - LEVEL {level}")
        print(f"{'='*60}")
        
        # Generate Koch curve
        print(f"1. Generating Koch curve at level {level}...")
        points, segments = self.analyzer.generate_fractal('koch', level)
        
        print(f"   ✓ Generated {len(segments)} line segments")
        print(f"   ✓ Theoretical dimension: {self.theoretical_dimension:.6f}")
        
        # Perform fractal analysis
        method_name = "with grid optimization" if use_optimization else "standard method"
        print(f"2. Analyzing fractal dimension ({method_name})...")
        
        results = self.analyzer.analyze_linear_region(
            segments=segments,
            fractal_type='koch',
            plot_results=create_plots,
            plot_boxes=(level <= 4),  # Only show boxes for small levels
            use_grid_optimization=use_optimization,
            box_size_factor=1.5,
            trim_boundary=1,
            return_box_data=True
        )
        
        windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, optimal_intercept, box_sizes, box_counts, bounding_box = results
        
        # Calculate validation metrics
        error_from_theoretical = abs(optimal_dimension - self.theoretical_dimension)
        relative_error_percent = (error_from_theoretical / self.theoretical_dimension) * 100
        measurement_error = errors[windows.index(optimal_window)]
        quality_r2 = r_squared[windows.index(optimal_window)]
        
        # Validation assessment
        print(f"3. Validation Results:")
        print(f"   Measured dimension: {optimal_dimension:.6f} ± {measurement_error:.6f}")
        print(f"   Theoretical dimension: {self.theoretical_dimension:.6f}")
        print(f"   Absolute error: {error_from_theoretical:.6f}")
        print(f"   Relative error: {relative_error_percent:.3f}%")
        print(f"   R-squared: {quality_r2:.6f}")
        print(f"   Optimal window: {optimal_window} points")
        
        # Validation status
        if relative_error_percent < 1.0 and quality_r2 > 0.99:
            status = "EXCELLENT"
        elif relative_error_percent < 2.0 and quality_r2 > 0.98:
            status = "GOOD"
        elif relative_error_percent < 5.0 and quality_r2 > 0.95:
            status = "ACCEPTABLE"
        else:
            status = "POOR"
        
        print(f"   Validation status: {status}")
        
        # Store results
        validation_result = {
            'level': level,
            'segments': len(segments),
            'measured_dimension': optimal_dimension,
            'measurement_error': measurement_error,
            'theoretical_dimension': self.theoretical_dimension,
            'absolute_error': error_from_theoretical,
            'relative_error_percent': relative_error_percent,
            'r_squared': quality_r2,
            'optimal_window': optimal_window,
            'status': status,
            'use_optimization': use_optimization,
            'box_sizes': box_sizes,
            'box_counts': box_counts
        }
        
        return validation_result
    
    def convergence_analysis(self, min_level=1, max_level=7, create_plots=True):
        """
        Analyze convergence of measured dimension as iteration level increases.
        
        Args:
            min_level: Minimum Koch curve level to test
            max_level: Maximum Koch curve level to test
            create_plots: Whether to create convergence plots
        
        Returns:
            Dictionary with convergence analysis results
        """
        print(f"\n{'='*60}")
        print(f"KOCH CURVE CONVERGENCE ANALYSIS")
        print(f"Testing levels {min_level} to {max_level}")
        print(f"{'='*60}")
        
        levels = []
        dimensions = []
        errors = []
        r_squared_values = []
        segment_counts = []
        
        for level in range(min_level, max_level + 1):
            print(f"\nTesting level {level}...")
            
            try:
                result = self.validate_single_level(
                    level=level, 
                    use_optimization=True, 
                    create_plots=False
                )
                
                levels.append(level)
                dimensions.append(result['measured_dimension'])
                errors.append(result['measurement_error'])
                r_squared_values.append(result['r_squared'])
                segment_counts.append(result['segments'])
                
                print(f"   Level {level}: D = {result['measured_dimension']:.6f} ± {result['measurement_error']:.6f}")
                print(f"   Error: {result['relative_error_percent']:.2f}%, R² = {result['r_squared']:.6f}")
                
            except Exception as e:
                print(f"   Level {level}: FAILED - {e}")
                continue
        
        # Convergence analysis
        if len(levels) >= 3:
            print(f"\nConvergence Analysis:")
            
            # Check if dimensions are converging to theoretical value
            dimension_trend = np.array(dimensions)
            theoretical_errors = np.abs(dimension_trend - self.theoretical_dimension)
            
            # Is error decreasing with level?
            is_converging = len(theoretical_errors) > 1 and theoretical_errors[-1] < theoretical_errors[0]
            
            # Calculate convergence metrics
            final_error = theoretical_errors[-1] if len(theoretical_errors) > 0 else float('inf')
            mean_dimension = np.mean(dimensions)
            std_dimension = np.std(dimensions)
            
            print(f"   Final error from theoretical: {final_error:.6f}")
            print(f"   Mean dimension: {mean_dimension:.6f}")
            print(f"   Standard deviation: {std_dimension:.6f}")
            print(f"   Converging trend: {'YES' if is_converging else 'NO'}")
            
            # Best level (closest to theoretical)
            best_idx = np.argmin(theoretical_errors)
            best_level = levels[best_idx]
            best_dimension = dimensions[best_idx]
            
            print(f"   Best level: {best_level} (D = {best_dimension:.6f})")
        
        # Create convergence plots
        if create_plots and len(levels) >= 2:
            self._create_convergence_plots(levels, dimensions, errors, r_squared_values, segment_counts)
        
        convergence_results = {
            'levels': levels,
            'dimensions': dimensions,
            'errors': errors,
            'r_squared': r_squared_values,
            'segment_counts': segment_counts,
            'theoretical_dimension': self.theoretical_dimension,
            'is_converging': is_converging if len(levels) >= 3 else None,
            'best_level': best_level if len(levels) >= 3 else None,
            'final_error': final_error if len(levels) >= 3 else None
        }
        
        self.results['convergence'] = convergence_results
        return convergence_results
    
    def method_comparison(self, level=5, create_plots=True):
        """
        Compare grid optimization vs standard box counting method.
        
        Args:
            level: Koch curve level to use for comparison
            create_plots: Whether to create comparison plots
        
        Returns:
            Dictionary with comparison results
        """
        print(f"\n{'='*60}")
        print(f"METHOD COMPARISON ANALYSIS")
        print(f"Grid Optimization vs Standard Box Counting")
        print(f"{'='*60}")
        
        # Test with grid optimization
        print("1. Testing with grid optimization...")
        result_optimized = self.validate_single_level(
            level=level, 
            use_optimization=True, 
            create_plots=False
        )
        
        # Test without grid optimization
        print("2. Testing standard method...")
        result_standard = self.validate_single_level(
            level=level, 
            use_optimization=False, 
            create_plots=False
        )
        
        # Comparison analysis
        print(f"\n3. Method Comparison Results:")
        print(f"   Grid Optimization:")
        print(f"     Dimension: {result_optimized['measured_dimension']:.6f} ± {result_optimized['measurement_error']:.6f}")
        print(f"     Error from theoretical: {result_optimized['relative_error_percent']:.3f}%")
        print(f"     R-squared: {result_optimized['r_squared']:.6f}")
        
        print(f"   Standard Method:")
        print(f"     Dimension: {result_standard['measured_dimension']:.6f} ± {result_standard['measurement_error']:.6f}")
        print(f"     Error from theoretical: {result_standard['relative_error_percent']:.3f}%")
        print(f"     R-squared: {result_standard['r_squared']:.6f}")
        
        # Calculate improvement metrics
        error_improvement = result_standard['absolute_error'] - result_optimized['absolute_error']
        r2_improvement = result_optimized['r_squared'] - result_standard['r_squared']
        
        print(f"\n   Improvement from Grid Optimization:")
        print(f"     Error reduction: {error_improvement:.6f}")
        print(f"     R-squared improvement: {r2_improvement:.6f}")
        
        better_method = "Grid Optimization" if result_optimized['absolute_error'] < result_standard['absolute_error'] else "Standard"
        print(f"     Better method: {better_method}")
        
        # Create comparison plots if requested
        if create_plots:
            self._create_method_comparison_plots(result_optimized, result_standard)
        
        comparison_results = {
            'optimized': result_optimized,
            'standard': result_standard,
            'error_improvement': error_improvement,
            'r2_improvement': r2_improvement,
            'better_method': better_method
        }
        
        self.results['comparison'] = comparison_results
        return comparison_results
    
    def statistical_validation(self, level=5, n_trials=10):
        """
        Perform statistical validation by running multiple analyses with slight variations.
        
        Args:
            level: Koch curve level to use
            n_trials: Number of trials to run
        
        Returns:
            Dictionary with statistical validation results
        """
        print(f"\n{'='*60}")
        print(f"STATISTICAL VALIDATION")
        print(f"Running {n_trials} trials with parameter variations")
        print(f"{'='*60}")
        
        dimensions = []
        errors = []
        r_squared_values = []
        
        # Parameters to vary slightly
        box_factors = np.linspace(1.3, 1.7, n_trials)
        trim_values = np.random.choice([0, 1, 2], n_trials)
        
        for i in range(n_trials):
            print(f"Trial {i+1}/{n_trials}...")
            
            try:
                # Generate Koch curve (same each time)
                points, segments = self.analyzer.generate_fractal('koch', level)
                
                # Analyze with slight parameter variations
                results = self.analyzer.analyze_linear_region(
                    segments=segments,
                    fractal_type='koch',
                    plot_results=False,
                    box_size_factor=box_factors[i],
                    trim_boundary=trim_values[i],
                    use_grid_optimization=True,
                    return_box_data=False
                )
                
                windows, dims, errs, r2s, optimal_window, optimal_dimension, optimal_intercept = results
                
                dimensions.append(optimal_dimension)
                errors.append(errs[windows.index(optimal_window)])
                r_squared_values.append(r2s[windows.index(optimal_window)])
                
            except Exception as e:
                print(f"   Trial {i+1} failed: {e}")
                continue
        
        if len(dimensions) >= 3:
            # Statistical analysis
            dimensions = np.array(dimensions)
            mean_dimension = np.mean(dimensions)
            std_dimension = np.std(dimensions)
            min_dimension = np.min(dimensions)
            max_dimension = np.max(dimensions)
            
            # Confidence interval (95%)
            sem = std_dimension / np.sqrt(len(dimensions))
            ci_95 = 1.96 * sem
            
            # Error from theoretical
            theoretical_errors = np.abs(dimensions - self.theoretical_dimension)
            mean_error = np.mean(theoretical_errors)
            max_error = np.max(theoretical_errors)
            
            print(f"\nStatistical Results:")
            print(f"   Mean dimension: {mean_dimension:.6f} ± {std_dimension:.6f}")
            print(f"   95% Confidence interval: [{mean_dimension-ci_95:.6f}, {mean_dimension+ci_95:.6f}]")
            print(f"   Range: [{min_dimension:.6f}, {max_dimension:.6f}]")
            print(f"   Theoretical dimension: {self.theoretical_dimension:.6f}")
            print(f"   Mean error from theoretical: {mean_error:.6f}")
            print(f"   Maximum error: {max_error:.6f}")
            print(f"   Coefficient of variation: {(std_dimension/mean_dimension)*100:.2f}%")
            
            # Assessment
            if std_dimension < 0.01 and mean_error < 0.01:
                assessment = "EXCELLENT - Very stable and accurate"
            elif std_dimension < 0.02 and mean_error < 0.02:
                assessment = "GOOD - Stable with good accuracy"
            elif std_dimension < 0.05 and mean_error < 0.05:
                assessment = "ACCEPTABLE - Reasonably stable"
            else:
                assessment = "POOR - High variability or systematic error"
            
            print(f"   Assessment: {assessment}")
        
        statistical_results = {
            'n_trials': len(dimensions),
            'dimensions': dimensions,
            'mean_dimension': mean_dimension,
            'std_dimension': std_dimension,
            'confidence_interval_95': [mean_dimension-ci_95, mean_dimension+ci_95],
            'range': [min_dimension, max_dimension],
            'mean_error_from_theoretical': mean_error,
            'max_error': max_error,
            'coefficient_of_variation': (std_dimension/mean_dimension)*100,
            'assessment': assessment
        }
        
        self.results['statistical'] = statistical_results
        return statistical_results
    
    def _create_convergence_plots(self, levels, dimensions, errors, r_squared, segment_counts):
        """Create convergence analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Dimension vs Level
        ax1.errorbar(levels, dimensions, yerr=errors, fmt='bo-', capsize=5, markersize=6)
        ax1.axhline(y=self.theoretical_dimension, color='red', linestyle='--', linewidth=2,
                   label=f'Theoretical D = {self.theoretical_dimension:.6f}')
        ax1.set_xlabel('Koch Curve Level')
        ax1.set_ylabel('Measured Fractal Dimension')
        ax1.set_title('Convergence: Dimension vs Iteration Level')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Error from theoretical vs Level
        theoretical_errors = [abs(d - self.theoretical_dimension) for d in dimensions]
        ax2.semilogy(levels, theoretical_errors, 'ro-', markersize=6)
        ax2.set_xlabel('Koch Curve Level')
        ax2.set_ylabel('|Error from Theoretical| (log scale)')
        ax2.set_title('Error Convergence')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: R-squared vs Level
        ax3.plot(levels, r_squared, 'go-', markersize=6)
        ax3.set_xlabel('Koch Curve Level')
        ax3.set_ylabel('R-squared Value')
        ax3.set_title('Analysis Quality vs Level')
        ax3.set_ylim(0.95, 1.001)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Segments vs Level
        ax4.semilogy(levels, segment_counts, 'mo-', markersize=6)
        ax4.set_xlabel('Koch Curve Level')
        ax4.set_ylabel('Number of Segments (log scale)')
        ax4.set_title('Complexity vs Level')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.analyzer._save_plot('koch_convergence_analysis')
        plt.close()
    
    def _create_method_comparison_plots(self, result_optimized, result_standard):
        """Create method comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Dimension comparison
        methods = ['Grid\nOptimization', 'Standard\nMethod']
        dimensions = [result_optimized['measured_dimension'], result_standard['measured_dimension']]
        errors = [result_optimized['measurement_error'], result_standard['measurement_error']]
        
        bars = ax1.bar(methods, dimensions, yerr=errors, capsize=5, 
                      color=['blue', 'orange'], alpha=0.7)
        ax1.axhline(y=self.theoretical_dimension, color='red', linestyle='--', linewidth=2,
                   label=f'Theoretical D = {self.theoretical_dimension:.6f}')
        ax1.set_ylabel('Fractal Dimension')
        ax1.set_title('Method Comparison: Measured Dimension')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, dim, err in zip(bars, dimensions, errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + err + 0.001,
                    f'{dim:.6f}', ha='center', va='bottom')
        
        # Plot 2: Error comparison
        error_types = ['Absolute\nError', 'R-squared']
        opt_values = [result_optimized['absolute_error'], result_optimized['r_squared']]
        std_values = [result_standard['absolute_error'], result_standard['r_squared']]
        
        x = np.arange(len(error_types))
        width = 0.35
        
        ax2.bar(x - width/2, opt_values, width, label='Grid Optimization', 
               color='blue', alpha=0.7)
        ax2.bar(x + width/2, std_values, width, label='Standard Method',
               color='orange', alpha=0.7)
        
        ax2.set_ylabel('Value')
        ax2.set_title('Method Comparison: Error Metrics')
        ax2.set_xticks(x)
        ax2.set_xticklabels(error_types)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.analyzer._save_plot('koch_method_comparison')
        plt.close()


def main():
    """
    Main function demonstrating comprehensive Koch curve validation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Koch Curve Validation Example')
    parser.add_argument('--max_level', type=int, default=6,
                       help='Maximum Koch curve level to test (default: 6)')
    parser.add_argument('--comparison_analysis', action='store_true',
                       help='Run method comparison analysis')
    parser.add_argument('--statistical_validation', action='store_true',
                       help='Run statistical validation with multiple trials')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--eps_plots', action='store_true',
                       help='Generate EPS plots for publication')
    parser.add_argument('--no_titles', action='store_true',
                       help='Disable plot titles for journal submission')
    
    args = parser.parse_args()
    
    # Create validation suite
    validator = KochValidationSuite(
        eps_plots=args.eps_plots,
        no_titles=args.no_titles
    )
    
    print("Koch Curve Fractal Dimension Validation Suite")
    print("=" * 60)
    print(f"Theoretical Koch dimension: {validator.theoretical_dimension:.6f}")
    
    # Run convergence analysis
    print("\nRunning convergence analysis...")
    convergence_results = validator.convergence_analysis(
        min_level=1, 
        max_level=args.max_level, 
        create_plots=not args.no_plots
    )
    
    # Run method comparison if requested
    if args.comparison_analysis:
        print("\nRunning method comparison...")
        comparison_results = validator.method_comparison(
            level=5, 
            create_plots=not args.no_plots
        )
    
    # Run statistical validation if requested
    if args.statistical_validation:
        print("\nRunning statistical validation...")
        statistical_results = validator.statistical_validation(level=5, n_trials=10)
    
    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if convergence_results['levels']:
        best_level = convergence_results.get('best_level')
        final_error = convergence_results.get('final_error')
        
        print(f"✓ Tested levels: {min(convergence_results['levels'])} to {max(convergence_results['levels'])}")
        print(f"✓ Best performing level: {best_level}")
        print(f"✓ Final error from theoretical: {final_error:.6f}")
        
        if final_error < 0.01:
            print("✓ Validation status: EXCELLENT - Algorithm is highly accurate")
        elif final_error < 0.02:
            print("✓ Validation status: GOOD - Algorithm performs well")
        else:
            print("⚠ Validation status: Needs improvement")
    
    if args.comparison_analysis and 'comparison' in validator.results:
        comp = validator.results['comparison']
        print(f"✓ Method comparison: {comp['better_method']} performs better")
        print(f"✓ Grid optimization error reduction: {comp['error_improvement']:.6f}")
    
    print(f"\n💡 Recommendation: Use Koch curves at level 5-6 for algorithm validation")
    print(f"💡 Expected accuracy: < 1% error from theoretical dimension")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
