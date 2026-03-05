#!/usr/bin/env python3
"""
Dragon Curve Algorithm Robustness Analysis
==========================================

This example demonstrates algorithm robustness testing using the Dragon curve,
which has complex geometric properties that challenge fractal dimension algorithms.

Key Features Demonstrated:
- Algorithm stress testing with complex geometry
- Robustness to parameter variations
- Boundary artifact handling in folded structures
- Performance with asymmetric self-similarity
- Window size sensitivity analysis

Dragon Curve Properties:
- Theoretical dimension: ~1.5236 (Hausdorff dimension)
- Complex folding pattern with self-intersections at boundaries
- Asymmetric self-similarity (unlike Koch/Sierpinski)  
- Space-filling tendencies without being truly space-filling
- Challenging test case for linear region identification

Usage:
    python dragon_robustness.py
    python dragon_robustness.py --stress_test --max_level 8
    python dragon_robustness.py --parameter_sweep --eps_plots
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools

# Import the fractal analyzer
try:
    from fractal_analyzer import FractalAnalyzer
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from fractal_analyzer import FractalAnalyzer


class DragonRobustnessAnalyzer:
    """
    Comprehensive robustness testing suite using Dragon curve complexity.
    """
    
    def __init__(self, eps_plots=False, no_titles=False):
        self.analyzer = FractalAnalyzer(
            fractal_type='dragon',
            eps_plots=eps_plots,
            no_titles=no_titles
        )
        # Dragon curve Hausdorff dimension (approximate)
        self.theoretical_dimension = 1.5236  
        self.results = {}
    
    def parameter_robustness_test(self, level=6, create_plots=True):
        """
        Test algorithm robustness to parameter variations using Dragon curve.
        
        Args:
            level: Dragon curve level to use
            create_plots: Whether to create robustness plots
        
        Returns:
            Dictionary with robustness test results
        """
        print(f"\n{'='*70}")
        print(f"DRAGON CURVE PARAMETER ROBUSTNESS TEST")
        print(f"Testing algorithm stability with parameter variations")
        print(f"{'='*70}")
        
        # Generate Dragon curve once
        print(f"1. Generating Dragon curve at level {level}...")
        points, segments = self.analyzer.generate_fractal('dragon', level)
        print(f"   ✓ Generated {len(segments)} segments with complex folding pattern")
        
        # Define parameter ranges to test
        test_parameters = {
            'box_size_factor': [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0],
            'trim_boundary': [0, 1, 2, 3, 4],
            'min_box_size_multiplier': [0.5, 1.0, 2.0, 4.0, 8.0]  # Multiplier for auto-estimated
        }
        
        results = {param: {'values': [], 'dimensions': [], 'errors': [], 'r_squared': []} 
                  for param in test_parameters}
        
        # Test each parameter independently
        for param_name, param_values in test_parameters.items():
            print(f"\n2. Testing {param_name} sensitivity...")
            
            for value in param_values:
                try:
                    # Set test parameters (keeping others at defaults)
                    kwargs = {
                        'segments': segments,
                        'fractal_type': 'dragon',
                        'plot_results': False,
                        'use_grid_optimization': True,
                        'return_box_data': False
                    }
                    
                    if param_name == 'box_size_factor':
                        kwargs['box_size_factor'] = value
                    elif param_name == 'trim_boundary':
                        kwargs['trim_boundary'] = value
                    elif param_name == 'min_box_size_multiplier':
                        # Get auto-estimated min box size and multiply
                        auto_min = self.analyzer.estimate_min_box_size_from_segments(segments, percentile=5, multiplier=value)
                        kwargs['min_box_size'] = auto_min
                    
                    # Run analysis
                    analysis_results = self.analyzer.analyze_linear_region(**kwargs)
                    windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, optimal_intercept = analysis_results
                    
                    optimal_error = errors[windows.index(optimal_window)]
                    optimal_r2 = r_squared[windows.index(optimal_window)]
                    
                    # Store results
                    results[param_name]['values'].append(value)
                    results[param_name]['dimensions'].append(optimal_dimension)
                    results[param_name]['errors'].append(optimal_error)
                    results[param_name]['r_squared'].append(optimal_r2)
                    
                    print(f"   {param_name}={value}: D={optimal_dimension:.6f} ± {optimal_error:.6f}, R²={optimal_r2:.6f}")
                    
                except Exception as e:
                    print(f"   {param_name}={value}: FAILED - {e}")
                    continue
        
        # Analyze robustness
        print(f"\n3. Robustness Analysis:")
        robustness_summary = {}
        
        for param_name, param_results in results.items():
            if len(param_results['dimensions']) >= 3:
                dimensions = np.array(param_results['dimensions'])
                
                # Calculate stability metrics
                mean_dim = np.mean(dimensions)
                std_dim = np.std(dimensions)
                range_dim = np.max(dimensions) - np.min(dimensions)
                cv_percent = (std_dim / mean_dim) * 100  # Coefficient of variation
                
                # Distance from theoretical
                theoretical_errors = np.abs(dimensions - self.theoretical_dimension)
                mean_theoretical_error = np.mean(theoretical_errors)
                
                print(f"   {param_name}:")
                print(f"     Mean dimension: {mean_dim:.6f} ± {std_dim:.6f}")
                print(f"     Range: {range_dim:.6f}")
                print(f"     Coefficient of variation: {cv_percent:.2f}%")
                print(f"     Mean error from theoretical: {mean_theoretical_error:.6f}")
                
                # Robustness assessment
                if cv_percent < 1.0 and mean_theoretical_error < 0.05:
                    assessment = "EXCELLENT"
                elif cv_percent < 2.0 and mean_theoretical_error < 0.1:
                    assessment = "GOOD"
                elif cv_percent < 5.0 and mean_theoretical_error < 0.2:
                    assessment = "ACCEPTABLE"
                else:
                    assessment = "POOR"
                
                print(f"     Robustness: {assessment}")
                
                robustness_summary[param_name] = {
                    'mean_dimension': mean_dim,
                    'std_dimension': std_dim,
                    'range': range_dim,
                    'cv_percent': cv_percent,
                    'mean_theoretical_error': mean_theoretical_error,
                    'assessment': assessment
                }
        
        # Create robustness plots
        if create_plots:
            self._create_robustness_plots(results, robustness_summary)
        
        robustness_results = {
            'level': level,
            'segments': len(segments),
            'parameter_results': results,
            'robustness_summary': robustness_summary,
            'theoretical_dimension': self.theoretical_dimension
        }
        
        self.results['robustness'] = robustness_results
        return robustness_results
    
    def boundary_complexity_analysis(self, level=6, create_plots=True):
        """
        Analyze how Dragon curve's complex boundary structure affects analysis.
        
        Args:
            level: Dragon curve level
            create_plots: Whether to create analysis plots
        
        Returns:
            Dictionary with boundary analysis results
        """
        print(f"\n{'='*70}")
        print(f"DRAGON CURVE BOUNDARY COMPLEXITY ANALYSIS")
        print(f"Testing boundary artifact handling with folded structures")
        print(f"{'='*70}")
        
        # Generate Dragon curve
        points, segments = self.analyzer.generate_fractal('dragon', level)
        
        # Test different boundary trimming strategies
        trim_strategies = [0, 1, 2, 3, 4, 5]
        results = {'trim_values': [], 'dimensions': [], 'errors': [], 'r_squared': [], 'window_sizes': []}
        
        print(f"1. Testing boundary trimming sensitivity...")
        
        for trim_val in trim_strategies:
            try:
                analysis_results = self.analyzer.analyze_linear_region(
                    segments=segments,
                    fractal_type='dragon',
                    plot_results=False,
                    trim_boundary=trim_val,
                    use_grid_optimization=True,
                    return_box_data=False
                )
                
                windows, dimensions, errors, r_squared, optimal_window, optimal_dimension = analysis_results
                optimal_error = errors[windows.index(optimal_window)]
                optimal_r2 = r_squared[windows.index(optimal_window)]
                
                results['trim_values'].append(trim_val)
                results['dimensions'].append(optimal_dimension)
                results['errors'].append(optimal_error)
                results['r_squared'].append(optimal_r2)
                results['window_sizes'].append(optimal_window)
                
                error_from_theoretical = abs(optimal_dimension - self.theoretical_dimension)
                print(f"   Trim {trim_val}: D={optimal_dimension:.6f} ± {optimal_error:.6f}, "
                      f"Error={error_from_theoretical:.6f}, R²={optimal_r2:.6f}")
                
            except Exception as e:
                print(f"   Trim {trim_val}: FAILED - {e}")
                continue
        
        # Enhanced boundary detection test
        print(f"\n2. Testing enhanced vs manual boundary detection...")
        
        # Manual trimming
        manual_results = self.analyzer.analyze_linear_region(
            segments=segments,
            fractal_type='dragon',
            plot_results=False,
            trim_boundary=2,  # Fixed manual trim
            use_grid_optimization=True,
            return_box_data=False
        )
        
        # Enhanced automatic detection
        enhanced_results = self.analyzer.analyze_linear_region(
            segments=segments,
            fractal_type='dragon',
            plot_results=False,
            trim_boundary=0,  # Let enhanced detection work
            use_grid_optimization=True,
            return_box_data=False
        )
        
        manual_dimension = manual_results[5]
        enhanced_dimension = enhanced_results[5]
        
        print(f"   Manual boundary trimming: D={manual_dimension:.6f}")
        print(f"   Enhanced detection: D={enhanced_dimension:.6f}")
        print(f"   Difference: {abs(enhanced_dimension - manual_dimension):.6f}")
        
        # Determine which performed better
        manual_error = abs(manual_dimension - self.theoretical_dimension)
        enhanced_error = abs(enhanced_dimension - self.theoretical_dimension)
        
        better_method = "Enhanced" if enhanced_error < manual_error else "Manual"
        print(f"   Better method: {better_method} (closer to theoretical)")
        
        # Create boundary analysis plots
        if create_plots and results['dimensions']:
            self._create_boundary_analysis_plots(results, manual_dimension, enhanced_dimension)
        
        boundary_results = {
            'trim_sensitivity': results,
            'manual_dimension': manual_dimension,
            'enhanced_dimension': enhanced_dimension,
            'better_method': better_method,
            'manual_error': manual_error,
            'enhanced_error': enhanced_error
        }
        
        self.results['boundary'] = boundary_results
        return boundary_results
    
    def algorithmic_stress_test(self, max_level=8, create_plots=True):
        """
        Stress test the algorithm with increasing Dragon curve complexity.
        
        Args:
            max_level: Maximum Dragon curve level to test
            create_plots: Whether to create stress test plots
        
        Returns:
            Dictionary with stress test results
        """
        print(f"\n{'='*70}")
        print(f"DRAGON CURVE ALGORITHMIC STRESS TEST")
        print(f"Testing performance with increasing geometric complexity")
        print(f"{'='*70}")
        
        levels = list(range(1, max_level + 1))
        results = {
            'levels': [],
            'segment_counts': [],
            'dimensions': [],
            'errors': [],
            'r_squared': [],
            'computation_times': [],
            'optimal_windows': []
        }
        
        for level in levels:
            print(f"\n--- Stress Testing Level {level} ---")
            
            try:
                import time
                start_time = time.time()
                
                # Generate Dragon curve
                points, segments = self.analyzer.generate_fractal('dragon', level)
                generation_time = time.time() - start_time
                
                print(f"   Generated {len(segments)} segments in {generation_time:.3f}s")
                
                # Analyze fractal dimension
                analysis_start = time.time()
                analysis_results = self.analyzer.analyze_linear_region(
                    segments=segments,
                    fractal_type='dragon',
                    plot_results=False,
                    use_grid_optimization=True,
                    box_size_factor=1.4,  # Slightly more conservative for stress test
                    return_box_data=False
                )
                analysis_time = time.time() - analysis_start
                total_time = time.time() - start_time
                
                windows, dimensions, errors, r_squared, optimal_window, optimal_dimension, optimal_intercept = analysis_results
                optimal_error = errors[windows.index(optimal_window)]
                optimal_r2 = r_squared[windows.index(optimal_window)]
                
                # Store results
                results['levels'].append(level)
                results['segment_counts'].append(len(segments))
                results['dimensions'].append(optimal_dimension)
                results['errors'].append(optimal_error)
                results['r_squared'].append(optimal_r2)
                results['computation_times'].append(total_time)
                results['optimal_windows'].append(optimal_window)
                
                # Analysis
                error_from_theoretical = abs(optimal_dimension - self.theoretical_dimension)
                
                print(f"   Dimension: {optimal_dimension:.6f} ± {optimal_error:.6f}")
                print(f"   Error from theoretical: {error_from_theoretical:.6f}")
                print(f"   R-squared: {optimal_r2:.6f}")
                print(f"   Computation time: {total_time:.3f}s")
                print(f"   Analysis time: {analysis_time:.3f}s")
                
                # Performance assessment
                if total_time < 1.0 and optimal_r2 > 0.98:
                    status = "EXCELLENT"
                elif total_time < 5.0 and optimal_r2 > 0.95:
                    status = "GOOD"
                elif total_time < 30.0 and optimal_r2 > 0.90:
                    status = "ACCEPTABLE"
                else:
                    status = "POOR"
                
                print(f"   Performance: {status}")
                
            except Exception as e:
                print(f"   Level {level}: FAILED - {e}")
                continue
        
        # Performance analysis
        if len(results['levels']) >= 3:
            print(f"\n3. Stress Test Analysis:")
            
            segment_counts = np.array(results['segment_counts'])
            times = np.array(results['computation_times'])
            dimensions = np.array(results['dimensions'])
            r_squares = np.array(results['r_squared'])
            
            # Scaling analysis
            # Theoretical: Dragon curve has 2^n segments at level n
            theoretical_segments = [2**level for level in results['levels']]
            
            print(f"   Complexity scaling:")
            print(f"     Level range: {min(results['levels'])} to {max(results['levels'])}")
            print(f"     Segment range: {min(segment_counts)} to {max(segment_counts)}")
            print(f"     Time range: {min(times):.3f}s to {max(times):.3f}s")
            
            # Performance metrics
            dimension_stability = np.std(dimensions)
            r2_degradation = max(r_squares) - min(r_squares)
            
            print(f"   Algorithm stability:")
            print(f"     Dimension std dev: {dimension_stability:.6f}")
            print(f"     R² degradation: {r2_degradation:.6f}")
            
            # Time complexity estimation
            if len(times) >= 3:
                # Fit power law: time = a * segments^b
                log_segments = np.log(segment_counts)
                log_times = np.log(times)
                coeffs = np.polyfit(log_segments, log_times, 1)
                complexity_exponent = coeffs[0]
                
                print(f"   Time complexity: O(n^{complexity_exponent:.2f})")
                
                if complexity_exponent < 1.2:
                    complexity_assessment = "EXCELLENT - Nearly linear scaling"
                elif complexity_exponent < 1.5:
                    complexity_assessment = "GOOD - Sub-quadratic scaling"
                elif complexity_exponent < 2.0:
                    complexity_assessment = "ACCEPTABLE - Reasonable scaling"
                else:
                    complexity_assessment = "POOR - High complexity scaling"
                
                print(f"   Complexity assessment: {complexity_assessment}")
        
        # Create stress test plots
        if create_plots and results['levels']:
            self._create_stress_test_plots(results)
        
        stress_results = {
            'test_results': results,
            'complexity_exponent': complexity_exponent if len(results['levels']) >= 3 else None,
            'dimension_stability': dimension_stability if len(results['levels']) >= 3 else None,
            'max_level_tested': max(results['levels']) if results['levels'] else 0
        }
        
        self.results['stress_test'] = stress_results
        return stress_results
    
    def _create_robustness_plots(self, results, summary):
        """Create parameter robustness plots."""
        n_params = len([p for p in results.keys() if results[p]['dimensions']])
        if n_params == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for param_name, param_results in results.items():
            if not param_results['dimensions'] or plot_idx >= 4:
                continue
                
            ax = axes[plot_idx]
            
            # Plot dimension vs parameter value
            ax.errorbar(param_results['values'], param_results['dimensions'], 
                       yerr=param_results['errors'], fmt='o-', capsize=5, markersize=6)
            ax.axhline(y=self.theoretical_dimension, color='red', linestyle='--', 
                      label=f'Theoretical D = {self.theoretical_dimension:.4f}')
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Fractal Dimension')
            ax.set_title(f'Robustness: {param_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        self.analyzer._save_plot('dragon_parameter_robustness')
        plt.close()
    
    def _create_boundary_analysis_plots(self, results, manual_dim, enhanced_dim):
        """Create boundary complexity analysis plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Dimension vs trim value
        ax1.errorbar(results['trim_values'], results['dimensions'], 
                    yerr=results['errors'], fmt='o-', capsize=5, markersize=6)
        ax1.axhline(y=self.theoretical_dimension, color='red', linestyle='--',
                   label=f'Theoretical D = {self.theoretical_dimension:.4f}')
        ax1.set_xlabel('Boundary Trim Value')
        ax1.set_ylabel('Fractal Dimension')
        ax1.set_title('Dragon Curve: Boundary Sensitivity')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Method comparison
        methods = ['Manual\nTrimming', 'Enhanced\nDetection']
        dimensions = [manual_dim, enhanced_dim]
        theoretical_errors = [abs(d - self.theoretical_dimension) for d in dimensions]
        
        bars = ax2.bar(methods, dimensions, color=['orange', 'blue'], alpha=0.7)
        ax2.axhline(y=self.theoretical_dimension, color='red', linestyle='--',
                   label=f'Theoretical D = {self.theoretical_dimension:.4f}')
        ax2.set_ylabel('Fractal Dimension')
        ax2.set_title('Boundary Detection Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add error values on bars
        for bar, dim, error in zip(bars, dimensions, theoretical_errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{dim:.6f}\n(error: {error:.6f})',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        self.analyzer._save_plot('dragon_boundary_analysis')
        plt.close()
    
    def _create_stress_test_plots(self, results):
        """Create stress test analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        levels = results['levels']
        segment_counts = results['segment_counts']
        dimensions = results['dimensions']
        times = results['computation_times']
        r_squared = results['r_squared']
        
        # Plot 1: Segments vs Level (log scale)
        ax1.semilogy(levels, segment_counts, 'bo-', markersize=6)
        theoretical_segments = [2**level for level in levels]
        ax1.semilogy(levels, theoretical_segments, 'r--', label='Theoretical 2^n')
        ax1.set_xlabel('Dragon Curve Level')
        ax1.set_ylabel('Number of Segments (log scale)')
        ax1.set_title('Complexity Growth')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Computation Time vs Segments (log-log)
        ax2.loglog(segment_counts, times, 'go-', markersize=6)
        ax2.set_xlabel('Number of Segments (log scale)')
        ax2.set_ylabel('Computation Time (s, log scale)')
        ax2.set_title('Time Complexity')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Dimension vs Level
        ax3.errorbar(levels, dimensions, yerr=results['errors'], 
                    fmt='ro-', capsize=5, markersize=6)
        ax3.axhline(y=self.theoretical_dimension, color='blue', linestyle='--',
                   label=f'Theoretical D = {self.theoretical_dimension:.4f}')
        ax3.set_xlabel('Dragon Curve Level')
        ax3.set_ylabel('Measured Fractal Dimension')
        ax3.set_title('Algorithm Accuracy vs Complexity')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: R-squared vs Level
        ax4.plot(levels, r_squared, 'mo-', markersize=6)
        ax4.set_xlabel('Dragon Curve Level')
        ax4.set_ylabel('R-squared Value')
        ax4.set_title('Analysis Quality vs Complexity')
        ax4.set_ylim(0.9, 1.001)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.analyzer._save_plot('dragon_stress_test')
        plt.close()


def main():
    """
    Main function for Dragon curve robustness analysis.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Dragon Curve Algorithm Robustness Analysis')
    parser.add_argument('--parameter_sweep', action='store_true',
                       help='Run comprehensive parameter robustness test')
    parser.add_argument('--boundary_analysis', action='store_true',
                       help='Analyze boundary complexity effects')
    parser.add_argument('--stress_test', action='store_true',
                       help='Run algorithmic stress test')
    parser.add_argument('--max_level', type=int, default=7,
                       help='Maximum Dragon curve level for stress test (default: 7)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--eps_plots', action='store_true',
                       help='Generate EPS plots for publication')
    parser.add_argument('--no_titles', action='store_true',
                       help='Disable plot titles for journal submission')
    
    args = parser.parse_args()
    
    # Create robustness analyzer
    analyzer = DragonRobustnessAnalyzer(
        eps_plots=args.eps_plots,
        no_titles=args.no_titles
    )
    
    print("Dragon Curve Algorithm Robustness Analysis")
    print("=" * 60)
    print(f"Theoretical Dragon dimension: {analyzer.theoretical_dimension:.6f}")
    
    # Run requested analyses
    if args.parameter_sweep or (not any([args.boundary_analysis, args.stress_test])):
        print("\nRunning parameter robustness test...")
        robustness_results = analyzer.parameter_robustness_test(
            level=6, 
            create_plots=not args.no_plots
        )
    
    if args.boundary_analysis:
        print("\nRunning boundary complexity analysis...")
        boundary_results = analyzer.boundary_complexity_analysis(
            level=6, 
            create_plots=not args.no_plots
        )
    
    if args.stress_test:
        print("\nRunning algorithmic stress test...")
        stress_results = analyzer.algorithmic_stress_test(
            max_level=args.max_level, 
            create_plots=not args.no_plots
        )
    
    # Print comprehensive summary
    print(f"\n{'='*60}")
    print("ROBUSTNESS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    if 'robustness' in analyzer.results:
        robustness = analyzer.results['robustness']['robustness_summary']
        print("Parameter Robustness:")
        for param, metrics in robustness.items():
            print(f"  {param}: {metrics['assessment']} (CV: {metrics['cv_percent']:.2f}%)")
    
    if 'boundary' in analyzer.results:
        boundary = analyzer.results['boundary']
        print(f"Boundary Handling: {boundary['better_method']} method performs better")
    
    if 'stress_test' in analyzer.results:
        stress = analyzer.results['stress_test']
        print(f"Stress Test: Handled up to level {stress['max_level_tested']}")
        if stress['complexity_exponent']:
            print(f"Time Complexity: O(n^{stress['complexity_exponent']:.2f})")
    
    print(f"\n💡 Dragon curve provides excellent robustness testing due to:")
    print(f"   • Complex folding patterns that challenge boundary detection")
    print(f"   • Asymmetric self-similarity testing algorithm assumptions")  
    print(f"   • Intermediate dimension (~1.52) testing scaling range")
    print(f"   • Exponential complexity growth for performance testing")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
