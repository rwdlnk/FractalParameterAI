"""
Wu-Enhanced FastFractalAnalyzer
===============================

Hybrid analyzer that automatically detects interface complexity and applies
Wu et al. mathematical enhancements for straight/nearly-straight interfaces
while maintaining standard performance for complex fractals.

Key features:
- Automatic interface complexity detection
- Wu mathematical enhancements for straight interfaces (D ≈ 0.996)
- Standard optimizations for complex interfaces
- Transparent method switching
- Backward compatibility with existing API
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats

from .data_types import SegmentArray, FractalResult
from .fractal_analyzer import FastFractalAnalyzer
from .performance_config import PerformanceMode


class InterfaceComplexityAnalyzer:
    """Analyze interface complexity to determine optimal processing method."""

    @staticmethod
    def estimate_interface_complexity(segments: SegmentArray) -> float:
        """
        Estimate interface complexity score.

        Returns:
            complexity: Score where ~1.0 = straight line, >1.1 = complex fractal
        """
        if segments.n_segments == 0:
            return 1.0

        # Method 1: Segment length variation
        # Calculate segment lengths from start and end points
        start_points = segments.start_points
        end_points = segments.end_points
        diff = end_points - start_points
        segment_lengths = np.sqrt(np.sum(diff**2, axis=1))

        if len(segment_lengths) == 0:
            return 1.0

        length_cv = np.std(segment_lengths) / np.mean(segment_lengths) if np.mean(segment_lengths) > 0 else 0

        # Method 2: Angular variation between consecutive segments
        # For RT interfaces, we need to look at how segments connect, not just individual directions
        angles = []
        start_points = segments.start_points
        end_points = segments.end_points

        # Calculate angles between consecutive connected segments
        for i in range(segments.n_segments - 1):
            # Vector for current segment
            v1 = end_points[i] - start_points[i]

            # Vector for next segment - need to check if they're connected
            # For RT interfaces, segments should be roughly connected in sequence
            v2 = end_points[i + 1] - start_points[i + 1]

            # Calculate angle between segment directions
            if np.linalg.norm(v1) > 1e-10 and np.linalg.norm(v2) > 1e-10:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)

                # For nearly straight interfaces, angles should be close to 0 (parallel) or π (opposite)
                # Convert to deviation from straight line
                deviation = min(angle, np.pi - angle)
                angles.append(deviation)

        # Use mean deviation rather than std for better straight-line detection
        angle_variation = np.mean(angles) if angles else 0

        # Method 3: Bounding box aspect ratio
        bbox = segments.bbox
        if bbox.max_x > bbox.min_x or bbox.max_y > bbox.min_y:
            width = bbox.max_x - bbox.min_x
            height = bbox.max_y - bbox.min_y

            # Handle degenerate cases
            if width == 0 and height == 0:
                aspect_ratio = 1.0  # Point
            elif width == 0:
                aspect_ratio = float('inf')  # Vertical line
            elif height == 0:
                aspect_ratio = float('inf')  # Horizontal line
            else:
                aspect_ratio = max(width, height) / min(width, height)
        else:
            aspect_ratio = 1.0

        # Combine metrics into complexity score
        # For RT interfaces with straight segments: low length variation, low angular deviation, high aspect ratio
        # Weight angular deviation more heavily since it's the key indicator of straightness
        aspect_factor = 1.0 / max(aspect_ratio, 1.0) if aspect_ratio != float('inf') else 0.0

        # Scale angular variation (convert from radians to a 0-1 scale)
        # Small angular deviations (< 0.1 radians ≈ 5.7°) should contribute minimally
        scaled_angle_variation = angle_variation / 0.1  # Normalize by 0.1 radians

        complexity = 1.0 + length_cv + scaled_angle_variation + aspect_factor

        # Debug output for development
        # print(f"Debug: length_cv={length_cv:.3f}, angle_variation={angle_variation:.3f} (scaled={scaled_angle_variation:.3f}), aspect_ratio={aspect_ratio:.3f}, aspect_factor={aspect_factor:.3f}")

        return complexity

    @staticmethod
    def is_nearly_straight(segments: SegmentArray, threshold: float = 1.15) -> bool:
        """
        Determine if interface is nearly straight.

        Args:
            segments: Interface segments
            threshold: Complexity threshold for "straight" classification

        Returns:
            True if interface is nearly straight
        """
        complexity = InterfaceComplexityAnalyzer.estimate_interface_complexity(segments)
        return complexity < threshold


class WuEnhancedAnalyzer(FastFractalAnalyzer):
    """
    Enhanced FastFractalAnalyzer with automatic Wu method integration.

    Maintains full backward compatibility while adding Wu enhancements
    for straight/nearly-straight interfaces.
    """

    def __init__(
        self,
        min_r_squared: float = 0.99,
        min_scaling_decades: float = 1.5,
        grid_optimization: bool = True,
        boundary_artifact_removal: bool = False,
        performance_mode: PerformanceMode = PerformanceMode.BALANCED,
        use_roy_method: bool = False,
        wu_complexity_threshold: float = 1.15,
        wu_density_factor: float = 50,
        auto_wu_enhancement: bool = True
    ):
        """
        Initialize Wu-enhanced analyzer.

        Args:
            wu_complexity_threshold: Threshold for activating Wu enhancements
            wu_density_factor: Point density factor for Wu interpolation
            auto_wu_enhancement: Enable automatic Wu enhancement detection
            Other args: Same as FastFractalAnalyzer
        """
        super().__init__(
            min_r_squared=min_r_squared,
            min_scaling_decades=min_scaling_decades,
            grid_optimization=grid_optimization,
            boundary_artifact_removal=boundary_artifact_removal,
            performance_mode=performance_mode,
            use_roy_method=use_roy_method
        )

        self.wu_complexity_threshold = wu_complexity_threshold
        self.wu_density_factor = wu_density_factor
        self.auto_wu_enhancement = auto_wu_enhancement

        # Statistics
        self.wu_enhancements_used = 0
        self.standard_analyses = 0

    def _create_wu_dense_interface(self, segments: SegmentArray, density_factor: float) -> np.ndarray:
        """
        Create Wu-style dense point cloud from segments.

        Args:
            segments: Interface segments
            density_factor: Points per unit length for interpolation

        Returns:
            Dense array of (x, y) coordinates
        """
        start_points = segments.start_points
        end_points = segments.end_points

        all_points = []

        for i in range(segments.n_segments):
            x1, y1 = start_points[i]
            x2, y2 = end_points[i]

            # Calculate segment length
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Number of interpolation points based on length
            n_points = max(3, int(segment_length * density_factor))

            # Linear interpolation along segment
            t_values = np.linspace(0, 1, n_points, endpoint=False)
            x_interp = x1 + t_values * (x2 - x1)
            y_interp = y1 + t_values * (y2 - y1)

            # Add interpolated points
            for j in range(n_points):
                all_points.append([x_interp[j], y_interp[j]])

        # Add final endpoints
        if segments.n_segments > 0:
            all_points.append([end_points[-1, 0], end_points[-1, 1]])

        return np.array(all_points)

    def _wu_mathematical_box_counting(self, points: np.ndarray, domain_factor: float = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wu et al. mathematical box counting with precise intervals.

        Args:
            points: Dense interface points
            domain_factor: Factor for minimum epsilon calculation

        Returns:
            epsilon_values, N_values: Box sizes and counts
        """
        if len(points) == 0:
            return np.array([]), np.array([])

        # Get domain
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)

        domain_size = max(x_max - x_min, y_max - y_min)

        # Wu-style epsilon range
        min_epsilon = domain_size / domain_factor  # Finer than standard
        max_epsilon = domain_size / 2
        n_scales = 20

        epsilon_values = np.logspace(
            np.log10(min_epsilon),
            np.log10(max_epsilon),
            n_scales
        )[::-1]

        N_values = []

        for eps in epsilon_values:
            # Wu mathematical interval: m*ε ≤ x < (m+1)*ε
            occupied_boxes = set()

            for point in points:
                x, y = point

                # Mathematical interval calculation
                m = int(np.floor((x - x_min) / eps))
                n = int(np.floor((y - y_min) / eps))

                # Ensure valid indices
                max_m = int(np.ceil((x_max - x_min) / eps))
                max_n = int(np.ceil((y_max - y_min) / eps))

                m = max(0, min(m, max_m - 1))
                n = max(0, min(n, max_n - 1))

                occupied_boxes.add((m, n))

            N_values.append(len(occupied_boxes))

        return np.array(epsilon_values), np.array(N_values)

    def _wu_optimal_dimension_calculation(self, epsilon_values: np.ndarray, N_values: np.ndarray) -> Tuple[float, float]:
        """
        Wu-style optimal dimension calculation with range selection.

        Returns:
            dimension, r_squared: Fractal dimension and fit quality
        """
        # Remove invalid data
        valid_mask = (N_values > 0) & (epsilon_values > 0) & \
                     np.isfinite(N_values) & np.isfinite(epsilon_values)

        if np.sum(valid_mask) < 5:
            return np.nan, np.nan

        log_eps = np.log(epsilon_values[valid_mask])
        log_N = np.log(N_values[valid_mask])

        # Find optimal scaling range
        best_r2 = 0
        best_dimension = np.nan

        n_points = len(log_eps)
        min_points = max(5, int(0.6 * n_points))

        for start_idx in range(0, n_points - min_points + 1):
            for end_idx in range(start_idx + min_points, n_points + 1):
                log_eps_subset = log_eps[start_idx:end_idx]
                log_N_subset = log_N[start_idx:end_idx]

                if len(log_eps_subset) < 5:
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps_subset, log_N_subset)
                r_squared = r_value**2

                if r_squared > best_r2:
                    best_r2 = r_squared
                    best_dimension = -slope

        return best_dimension, best_r2

    def analyze_mathematical_fractal(self, segments: SegmentArray) -> Optional[FractalResult]:
        """
        Enhanced analysis with automatic Wu method selection.

        This method maintains the same API as the parent class but automatically
        applies Wu enhancements for straight interfaces.

        Args:
            segments: Interface segments to analyze

        Returns:
            FractalResult with enhanced accuracy for straight interfaces
        """
        if not self.auto_wu_enhancement:
            # Fall back to standard method
            return super().compute_fractal_dimension(segments)

        # Analyze interface complexity
        complexity = InterfaceComplexityAnalyzer.estimate_interface_complexity(segments)
        is_straight = complexity < self.wu_complexity_threshold

        if is_straight:
            # Apply Wu enhancements
            print(f"Detected nearly-straight interface (complexity={complexity:.3f}) - applying Wu enhancements")
            self.wu_enhancements_used += 1

            # Create dense interface representation
            dense_points = self._create_wu_dense_interface(segments, self.wu_density_factor)

            # Wu mathematical box counting
            epsilon_values, N_values = self._wu_mathematical_box_counting(dense_points)

            if len(epsilon_values) == 0:
                print("Wu method failed - falling back to standard")
                self.standard_analyses += 1
                return super().compute_fractal_dimension(segments)

            # Wu optimal dimension calculation
            dimension, r_squared = self._wu_optimal_dimension_calculation(epsilon_values, N_values)

            if np.isnan(dimension):
                print("Wu dimension calculation failed - falling back to standard")
                self.standard_analyses += 1
                return super().compute_fractal_dimension(segments)

            # Create result with Wu enhancement information
            result = FractalResult(
                dimension=dimension,
                r_squared=r_squared,
                box_sizes=epsilon_values,
                box_counts=N_values,
                slope=-dimension,  # Slope is negative dimension in box counting
                intercept=0.0,  # Default intercept
                scaling_range=(epsilon_values.min(), epsilon_values.max()),
                n_points=len(epsilon_values)
            )
            # Add Wu-specific attributes
            result.method_used = "Wu Enhanced"
            result.complexity_score = complexity
            result.n_segments = segments.n_segments

            print(f"Wu enhanced result: D = {dimension:.6f}, R² = {r_squared:.6f}")
            return result

        else:
            # Use standard method for complex interfaces
            print(f"Detected complex interface (complexity={complexity:.3f}) - using standard optimizations")
            self.standard_analyses += 1
            result = super().compute_fractal_dimension(segments)
            if result:
                result.method_used = "Standard Optimized"
                result.complexity_score = complexity
            return result

    def get_capabilities(self):
        """Get analyzer capabilities for compatibility with RT CLI."""
        return {
            'advanced_framework': True,
            'expected_improvement': 'Wu-enhanced analysis for straight interfaces',
            'wu_enhancements': True,
            'automatic_method_selection': True
        }

    def print_wu_statistics(self):
        """Print Wu enhancement usage statistics."""
        total = self.wu_enhancements_used + self.standard_analyses
        if total > 0:
            wu_percentage = (self.wu_enhancements_used / total) * 100
            print(f"\nWu Enhancement Statistics:")
            print(f"  Wu enhancements used: {self.wu_enhancements_used}")
            print(f"  Standard analyses: {self.standard_analyses}")
            print(f"  Wu usage rate: {wu_percentage:.1f}%")
        else:
            print("No analyses performed yet")

    @classmethod
    def create_rt_analyzer(cls, **kwargs):
        """Create analyzer optimized for RT interfaces with Wu enhancements."""
        return cls(
            wu_complexity_threshold=1.25,  # Relaxed for RT interfaces with straight segments
            wu_density_factor=50,          # Balanced density
            auto_wu_enhancement=True,
            performance_mode=PerformanceMode.BALANCED,
            **kwargs
        )

    @classmethod
    def create_strict_wu_analyzer(cls, **kwargs):
        """Create analyzer with strict Wu criteria for maximum accuracy."""
        return cls(
            wu_complexity_threshold=1.05,  # Very strict threshold
            wu_density_factor=100,         # High density
            auto_wu_enhancement=True,
            performance_mode=PerformanceMode.PRECISE,
            **kwargs
        )