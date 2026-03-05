"""
FastFractalAnalyzer v3: Main analyzer class.

High-performance fractal dimension analysis with clean, simple API.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats

from .data_types import SegmentArray, FractalResult
from .box_counting import VectorizedBoxCounter
from .performance_config import PerformanceMode, PerformanceConfig


class FastFractalAnalyzer:
    """
    High-performance fractal dimension analyzer.

    Simplified API focused on core functionality with maximum performance.
    """

    # Quality thresholds (matching original)
    MIN_R_SQUARED_THRESHOLD = 0.99
    SLOPE_DEVIATION_THRESHOLD = 0.12

    def __init__(
        self,
        performance_mode: Optional[PerformanceMode] = None,
        min_r_squared: float = 0.99,
        min_scaling_decades: float = 1.5,
        grid_optimization: bool = True,
        boundary_artifact_removal: bool = True
    ):
        """
        Initialize analyzer.

        Args:
            performance_mode: Optional performance mode to configure defaults
            min_r_squared: Minimum R² for valid results
            min_scaling_decades: Minimum log10 range for scaling region
            grid_optimization: Use grid offset optimization
            boundary_artifact_removal: Use boundary artifact detection and removal
        """
        # Apply performance mode configuration if provided
        if performance_mode is not None:
            config = PerformanceConfig.from_mode(performance_mode)
            self.min_r_squared = config.min_r_squared
            self.min_scaling_decades = config.min_scaling_decades
            self.grid_optimization = config.grid_optimization
            self.boundary_artifact_removal = config.boundary_artifact_removal
            self.performance_config = config
        else:
            self.min_r_squared = min_r_squared
            self.min_scaling_decades = min_scaling_decades
            self.grid_optimization = grid_optimization
            self.boundary_artifact_removal = boundary_artifact_removal
            self.performance_config = None

        self.box_counter = VectorizedBoxCounter()

    def load_segments(self, filename: str) -> SegmentArray:
        """Load segments from file."""
        return SegmentArray.from_file(filename)

    def compute_fractal_dimension(
        self,
        segments: SegmentArray,
        min_box_size: Optional[float] = None,
        max_box_size: Optional[float] = None,
        size_factor: float = 1.5
    ) -> FractalResult:
        """
        Compute fractal dimension using box counting method.

        Args:
            segments: Line segments to analyze
            min_box_size: Minimum box size (auto-estimated if None)
            max_box_size: Maximum box size (auto-estimated if None)
            size_factor: Factor to reduce box size each iteration

        Returns:
            FractalResult with dimension and quality metrics
        """
        # Auto-estimate box size range if not provided
        if min_box_size is None or max_box_size is None:
            auto_min, auto_max = self._estimate_box_size_range(segments)
            min_box_size = min_box_size or auto_min
            max_box_size = max_box_size or auto_max

        # Compute box counting spectrum
        box_sizes, box_counts = self.box_counter.compute_box_spectrum(
            segments=segments,
            min_box_size=min_box_size,
            max_box_size=max_box_size,
            size_factor=size_factor,
            grid_optimization=self.grid_optimization
        )

        # Apply boundary artifact removal if enabled
        if self.boundary_artifact_removal:
            box_sizes, box_counts = self._enhanced_boundary_removal(box_sizes, box_counts)

        # Find optimal scaling region
        best_result = self._find_optimal_scaling_region(box_sizes, box_counts)

        return best_result

    def analyze_segments(
        self,
        segments: SegmentArray,
        initial_delta: Optional[float] = None,
        delta_factor: float = 1.5,
        num_steps: int = 25,
        min_box_size: Optional[float] = None
    ) -> FractalResult:
        """
        Analyze segments using box counting method.

        This is a convenience wrapper around compute_fractal_dimension that
        provides a simple interface expected by many test scripts.

        Args:
            segments: Line segments to analyze
            initial_delta: Initial box size (auto-estimated if None)
            delta_factor: Factor to reduce box size each iteration
            num_steps: Number of steps in the analysis
            min_box_size: Minimum box size cutoff (stops early if reached, overrides num_steps)

        Returns:
            FractalResult with dimension and quality metrics
        """
        if initial_delta is not None:
            # Convert parameters to min/max box size range
            max_box_size = initial_delta

            # Calculate min from num_steps, but enforce hard cutoff if provided
            calculated_min = initial_delta / (delta_factor ** (num_steps - 1))

            if min_box_size is not None:
                # Use the larger of calculated_min and provided min_box_size
                # This ensures we don't go below the specified cutoff
                final_min = max(calculated_min, min_box_size)
            else:
                final_min = calculated_min

            return self.compute_fractal_dimension(
                segments=segments,
                min_box_size=final_min,
                max_box_size=max_box_size,
                size_factor=delta_factor
            )
        else:
            # Use auto-estimation, but respect min_box_size if provided
            return self.compute_fractal_dimension(
                segments=segments,
                min_box_size=min_box_size,
                size_factor=delta_factor
            )

    def _estimate_box_size_range(self, segments: SegmentArray) -> Tuple[float, float]:
        """
        Automatically estimate reasonable box size range.

        Based on domain size and segment density.
        """
        bbox = segments.bbox
        domain_size = max(bbox.width, bbox.height)

        # Statistical approach: use segment length distribution
        diff = segments.end_points - segments.start_points
        segment_lengths = np.sqrt(np.sum(diff**2, axis=1))

        # Min box size: small fraction of median segment length
        min_size = np.percentile(segment_lengths, 10) * 0.1

        # Max box size: fraction of domain size to ensure good coverage
        max_size = domain_size * 0.3

        # Ensure reasonable range
        min_size = max(min_size, domain_size / 1000)  # Not too small
        max_size = min(max_size, domain_size / 3)     # Not too large

        # Ensure sufficient scaling range
        while np.log10(max_size / min_size) < self.min_scaling_decades:
            min_size *= 0.7
            max_size *= 1.2

        return min_size, max_size

    def _find_optimal_scaling_region(
        self,
        box_sizes: np.ndarray,
        box_counts: np.ndarray
    ) -> FractalResult:
        """
        Find the optimal linear scaling region using sliding window analysis.

        Returns the best fit region that meets quality criteria.
        """
        log_sizes = np.log10(box_sizes)
        log_counts = np.log10(box_counts)

        best_result = None
        best_score = -1

        # Try different window sizes
        min_points = 5
        max_points = min(len(log_sizes), 15)

        for n_points in range(min_points, max_points + 1):
            for start_idx in range(len(log_sizes) - n_points + 1):
                end_idx = start_idx + n_points

                # Extract window data
                x = log_sizes[start_idx:end_idx]
                y = log_counts[start_idx:end_idx]

                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared = r_value**2

                # Check quality criteria
                scaling_range = x[-1] - x[0]
                if (r_squared >= self.min_r_squared and
                    scaling_range >= self.min_scaling_decades):

                    # Score combines R² and scaling range
                    score = r_squared * scaling_range

                    if score > best_score:
                        best_score = score
                        best_result = FractalResult(
                            dimension=-slope,  # Negative slope = positive dimension
                            r_squared=r_squared,
                            box_sizes=box_sizes[start_idx:end_idx],
                            box_counts=box_counts[start_idx:end_idx],
                            slope=slope,
                            intercept=intercept,
                            scaling_range=(10**x[0], 10**x[-1]),
                            n_points=n_points
                        )

        if best_result is None:
            # Fallback: use all data even if quality is poor
            slope, intercept, r_value, _, _ = stats.linregress(log_sizes, log_counts)
            best_result = FractalResult(
                dimension=-slope,
                r_squared=r_value**2,
                box_sizes=box_sizes,
                box_counts=box_counts,
                slope=slope,
                intercept=intercept,
                scaling_range=(box_sizes[-1], box_sizes[0]),
                n_points=len(box_sizes)
            )

        return best_result

    def validate_result(self, result: FractalResult) -> bool:
        """Check if result meets quality standards."""
        return result.is_valid

    def analyze_mathematical_fractal(
        self,
        fractal_type: str,
        level: int = 5,
        **kwargs
    ) -> FractalResult:
        """
        Generate and analyze mathematical fractals.

        Args:
            fractal_type: 'koch', 'sierpinski', etc.
            level: Iteration level
            **kwargs: Additional parameters for fractal generation

        Returns:
            FractalResult for the generated fractal
        """
        segments = self._generate_fractal(fractal_type, level, **kwargs)
        return self.compute_fractal_dimension(segments)

    def _generate_fractal(
        self,
        fractal_type: str,
        level: int,
        **kwargs
    ) -> SegmentArray:
        """Generate mathematical fractal line segments."""
        if fractal_type.lower() == 'koch':
            return self._generate_koch_curve(level)
        elif fractal_type.lower() == 'sierpinski':
            return self._generate_sierpinski_triangle(level)
        elif fractal_type.lower() == 'dragon':
            return self._generate_dragon_curve(level)
        elif fractal_type.lower() == 'minkowski':
            return self._generate_minkowski_curve(level)
        elif fractal_type.lower() == 'hilbert':
            return self._generate_hilbert_curve(level)
        else:
            raise ValueError(f"Unsupported fractal type: {fractal_type}. "
                           f"Supported types: koch, sierpinski, dragon, minkowski, hilbert")

    def _generate_koch_curve(self, level: int) -> SegmentArray:
        """Generate Koch curve segments using the proven original algorithm."""
        import math
        from numba import jit

        @jit(nopython=True)
        def koch_points_jit(x1, y1, x2, y2, level, points_array, idx):
            """JIT-compiled Koch curve generator (from original working code)."""
            if level == 0:
                points_array[idx] = (x1, y1)
                return idx + 1
            else:
                angle = math.pi / 3
                x3 = x1 + (x2 - x1) / 3
                y3 = y1 + (y2 - y1) / 3
                x4 = (x1 + x2) / 2 + (y2 - y1) * math.sin(angle) / 3
                y4 = (y1 + y2) / 2 - (x2 - x1) * math.sin(angle) / 3
                x5 = x1 + 2 * (x2 - x1) / 3
                y5 = y1 + 2 * (y2 - y1) / 3

                idx = koch_points_jit(x1, y1, x3, y3, level - 1, points_array, idx)
                idx = koch_points_jit(x3, y3, x4, y4, level - 1, points_array, idx)
                idx = koch_points_jit(x4, y4, x5, y5, level - 1, points_array, idx)
                idx = koch_points_jit(x5, y5, x2, y2, level - 1, points_array, idx)

                return idx

        num_points = 4**level + 1
        points_array = np.zeros((num_points, 2), dtype=np.float64)

        final_idx = koch_points_jit(0, 0, 1, 0, level, points_array, 0)
        points_array[final_idx] = (1, 0)
        points = points_array[:final_idx+1]

        # Convert points to segments
        segments = []
        for i in range(len(points) - 1):
            segments.append(((points[i][0], points[i][1]),
                            (points[i+1][0], points[i+1][1])))

        return SegmentArray.from_list(segments)

    def _generate_sierpinski_triangle(self, level: int) -> SegmentArray:
        """Generate Sierpinski triangle outline segments using proper triangular subdivision."""
        import math

        def triangle_midpoints(p1, p2, p3):
            """Get midpoints of triangle edges."""
            m12 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            m23 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
            m31 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
            return m12, m23, m31

        def triangle_segments(p1, p2, p3):
            """Get segments forming a triangle."""
            return [(p1, p2), (p2, p3), (p3, p1)]

        # Start with equilateral triangle
        height = math.sqrt(3) / 2
        triangles = [((0.0, 0.0), (1.0, 0.0), (0.5, height))]

        # Sierpinski subdivision: each triangle becomes 3 smaller triangles
        for _ in range(level):
            new_triangles = []
            for p1, p2, p3 in triangles:
                # Get midpoints
                m12, m23, m31 = triangle_midpoints(p1, p2, p3)

                # Create 3 new triangles (corners), leaving middle empty
                new_triangles.extend([
                    (p1, m12, m31),     # Bottom-left triangle
                    (m12, p2, m23),     # Bottom-right triangle
                    (m31, m23, p3)      # Top triangle
                ])
            triangles = new_triangles

        # Convert triangles to line segments
        all_segments = []
        for p1, p2, p3 in triangles:
            all_segments.extend(triangle_segments(p1, p2, p3))

        # Remove duplicate segments (where triangles touch)
        unique_segments = []
        segment_set = set()

        for seg in all_segments:
            # Normalize segment direction for duplicate detection
            normalized = tuple(sorted([seg[0], seg[1]]))
            if normalized not in segment_set:
                segment_set.add(normalized)
                unique_segments.append(seg)

        return SegmentArray.from_list(unique_segments)

    def _generate_sierpinski_curve(self, level: int) -> SegmentArray:
        """
        Backward compatibility wrapper for Sierpinski generation.

        Tests expect this method name, but the implementation is actually
        _generate_sierpinski_triangle which generates the triangle outline.
        """
        return self._generate_sierpinski_triangle(level)

    def _generate_dragon_curve(self, level: int) -> SegmentArray:
        """Generate Dragon curve segments using L-system approach."""
        import math

        def dragon_sequence(n):
            """Generate dragon curve turn sequence."""
            if n == 0:
                return [1]  # Right turn

            prev = dragon_sequence(n - 1)
            # Dragon curve: add 1, then mirror and negate previous sequence
            # Manual reversal to avoid Numba compatibility issues
            reversed_prev = []
            for i in range(len(prev) - 1, -1, -1):
                reversed_prev.append(-prev[i])

            result = prev + [1] + reversed_prev
            return result

        # Generate turn sequence
        turns = dragon_sequence(level)

        # Convert turns to line segments
        x, y = 0.0, 0.0
        direction = 0  # 0=right, 1=up, 2=left, 3=down
        step_size = 1.0 / (2**level * 0.7)  # Scale appropriately

        segments = []
        points = [(x, y)]

        # Direction vectors: right, up, left, down
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]

        for turn in turns:
            # Move forward one step
            x += dx[direction] * step_size
            y += dy[direction] * step_size
            points.append((x, y))

            # Turn (1=right, -1=left)
            direction = (direction + turn) % 4

        # Convert points to segments
        for i in range(len(points) - 1):
            segments.append((points[i], points[i + 1]))

        return SegmentArray.from_list(segments)

    def _generate_minkowski_curve(self, level: int) -> SegmentArray:
        """Generate Minkowski sausage curve segments using proper replacement rule."""
        import math

        def minkowski_replace(segment):
            """Replace a segment with proper Minkowski sausage pattern."""
            (x1, y1), (x2, y2) = segment

            # Segment vector and length
            dx = x2 - x1
            dy = y2 - y1

            # Perpendicular vector (90-degree rotation)
            px = -dy
            py = dx

            # Scale factors for the pattern
            length_scale = 1.0 / 8.0  # Divide segment into 8 parts
            bulge_scale = 1.0 / 4.0   # Height of perpendicular bulges

            # Generate the classic Minkowski sausage pattern
            # This creates a connected path with 4 rectangular bulges
            points = [
                (x1, y1),  # Start point

                # First bulge (quarter way)
                (x1 + dx * length_scale, y1 + dy * length_scale),
                (x1 + dx * length_scale + px * bulge_scale, y1 + dy * length_scale + py * bulge_scale),
                (x1 + dx * 2 * length_scale + px * bulge_scale, y1 + dy * 2 * length_scale + py * bulge_scale),
                (x1 + dx * 2 * length_scale, y1 + dy * 2 * length_scale),

                # Second bulge (halfway)
                (x1 + dx * 3 * length_scale, y1 + dy * 3 * length_scale),
                (x1 + dx * 3 * length_scale - px * bulge_scale, y1 + dy * 3 * length_scale - py * bulge_scale),
                (x1 + dx * 4 * length_scale - px * bulge_scale, y1 + dy * 4 * length_scale - py * bulge_scale),
                (x1 + dx * 4 * length_scale, y1 + dy * 4 * length_scale),

                # Third bulge (three-quarters)
                (x1 + dx * 5 * length_scale, y1 + dy * 5 * length_scale),
                (x1 + dx * 5 * length_scale + px * bulge_scale, y1 + dy * 5 * length_scale + py * bulge_scale),
                (x1 + dx * 6 * length_scale + px * bulge_scale, y1 + dy * 6 * length_scale + py * bulge_scale),
                (x1 + dx * 6 * length_scale, y1 + dy * 6 * length_scale),

                # Final section to end
                (x1 + dx * 7 * length_scale, y1 + dy * 7 * length_scale),
                (x2, y2)  # End point
            ]

            # Convert points to connected segments
            segments = []
            for i in range(len(points) - 1):
                segments.append((points[i], points[i + 1]))

            return segments

        # Start with unit horizontal segment
        segments = [((0.0, 0.0), (1.0, 0.0))]

        # Apply Minkowski replacement iteratively
        for iteration in range(level):
            new_segments = []
            for segment in segments:
                new_segments.extend(minkowski_replace(segment))
            segments = new_segments

        return SegmentArray.from_list(segments)

    def _generate_hilbert_curve(self, level: int) -> SegmentArray:
        """Generate Hilbert curve segments using L-system."""
        import math

        def hilbert_points(level):
            """Generate Hilbert curve points recursively."""
            if level == 0:
                return [(0, 0)]

            # Recursive Hilbert curve generation
            points = []
            size = 2**(level - 1)

            # Four quadrants of Hilbert curve
            # Bottom-left (rotated)
            for x, y in hilbert_points(level - 1):
                points.append((y, x))

            # Top-left
            for x, y in hilbert_points(level - 1):
                points.append((x, y + size))

            # Top-right
            for x, y in hilbert_points(level - 1):
                points.append((x + size, y + size))

            # Bottom-right (rotated)
            for x, y in hilbert_points(level - 1):
                points.append((size * 2 - 1 - y, size - 1 - x))

            return points

        if level <= 0:
            return SegmentArray.from_list([((0.0, 0.0), (1.0, 0.0))])

        # Generate points and normalize to [0,1] range
        points = hilbert_points(level)
        if not points:
            return SegmentArray.from_list([((0.0, 0.0), (1.0, 0.0))])

        # Normalize coordinates
        max_coord = 2**level - 1
        normalized_points = [(x / max_coord, y / max_coord) for x, y in points]

        # Convert to segments
        segments = []
        for i in range(len(normalized_points) - 1):
            segments.append((normalized_points[i], normalized_points[i + 1]))

        return SegmentArray.from_list(segments)

    def benchmark_performance(
        self,
        segments: SegmentArray,
        n_runs: int = 3
    ) -> dict:
        """
        Benchmark analysis performance.

        Returns:
            Dictionary with timing statistics
        """
        import time

        times = []
        for _ in range(n_runs):
            start_time = time.perf_counter()
            result = self.compute_fractal_dimension(segments)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'n_segments': segments.n_segments,
            'dimension': result.dimension if 'result' in locals() else None
        }

    def _enhanced_boundary_removal(
        self,
        box_sizes: np.ndarray,
        box_counts: np.ndarray,
        trim_boundary: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced boundary artifact detection and removal.

        Matches the original algorithm from FractalAnalyzer v2.
        """
        original_length = len(box_sizes)

        # Apply manual trimming first if specified
        if trim_boundary > 0:
            if trim_boundary * 2 < len(box_sizes):
                box_sizes = box_sizes[trim_boundary:-trim_boundary]
                box_counts = box_counts[trim_boundary:-trim_boundary]
                print(f"Applied manual boundary trimming: {trim_boundary} points from each end")

        # Enhanced automatic boundary artifact detection
        if len(box_sizes) > 8:  # Need enough points for meaningful analysis
            log_sizes = np.log(box_sizes)
            log_counts = np.log(box_counts)

            # Check for deviations from linearity at ends
            n = len(log_sizes)
            segment_size = max(3, n // 4)  # Use quarter segments, minimum 3 points

            if n >= 3 * segment_size:  # Ensure we have enough points
                try:
                    # Calculate slopes for first, middle, and last segments
                    slope_first, _, r2_first, _, _ = stats.linregress(
                        log_sizes[:segment_size], log_counts[:segment_size])
                    slope_middle, _, r2_middle, _, _ = stats.linregress(
                        log_sizes[segment_size:2*segment_size], log_counts[segment_size:2*segment_size])
                    slope_last, _, r2_last, _, _ = stats.linregress(
                        log_sizes[-segment_size:], log_counts[-segment_size:])

                    # Boundary detection thresholds
                    additional_trim = 0

                    # Check first segment - both slope deviation AND R² quality
                    first_slope_dev = abs(slope_first - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (first_slope_dev > self.SLOPE_DEVIATION_THRESHOLD) or (r2_first < self.MIN_R_SQUARED_THRESHOLD):
                        additional_trim = max(additional_trim, 1)
                        print(f"Detected boundary artifact at start: slope deviation {first_slope_dev:.3f}, R² {r2_first:.3f}")

                    # Check last segment - both slope deviation AND R² quality
                    last_slope_dev = abs(slope_last - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
                    if (last_slope_dev > self.SLOPE_DEVIATION_THRESHOLD) or (r2_last < self.MIN_R_SQUARED_THRESHOLD):
                        additional_trim = max(additional_trim, 1)
                        print(f"Detected boundary artifact at end: slope deviation {last_slope_dev:.3f}, R² {r2_last:.3f}")

                    # Apply additional trimming if artifacts detected
                    if additional_trim > 0 and len(box_sizes) > 2 * additional_trim + 4:  # Keep at least 4 points
                        print(f"Removing {additional_trim} additional boundary points from each end")
                        box_sizes = box_sizes[additional_trim:-additional_trim]
                        box_counts = box_counts[additional_trim:-additional_trim]

                        # Verify the trimming improved the linearity
                        new_log_sizes = np.log(box_sizes)
                        new_log_counts = np.log(box_counts)
                        _, _, new_r2, _, _ = stats.linregress(new_log_sizes, new_log_counts)
                        print(f"R² after boundary trimming: {new_r2:.4f} (was {r2_middle:.4f})")

                except Exception as e:
                    print(f"Warning: Could not perform enhanced boundary detection: {e}")

        removed_points = original_length - len(box_sizes)
        if removed_points > 0:
            print(f"Total boundary points removed: {removed_points}")

        return box_sizes, box_counts