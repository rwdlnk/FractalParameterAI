"""
Vectorized box counting implementation for FastFractalAnalyzer v3.

High-performance box counting using NumPy broadcasting and optimized algorithms.
"""

from typing import Tuple
import numpy as np
from numba import jit

from .data_types import SegmentArray, BoundingBox
from .spatial_index import SpatiallyIndexedBoxCounter
from .box_cache import BoxCountCache


class VectorizedBoxCounter:
    """
    High-performance box counting using vectorized operations and spatial indexing.

    Uses proper Liang-Barsky line clipping for accurate line-box intersections,
    based on the proven algorithm from the original FractalAnalyzer.

    Automatically switches between vectorized and spatially-indexed algorithms
    based on problem size for optimal performance.
    """

    def __init__(self, margin_factor: float = 0.01, use_spatial_index: bool = True,
                 enable_caching: bool = True):
        """
        Initialize box counter.

        Args:
            margin_factor: Margin factor for box size estimation
            use_spatial_index: Enable spatial indexing for large datasets
            enable_caching: Enable result caching for repeated computations
        """
        self.margin_factor = margin_factor
        self.use_spatial_index = use_spatial_index
        self.enable_caching = enable_caching
        self.spatial_counter = None

        # Initialize cache if enabled
        if enable_caching:
            self.cache = BoxCountCache(max_size=2000, ttl_seconds=7200)  # 2 hours TTL
        else:
            self.cache = None

    def count_boxes(
        self,
        segments: SegmentArray,
        box_size: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> int:
        """
        Count occupied boxes using optimal algorithm with caching.

        Automatically chooses between vectorized and spatially-indexed algorithms
        based on problem complexity, with intelligent result caching.

        Args:
            segments: Line segments to analyze
            box_size: Size of counting boxes
            offset_x, offset_y: Grid offset for optimization

        Returns:
            Number of occupied boxes
        """
        # Check cache first if enabled
        if self.cache is not None:
            cached_result = self.cache.get(segments, box_size, offset_x, offset_y)
            if cached_result is not None:
                return cached_result
        # Decision heuristic: use spatial indexing for complex cases
        complexity_threshold = 10000  # segments * estimated_boxes threshold

        bbox = segments.bbox
        domain_width = bbox.width + 2 * max(bbox.width, bbox.height) * self.margin_factor
        domain_height = bbox.height + 2 * max(bbox.width, bbox.height) * self.margin_factor
        estimated_boxes = (domain_width / box_size) * (domain_height / box_size)
        complexity = segments.n_segments * estimated_boxes

        # Compute result using optimal algorithm
        if self.use_spatial_index and complexity > complexity_threshold:
            result = self._count_boxes_spatial_index(segments, box_size, offset_x, offset_y)
        else:
            result = self._count_boxes_vectorized(segments, box_size, offset_x, offset_y)

        # Store result in cache if enabled
        if self.cache is not None:
            self.cache.put(segments, box_size, result, offset_x, offset_y)

        return result

    def _count_boxes_vectorized(
        self,
        segments: SegmentArray,
        box_size: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> int:
        """
        Count occupied boxes using vectorized operations (original method).

        Args:
            segments: Line segments to analyze
            box_size: Size of counting boxes
            offset_x, offset_y: Grid offset for optimization

        Returns:
            Number of occupied boxes
        """
        bbox = segments.bbox

        # Add margin to bounding box
        margin = max(bbox.width, bbox.height) * self.margin_factor
        domain = BoundingBox(
            min_x=bbox.min_x - margin + offset_x,
            min_y=bbox.min_y - margin + offset_y,
            max_x=bbox.max_x + margin + offset_x,
            max_y=bbox.max_y + margin + offset_y
        )

        # Calculate grid dimensions
        n_boxes_x = int(np.ceil(domain.width / box_size))
        n_boxes_y = int(np.ceil(domain.height / box_size))

        if n_boxes_x * n_boxes_y > 1_000_000:  # Safety check
            raise ValueError(f"Grid too large: {n_boxes_x}x{n_boxes_y} boxes")

        # Vectorized intersection test
        occupied = self._vectorized_intersection_test(
            segments, box_size, domain, n_boxes_x, n_boxes_y
        )

        return int(np.sum(occupied))

    def _count_boxes_spatial_index(
        self,
        segments: SegmentArray,
        box_size: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> int:
        """
        Count occupied boxes using spatial indexing for better performance on complex data.

        Args:
            segments: Line segments to analyze
            box_size: Size of counting boxes
            offset_x, offset_y: Grid offset for optimization

        Returns:
            Number of occupied boxes
        """
        # Initialize or reuse spatial counter
        if self.spatial_counter is None or self.spatial_counter.segments != segments:
            self.spatial_counter = SpatiallyIndexedBoxCounter(segments)

        return self.spatial_counter.count_boxes_spatially_indexed(box_size, offset_x, offset_y)

    def _vectorized_intersection_test(
        self,
        segments: SegmentArray,
        box_size: float,
        domain: BoundingBox,
        n_boxes_x: int,
        n_boxes_y: int
    ) -> np.ndarray:
        """
        Vectorized line-box intersection test using NumPy broadcasting.

        Returns:
            Boolean array of shape (n_boxes_y, n_boxes_x) indicating occupied boxes
        """
        # Create box grid coordinates
        box_i = np.arange(n_boxes_x)
        box_j = np.arange(n_boxes_y)
        grid_i, grid_j = np.meshgrid(box_i, box_j)

        # Box boundaries (vectorized)
        box_x_min = domain.min_x + grid_i * box_size
        box_y_min = domain.min_y + grid_j * box_size
        box_x_max = box_x_min + box_size
        box_y_max = box_y_min + box_size

        # Segment coordinates
        seg_data = segments.segments  # Shape: (n_segs, 2, 2)
        x1, y1 = seg_data[:, 0, 0], seg_data[:, 0, 1]  # Start points
        x2, y2 = seg_data[:, 1, 0], seg_data[:, 1, 1]  # End points

        # Vectorized intersection test using broadcasting
        # Shape: (n_segs, n_boxes_y, n_boxes_x)
        intersects = _vectorized_line_box_intersection(
            x1[:, None, None], y1[:, None, None],
            x2[:, None, None], y2[:, None, None],
            box_x_min[None, :, :], box_y_min[None, :, :],
            box_x_max[None, :, :], box_y_max[None, :, :]
        )

        # Any segment intersects box -> box is occupied
        return np.any(intersects, axis=0)

    def compute_box_spectrum(
        self,
        segments: SegmentArray,
        min_box_size: float,
        max_box_size: float,
        size_factor: float = 1.5,
        grid_optimization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full box counting spectrum.

        Args:
            segments: Line segments to analyze
            min_box_size: Smallest box size
            max_box_size: Largest box size
            size_factor: Factor to reduce box size each iteration
            grid_optimization: Use multiple grid offsets to find minimum

        Returns:
            Tuple of (box_sizes, box_counts) arrays
        """
        box_sizes = []
        box_counts = []

        current_size = max_box_size

        print("Box counting with grid optimization:")
        print("  Box size  |  Min count |  Grid tests  | Improvement | Time (s)")
        print("-----------------------------------------------------------")

        while current_size >= min_box_size:
            import time
            start_time = time.time()

            if grid_optimization:
                min_count, max_count, grid_tests = self._optimized_grid_count(
                    segments, current_size, min_box_size)

                improvement = (max_count - min_count) / max_count * 100 if max_count > 0 else 0
                elapsed = time.time() - start_time
                print(f"  {current_size:.6f}  |  {min_count:8d}  |  {grid_tests:10d}  |  {improvement:5.1f}%  |  {elapsed:.2f}")

                # Use min_count if non-zero, otherwise use max_count
                # Zero counts are often grid alignment artifacts, not real geometry
                count = min_count if min_count > 0 else max_count
            else:
                count = self.count_boxes(segments, current_size)
                elapsed = time.time() - start_time
                print(f"  {current_size:.6f}  |  {count:8d}  |  {elapsed:.2f}")

            if count > 0:
                box_sizes.append(current_size)
                box_counts.append(count)

            current_size /= size_factor

        if len(box_sizes) < 2:
            raise ValueError("Need at least 2 valid box sizes for analysis")

        return np.array(box_sizes), np.array(box_counts)

    def _optimized_grid_count(self, segments: SegmentArray, box_size: float, min_box_size: float) -> Tuple[int, int, int]:
        """
        Find min/max box count across multiple grid offsets with full statistics.

        Returns:
            Tuple of (min_count, max_count, grid_tests)
        """
        # Adaptive grid density based on box size (matching original logic)
        if box_size < min_box_size * 5:  # Very small boxes
            offset_increments = np.linspace(0, 0.75, 4)  # 4×4 = 16 tests
            grid_type = "FINE (4×4, 16 tests)"
        elif box_size < min_box_size * 20:  # Medium boxes
            offset_increments = np.linspace(0, 0.5, 3)   # 3×3 = 9 tests
            grid_type = "MEDIUM (3×3, 9 tests)"
        else:  # Large boxes
            offset_increments = np.linspace(0, 0.5, 2)   # 2×2 = 4 tests
            grid_type = "COARSE (2×2, 4 tests)"

        print(f"Box size {box_size:.6f}: Using {grid_type}")

        min_count = float('inf')
        max_count = 0
        grid_tests = 0

        bbox = segments.bbox

        for dx_fraction in offset_increments:
            for dy_fraction in offset_increments:
                grid_tests += 1

                offset_x = dx_fraction * box_size
                offset_y = dy_fraction * box_size

                count = self.count_boxes(segments, box_size, offset_x, offset_y)

                min_count = min(min_count, count)
                max_count = max(max_count, count)

        return int(min_count), int(max_count), grid_tests

    def get_performance_info(self, segments: SegmentArray) -> dict:
        """
        Get performance information including spatial indexing and caching statistics.

        Args:
            segments: Segments for analysis

        Returns:
            Dictionary with performance metrics
        """
        info = {
            'use_spatial_index': self.use_spatial_index,
            'enable_caching': self.enable_caching,
            'segments_count': segments.n_segments,
            'spatial_index_active': self.spatial_counter is not None,
            'cache_active': self.cache is not None
        }

        if self.spatial_counter is not None:
            spatial_stats = self.spatial_counter.get_performance_stats()
            info.update({f'spatial_{k}': v for k, v in spatial_stats.items()})

        if self.cache is not None:
            cache_stats = self.cache.get_stats()
            info.update({f'cache_{k}': v for k, v in cache_stats.items()})

        return info


def _vectorized_line_box_intersection(
    x1: np.ndarray, y1: np.ndarray,
    x2: np.ndarray, y2: np.ndarray,
    box_x_min: np.ndarray, box_y_min: np.ndarray,
    box_x_max: np.ndarray, box_y_max: np.ndarray
) -> np.ndarray:
    """
    Vectorized line-box intersection using proper Liang-Barsky algorithm.

    Based on the proven working algorithm from the original FractalAnalyzer.
    """
    # Vectorized Liang-Barsky line clipping
    dx = x2 - x1
    dy = y2 - y1

    # Liang-Barsky parameters
    p = np.stack([-dx, dx, -dy, dy], axis=-1)
    q = np.stack([
        x1 - box_x_min,
        box_x_max - x1,
        y1 - box_y_min,
        box_y_max - y1
    ], axis=-1)

    # Handle zero-length segments
    zero_dx = (np.abs(dx) < 1e-10)
    zero_dy = (np.abs(dy) < 1e-10)

    # Check vertical lines (dx = 0)
    outside_x = zero_dx & ((x1 < box_x_min) | (x1 > box_x_max))

    # Check horizontal lines (dy = 0)
    outside_y = zero_dy & ((y1 < box_y_min) | (y1 > box_y_max))

    # Point segments (both dx and dy = 0)
    point_segments = zero_dx & zero_dy
    point_inside = (
        (x1 >= box_x_min) & (x1 <= box_x_max) &
        (y1 >= box_y_min) & (y1 <= box_y_max)
    )

    # Initialize t_min and t_max
    t_min = np.zeros_like(x1)
    t_max = np.ones_like(x1)

    # Process each boundary
    for i in range(4):
        p_i = p[..., i]
        q_i = q[..., i]

        parallel = (np.abs(p_i) < 1e-10)
        outside = parallel & (q_i < 0)

        # For non-parallel cases, compute intersection parameter
        non_parallel = ~parallel
        # Suppress divide by zero warning - we handle it with np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(non_parallel & (np.abs(p_i) > 1e-10), q_i / p_i, 0)

        # Update t_min for entering boundaries (p_i < 0)
        entering = non_parallel & (p_i < 0)
        t_min = np.where(entering, np.maximum(t_min, t), t_min)

        # Update t_max for leaving boundaries (p_i > 0)
        leaving = non_parallel & (p_i > 0)
        t_max = np.where(leaving, np.minimum(t_max, t), t_max)

    # Final intersection test
    valid_intersection = (
        (t_min <= t_max) &
        (t_max >= 0.0) &
        (t_min <= 1.0) &
        ~outside_x &
        ~outside_y
    )

    # Handle point segments separately
    result = np.where(point_segments, point_inside, valid_intersection)

    return result


