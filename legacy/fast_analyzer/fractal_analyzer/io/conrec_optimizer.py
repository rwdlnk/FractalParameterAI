"""
Optimized CONREC algorithm with collinear segment merging.

Addresses the critical issue where early-time RT interfaces (nearly straight lines)
generate millions of tiny segments that hurt fractal analysis accuracy.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

from ..core.data_types import SegmentArray


@dataclass
class ContourSegment:
    """A single contour line segment."""
    x1: float
    y1: float
    x2: float
    y2: float

    def length(self) -> float:
        """Calculate segment length."""
        return math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def angle(self) -> float:
        """Calculate segment angle in radians."""
        return math.atan2(self.y2 - self.y1, self.x2 - self.x1)


class OptimizedCONREC:
    """
    High-performance CONREC implementation with collinear segment merging.

    Solves the critical problem where early-time RT interfaces generate
    excessive segments (>10^6) that degrade fractal analysis accuracy.
    """

    def __init__(self, collinear_tolerance: float = 1e-6, min_segment_length: float = 1e-8):
        """
        Initialize optimizer.

        Args:
            collinear_tolerance: Angular tolerance for collinear segments (radians)
            min_segment_length: Minimum segment length to keep
        """
        self.collinear_tolerance = collinear_tolerance
        self.min_segment_length = min_segment_length

    def extract_contour(self, z_grid: np.ndarray, x_coords: np.ndarray,
                       y_coords: np.ndarray, contour_level: float) -> SegmentArray:
        """
        Extract contour at specified level with optimized CONREC algorithm.

        Args:
            z_grid: 2D field values (e.g., volume fraction)
            x_coords: X coordinate array
            y_coords: Y coordinate array
            contour_level: Contour value to extract

        Returns:
            SegmentArray with merged collinear segments
        """
        # Get raw segments from CONREC
        raw_segments = self._conrec_extract(z_grid, x_coords, y_coords, contour_level)

        if not raw_segments:
            return SegmentArray.from_list([])

        # Merge collinear segments to solve over-segmentation
        merged_segments = self._merge_collinear_segments(raw_segments)

        # Convert to SegmentArray format
        segment_list = [((seg.x1, seg.y1), (seg.x2, seg.y2)) for seg in merged_segments]

        return SegmentArray.from_list(segment_list)

    def _conrec_extract(self, z_grid: np.ndarray, x_coords: np.ndarray,
                       y_coords: np.ndarray, contour_level: float) -> List[ContourSegment]:
        """
        Core CONREC algorithm - optimized for performance.

        Based on proven algorithm but with vectorized operations where possible.
        """
        segments = []
        nx, ny = z_grid.shape

        # Pre-allocate coordinate differences for efficiency
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # Process each cell
        for i in range(nx - 1):
            for j in range(ny - 1):
                # Get cell corner values
                z11 = z_grid[i, j]
                z12 = z_grid[i, j + 1]
                z21 = z_grid[i + 1, j]
                z22 = z_grid[i + 1, j + 1]

                # Skip if contour doesn't pass through this cell
                z_min = min(z11, z12, z21, z22)
                z_max = max(z11, z12, z21, z22)

                if contour_level < z_min or contour_level > z_max:
                    continue

                # Get cell coordinates
                x1, x2 = x_coords[i], x_coords[i + 1]
                y1, y2 = y_coords[j], y_coords[j + 1]

                # Find intersection points on cell edges
                intersections = self._find_cell_intersections(
                    z11, z12, z21, z22, contour_level, x1, x2, y1, y2
                )

                # Create segments from intersection pairs
                if len(intersections) >= 2:
                    # For multiple intersections, pair them appropriately
                    for k in range(0, len(intersections) - 1, 2):
                        if k + 1 < len(intersections):
                            x_start, y_start = intersections[k]
                            x_end, y_end = intersections[k + 1]

                            # Only add non-zero length segments
                            seg_length = math.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
                            if seg_length > self.min_segment_length:
                                segments.append(ContourSegment(x_start, y_start, x_end, y_end))

        return segments

    def _find_cell_intersections(self, z11: float, z12: float, z21: float, z22: float,
                               contour_level: float, x1: float, x2: float,
                               y1: float, y2: float) -> List[Tuple[float, float]]:
        """
        Find where contour intersects cell edges.

        Uses linear interpolation for sub-pixel accuracy.
        """
        intersections = []

        # Check each edge for intersections
        edges = [
            (z11, z21, x1, x2, y1, y1),  # Bottom edge
            (z21, z22, x2, x2, y1, y2),  # Right edge
            (z22, z12, x2, x1, y2, y2),  # Top edge
            (z12, z11, x1, x1, y2, y1),  # Left edge
        ]

        for z_start, z_end, x_start, x_end, y_start, y_end in edges:
            if self._crosses_contour(z_start, z_end, contour_level):
                # Linear interpolation to find exact intersection
                t = (contour_level - z_start) / (z_end - z_start)
                x_intersect = x_start + t * (x_end - x_start)
                y_intersect = y_start + t * (y_end - y_start)
                intersections.append((x_intersect, y_intersect))

        return intersections

    def _crosses_contour(self, z1: float, z2: float, level: float) -> bool:
        """Check if contour level crosses between two values."""
        return (z1 <= level <= z2) or (z2 <= level <= z1)

    def _merge_collinear_segments(self, segments: List[ContourSegment]) -> List[ContourSegment]:
        """
        Merge collinear segments to solve over-segmentation problem.

        This is the key optimization that reduces millions of tiny segments
        to a manageable number for fractal analysis.
        """
        if len(segments) <= 1:
            return segments

        # Build connectivity graph to find connected chains
        chains = self._build_segment_chains(segments)

        # Merge collinear segments within each chain
        merged_chains = []
        for chain in chains:
            merged_chain = self._merge_collinear_in_chain(chain)
            merged_chains.extend(merged_chain)

        return merged_chains

    def _build_segment_chains(self, segments: List[ContourSegment]) -> List[List[ContourSegment]]:
        """
        Build chains of connected segments.

        Segments are connected if their endpoints are close together.
        """
        if not segments:
            return []

        # Track which segments have been used
        used = [False] * len(segments)
        chains = []

        for i, start_seg in enumerate(segments):
            if used[i]:
                continue

            # Start a new chain
            chain = [start_seg]
            used[i] = True

            # Try to extend the chain in both directions
            self._extend_chain_forward(chain, segments, used)
            self._extend_chain_backward(chain, segments, used)

            chains.append(chain)

        return chains

    def _extend_chain_forward(self, chain: List[ContourSegment], segments: List[ContourSegment], used: List[bool]):
        """Extend chain forward by finding connected segments."""
        tolerance = max(self.min_segment_length * 10, 1e-6)  # More generous tolerance

        while True:
            last_seg = chain[-1]
            best_idx = -1
            best_distance = float('inf')

            # Find closest unused segment that connects to the end
            for i, seg in enumerate(segments):
                if used[i]:
                    continue

                # Check if this segment connects to the end of our chain
                distances = [
                    math.sqrt((last_seg.x2 - seg.x1)**2 + (last_seg.y2 - seg.y1)**2),  # End to start
                    math.sqrt((last_seg.x2 - seg.x2)**2 + (last_seg.y2 - seg.y2)**2),  # End to end
                ]

                min_dist = min(distances)
                if min_dist < tolerance and min_dist < best_distance:
                    best_distance = min_dist
                    best_idx = i

            if best_idx == -1:
                break  # No more connected segments

            # Add the best segment (flip if needed for proper orientation)
            seg = segments[best_idx]
            dist_start = math.sqrt((chain[-1].x2 - seg.x1)**2 + (chain[-1].y2 - seg.y1)**2)
            dist_end = math.sqrt((chain[-1].x2 - seg.x2)**2 + (chain[-1].y2 - seg.y2)**2)

            if dist_end < dist_start:
                # Flip segment for proper orientation
                seg = ContourSegment(seg.x2, seg.y2, seg.x1, seg.y1)

            chain.append(seg)
            used[best_idx] = True

    def _extend_chain_backward(self, chain: List[ContourSegment], segments: List[ContourSegment], used: List[bool]):
        """Extend chain backward by finding connected segments."""
        tolerance = max(self.min_segment_length * 10, 1e-6)  # More generous tolerance

        while True:
            first_seg = chain[0]
            best_idx = -1
            best_distance = float('inf')

            # Find closest unused segment that connects to the start
            for i, seg in enumerate(segments):
                if used[i]:
                    continue

                # Check if this segment connects to the start of our chain
                distances = [
                    math.sqrt((first_seg.x1 - seg.x2)**2 + (first_seg.y1 - seg.y2)**2),  # Start to end
                    math.sqrt((first_seg.x1 - seg.x1)**2 + (first_seg.y1 - seg.y1)**2),  # Start to start
                ]

                min_dist = min(distances)
                if min_dist < tolerance and min_dist < best_distance:
                    best_distance = min_dist
                    best_idx = i

            if best_idx == -1:
                break  # No more connected segments

            # Add the best segment at the beginning (flip if needed)
            seg = segments[best_idx]
            dist_end = math.sqrt((chain[0].x1 - seg.x2)**2 + (chain[0].y1 - seg.y2)**2)
            dist_start = math.sqrt((chain[0].x1 - seg.x1)**2 + (chain[0].y1 - seg.y1)**2)

            if dist_start < dist_end:
                # Flip segment for proper orientation
                seg = ContourSegment(seg.x2, seg.y2, seg.x1, seg.y1)

            chain.insert(0, seg)
            used[best_idx] = True

    def _merge_collinear_in_chain(self, chain: List[ContourSegment]) -> List[ContourSegment]:
        """
        Merge collinear segments within a connected chain.
        """
        if len(chain) <= 1:
            return chain

        merged = []
        current_group = [chain[0]]

        for i in range(1, len(chain)):
            # Check if current segment is collinear with the group
            if self._is_collinear_with_group(chain[i], current_group):
                current_group.append(chain[i])
            else:
                # Finalize current group and start new one
                merged_seg = self._merge_segment_group(current_group)
                if merged_seg:
                    merged.append(merged_seg)
                current_group = [chain[i]]

        # Don't forget the last group
        merged_seg = self._merge_segment_group(current_group)
        if merged_seg:
            merged.append(merged_seg)

        return merged

    def _is_collinear_with_group(self, segment: ContourSegment, group: List[ContourSegment]) -> bool:
        """Check if segment is collinear with the group direction."""
        if not group:
            return True

        # Calculate overall direction of the group
        first_seg = group[0]
        last_seg = group[-1]

        # Group direction from first start to last end
        group_angle = math.atan2(last_seg.y2 - first_seg.y1, last_seg.x2 - first_seg.x1)
        segment_angle = segment.angle()

        # Handle angle wraparound
        angle_diff = abs(group_angle - segment_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)

        return angle_diff < self.collinear_tolerance

    def _merge_segment_group(self, group: List[ContourSegment]) -> Optional[ContourSegment]:
        """Merge a group of collinear segments into one."""
        if not group:
            return None

        if len(group) == 1:
            return group[0]

        # Create merged segment from first start to last end
        first = group[0]
        last = group[-1]

        merged = ContourSegment(first.x1, first.y1, last.x2, last.y2)

        # Only return if it has meaningful length
        if merged.length() > self.min_segment_length:
            return merged

        return None



def optimize_conrec_for_vtk(volume_fraction: np.ndarray, x_grid: np.ndarray,
                          y_grid: np.ndarray, interface_value: float = 0.5,
                          collinear_tolerance: float = 1e-4) -> SegmentArray:
    """
    Extract interface using optimized CONREC with collinear merging.

    This function replaces the scikit-image marching squares approach
    with the proven CONREC algorithm optimized for RT simulation data.

    Args:
        volume_fraction: 2D volume fraction field
        x_grid, y_grid: Coordinate grids
        interface_value: Volume fraction for interface (default: 0.5)
        collinear_tolerance: Angular tolerance for merging (radians)

    Returns:
        SegmentArray with optimally merged segments
    """
    # Extract unique coordinates (assuming regular grid)
    x_coords = x_grid[:, 0]  # First column
    y_coords = y_grid[0, :]  # First row

    # Create optimizer with appropriate tolerance
    optimizer = OptimizedCONREC(
        collinear_tolerance=collinear_tolerance,
        min_segment_length=1e-8
    )

    # Extract and optimize contour
    return optimizer.extract_contour(volume_fraction, x_coords, y_coords, interface_value)