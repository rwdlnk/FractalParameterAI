"""
Spatial indexing for fast segment-box intersection queries.

Implements a simple but effective spatial index using uniform grid partitioning
optimized for fractal analysis box counting operations.
"""

import numpy as np
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import math

from .data_types import SegmentArray, BoundingBox


@dataclass
class GridCell:
    """A single cell in the spatial grid."""
    segment_indices: Set[int]

    def __post_init__(self):
        if not isinstance(self.segment_indices, set):
            self.segment_indices = set(self.segment_indices)


class SpatialIndex:
    """
    Uniform grid spatial index for fast segment queries.

    Partitions space into uniform grid cells and indexes segments by
    which cells they intersect. Provides fast lookup of segments
    that might intersect with query boxes.
    """

    def __init__(self, segments: SegmentArray, grid_resolution: Optional[int] = None):
        """
        Initialize spatial index.

        Args:
            segments: Line segments to index
            grid_resolution: Number of grid cells per dimension (auto if None)
        """
        self.segments = segments
        self.bbox = segments.bbox

        if segments.n_segments == 0:
            self.grid_resolution = 1
            self.cells = {}
            return

        # Auto-determine optimal grid resolution
        if grid_resolution is None:
            # Rule of thumb: aim for ~10-20 segments per cell on average
            target_segments_per_cell = 15
            total_area = (self.bbox.max_x - self.bbox.min_x) * (self.bbox.max_y - self.bbox.min_y)
            if total_area > 0 and segments.n_segments > 0:
                # Calculate optimal cell size based on segment density
                cells_needed = max(4, segments.n_segments / target_segments_per_cell)
                self.grid_resolution = max(4, min(50, int(math.sqrt(cells_needed))))
            else:
                self.grid_resolution = 4
        else:
            self.grid_resolution = grid_resolution

        # Calculate cell dimensions
        self.cell_width = (self.bbox.max_x - self.bbox.min_x) / self.grid_resolution
        self.cell_height = (self.bbox.max_y - self.bbox.min_y) / self.grid_resolution

        # Avoid division by zero for degenerate cases
        if self.cell_width == 0:
            self.cell_width = 1.0
        if self.cell_height == 0:
            self.cell_height = 1.0

        # Build the spatial index
        self.cells = {}
        self._build_index()

    def _build_index(self):
        """Build the spatial index by inserting all segments."""
        # Get segment coordinates
        start_points = self.segments.start_points
        end_points = self.segments.end_points

        for i in range(self.segments.n_segments):
            # Get segment bounding box
            x1, y1 = start_points[i]
            x2, y2 = end_points[i]

            seg_min_x = min(x1, x2)
            seg_max_x = max(x1, x2)
            seg_min_y = min(y1, y2)
            seg_max_y = max(y1, y2)

            # Find grid cells that this segment intersects
            cells_to_update = self._get_cells_for_bbox(seg_min_x, seg_min_y, seg_max_x, seg_max_y)

            # Add segment to all intersecting cells
            for cell_key in cells_to_update:
                if cell_key not in self.cells:
                    self.cells[cell_key] = GridCell(set())
                self.cells[cell_key].segment_indices.add(i)

    def _get_cells_for_bbox(self, min_x: float, min_y: float,
                          max_x: float, max_y: float) -> List[Tuple[int, int]]:
        """Get all grid cells that intersect with given bounding box."""
        # Convert coordinates to cell indices
        min_cell_x = max(0, int((min_x - self.bbox.min_x) / self.cell_width))
        max_cell_x = min(self.grid_resolution - 1, int((max_x - self.bbox.min_x) / self.cell_width))
        min_cell_y = max(0, int((min_y - self.bbox.min_y) / self.cell_height))
        max_cell_y = min(self.grid_resolution - 1, int((max_y - self.bbox.min_y) / self.cell_height))

        cells = []
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                cells.append((cell_x, cell_y))

        return cells

    def query_box(self, box_min_x: float, box_min_y: float,
                  box_max_x: float, box_max_y: float) -> Set[int]:
        """
        Query spatial index for segments that might intersect with box.

        Args:
            box_min_x, box_min_y: Box minimum coordinates
            box_max_x, box_max_y: Box maximum coordinates

        Returns:
            Set of segment indices that might intersect the box
        """
        if not self.cells:
            return set()

        # Find cells that intersect with query box
        intersecting_cells = self._get_cells_for_bbox(box_min_x, box_min_y, box_max_x, box_max_y)

        # Collect all segment indices from intersecting cells
        candidate_segments = set()
        for cell_key in intersecting_cells:
            if cell_key in self.cells:
                candidate_segments.update(self.cells[cell_key].segment_indices)

        return candidate_segments

    def query_boxes_batch(self, box_x_min: np.ndarray, box_y_min: np.ndarray,
                         box_x_max: np.ndarray, box_y_max: np.ndarray) -> List[Set[int]]:
        """
        Batch query for multiple boxes.

        Args:
            box_x_min, box_y_min: Arrays of box minimum coordinates
            box_x_max, box_y_max: Arrays of box maximum coordinates

        Returns:
            List of sets, each containing segment indices for corresponding box
        """
        results = []

        # Flatten box arrays for iteration
        flat_x_min = box_x_min.flatten()
        flat_y_min = box_y_min.flatten()
        flat_x_max = box_x_max.flatten()
        flat_y_max = box_y_max.flatten()

        for i in range(len(flat_x_min)):
            candidates = self.query_box(flat_x_min[i], flat_y_min[i],
                                      flat_x_max[i], flat_y_max[i])
            results.append(candidates)

        return results

    def get_stats(self) -> dict:
        """Get spatial index statistics for debugging/optimization."""
        if not self.cells:
            return {
                'grid_resolution': self.grid_resolution,
                'total_cells': 0,
                'occupied_cells': 0,
                'avg_segments_per_cell': 0,
                'max_segments_per_cell': 0
            }

        occupied_cells = len(self.cells)
        total_segments_indexed = sum(len(cell.segment_indices) for cell in self.cells.values())
        avg_segments_per_cell = total_segments_indexed / occupied_cells if occupied_cells > 0 else 0
        max_segments_per_cell = max(len(cell.segment_indices) for cell in self.cells.values()) if self.cells else 0

        return {
            'grid_resolution': self.grid_resolution,
            'cell_size': (self.cell_width, self.cell_height),
            'total_cells': self.grid_resolution * self.grid_resolution,
            'occupied_cells': occupied_cells,
            'occupancy_ratio': occupied_cells / (self.grid_resolution * self.grid_resolution),
            'total_segments': self.segments.n_segments,
            'total_indexed_entries': total_segments_indexed,
            'avg_segments_per_cell': avg_segments_per_cell,
            'max_segments_per_cell': max_segments_per_cell
        }


class SpatiallyIndexedBoxCounter:
    """
    Box counter that uses spatial indexing for fast segment-box intersection queries.

    Replaces the O(n*m) brute force approach with O(k*log(n)) where k is the
    average number of segments per box (much smaller than n for typical fractals).
    """

    def __init__(self, segments: SegmentArray):
        """Initialize with segments and build spatial index."""
        self.segments = segments
        self.spatial_index = SpatialIndex(segments)

        # Cache segment coordinates for fast access
        start_points = segments.start_points
        end_points = segments.end_points
        self.x1 = start_points[:, 0]
        self.y1 = start_points[:, 1]
        self.x2 = end_points[:, 0]
        self.y2 = end_points[:, 1]

    def count_boxes_spatially_indexed(self, box_size: float,
                                    offset_x: float = 0, offset_y: float = 0) -> int:
        """
        Count boxes using spatial index for candidate pruning.

        This provides the same result as brute force but much faster
        for complex interfaces with many segments.
        """
        if self.segments.n_segments == 0:
            return 0

        bbox = self.segments.bbox

        # Calculate grid parameters
        x_min = bbox.min_x - offset_x
        y_min = bbox.min_y - offset_y

        # Extend slightly to ensure coverage
        x_max = bbox.max_x + box_size
        y_max = bbox.max_y + box_size

        # Grid dimensions
        nx = int(np.ceil((x_max - x_min) / box_size))
        ny = int(np.ceil((y_max - y_min) / box_size))

        occupied_count = 0

        # Process all boxes directly (simpler and often faster for moderate sizes)
        for i in range(nx):
            for j in range(ny):
                box_x_min = x_min + i * box_size
                box_y_min = y_min + j * box_size
                box_x_max = box_x_min + box_size
                box_y_max = box_y_min + box_size

                # Query spatial index for candidate segments
                candidates = self.spatial_index.query_box(
                    box_x_min, box_y_min, box_x_max, box_y_max
                )

                if not candidates:
                    continue

                # Test intersection with candidate segments only
                intersected = False
                for seg_idx in candidates:
                    if self._line_box_intersection(
                        self.x1[seg_idx], self.y1[seg_idx],
                        self.x2[seg_idx], self.y2[seg_idx],
                        box_x_min, box_y_min, box_x_max, box_y_max
                    ):
                        intersected = True
                        break

                if intersected:
                    occupied_count += 1

        return occupied_count

    def _line_box_intersection(self, x1: float, y1: float, x2: float, y2: float,
                             box_x_min: float, box_y_min: float,
                             box_x_max: float, box_y_max: float) -> bool:
        """
        Fast line-box intersection test using Liang-Barsky algorithm.

        Single segment version for spatial index queries.
        """
        dx = x2 - x1
        dy = y2 - y1

        # Handle degenerate cases
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            # Point segment
            return (box_x_min <= x1 <= box_x_max and
                   box_y_min <= y1 <= box_y_max)

        # Liang-Barsky clipping parameters
        p = [-dx, dx, -dy, dy]
        q = [x1 - box_x_min, box_x_max - x1, y1 - box_y_min, box_y_max - y1]

        u_min, u_max = 0.0, 1.0

        for i in range(4):
            if abs(p[i]) < 1e-10:
                # Line is parallel to this edge
                if q[i] < 0:
                    return False  # Outside and parallel
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    # Entering edge
                    u_min = max(u_min, t)
                else:
                    # Exiting edge
                    u_max = min(u_max, t)

                if u_min > u_max:
                    return False  # No intersection

        return u_min <= u_max

    def get_performance_stats(self) -> dict:
        """Get performance statistics including spatial index info."""
        stats = self.spatial_index.get_stats()
        stats['segments_total'] = self.segments.n_segments
        return stats