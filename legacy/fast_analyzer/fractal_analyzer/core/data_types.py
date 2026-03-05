"""
Data types and structures for FastFractalAnalyzer v3.
Clean, type-safe data structures optimized for performance.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class FractalResult:
    """Results from fractal dimension analysis."""

    dimension: float
    r_squared: float
    box_sizes: np.ndarray
    box_counts: np.ndarray
    slope: float
    intercept: float
    scaling_range: Tuple[float, float]
    n_points: int

    @property
    def is_valid(self) -> bool:
        """Check if result meets quality thresholds."""
        return (
            self.r_squared >= 0.99 and
            self.n_points >= 5 and
            0.5 <= self.dimension <= 3.0
        )


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else float('inf')


class SegmentArray:
    """
    Optimized container for line segments using NumPy arrays.

    Internal storage: segments[i] = [[x1, y1], [x2, y2]]
    Shape: (n_segments, 2, 2)
    """

    def __init__(self, segments: np.ndarray):
        if segments.ndim != 3 or segments.shape[1:] != (2, 2):
            raise ValueError("Segments must have shape (n_segments, 2, 2)")

        self._segments = segments.astype(np.float64)
        self._bbox: Optional[BoundingBox] = None
        self._total_length: Optional[float] = None

    @classmethod
    def from_list(cls, segment_list) -> 'SegmentArray':
        """Create from list of ((x1,y1), (x2,y2)) tuples."""
        segments = np.array(segment_list, dtype=np.float64)
        return cls(segments)

    @classmethod
    def from_file(cls, filename: str) -> 'SegmentArray':
        """Load segments from text file (x1 y1 x2 y2 format)."""
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 4:
            raise ValueError(f"Expected 4 columns (x1 y1 x2 y2), got {data.shape[1]}")

        segments = data.reshape(-1, 2, 2)
        return cls(segments)

    @property
    def segments(self) -> np.ndarray:
        """Access to underlying segment array (read-only view)."""
        return self._segments

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return len(self._segments)

    @property
    def start_points(self) -> np.ndarray:
        """Start points of all segments. Shape: (n_segments, 2)"""
        return self._segments[:, 0, :]

    @property
    def end_points(self) -> np.ndarray:
        """End points of all segments. Shape: (n_segments, 2)"""
        return self._segments[:, 1, :]

    @property
    def bbox(self) -> BoundingBox:
        """Compute and cache bounding box."""
        if self._bbox is None:
            all_points = self._segments.reshape(-1, 2)
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            self._bbox = BoundingBox(
                min_x=min_coords[0],
                min_y=min_coords[1],
                max_x=max_coords[0],
                max_y=max_coords[1]
            )
        return self._bbox

    @property
    def total_length(self) -> float:
        """Compute and cache total length of all segments."""
        if self._total_length is None:
            diff = self.end_points - self.start_points
            lengths = np.sqrt(np.sum(diff**2, axis=1))
            self._total_length = np.sum(lengths)
        return self._total_length

    def __len__(self) -> int:
        return self.n_segments

    def __repr__(self) -> str:
        return f"SegmentArray({self.n_segments} segments, bbox={self.bbox})"