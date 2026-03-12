"""
GPU-accelerated 2D segment-box counting using Numba CUDA.

Provides CUDA kernels for Liang-Barsky line segment vs 2D box intersection,
used for multifractal box counting of interface contours. One thread per
segment, atomicAdd to accumulate segment counts per box.

Falls back gracefully when CUDA is not available.
"""

import numpy as np
from typing import Tuple

# Check CUDA availability
try:
    from numba import cuda
    if cuda.is_available():
        HAS_CUDA = True
    else:
        HAS_CUDA = False
except (ImportError, Exception):
    HAS_CUDA = False


if HAS_CUDA:
    from numba import cuda as _cuda

    @_cuda.jit(device=True)
    def _liang_barsky_2d(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """
        Device function: Liang-Barsky segment-box intersection test.

        Returns 1 if segment intersects box, 0 if not.
        """
        dx = x2 - x1
        dy = y2 - y1

        t_min = 0.0
        t_max = 1.0

        # Edge 0: left  (p=-dx, q=x1-xmin)
        p = -dx
        q = x1 - xmin
        if p == 0.0:
            if q < 0.0:
                return 0
        else:
            t = q / p
            if p < 0.0:
                if t > t_min:
                    t_min = t
            else:
                if t < t_max:
                    t_max = t
            if t_min > t_max:
                return 0

        # Edge 1: right (p=dx, q=xmax-x1)
        p = dx
        q = xmax - x1
        if p == 0.0:
            if q < 0.0:
                return 0
        else:
            t = q / p
            if p < 0.0:
                if t > t_min:
                    t_min = t
            else:
                if t < t_max:
                    t_max = t
            if t_min > t_max:
                return 0

        # Edge 2: bottom (p=-dy, q=y1-ymin)
        p = -dy
        q = y1 - ymin
        if p == 0.0:
            if q < 0.0:
                return 0
        else:
            t = q / p
            if p < 0.0:
                if t > t_min:
                    t_min = t
            else:
                if t < t_max:
                    t_max = t
            if t_min > t_max:
                return 0

        # Edge 3: top (p=dy, q=ymax-y1)
        p = dy
        q = ymax - y1
        if p == 0.0:
            if q < 0.0:
                return 0
        else:
            t = q / p
            if p < 0.0:
                if t > t_min:
                    t_min = t
            else:
                if t < t_max:
                    t_max = t
            if t_min > t_max:
                return 0

        return 1

    @_cuda.jit
    def _count_segments_kernel_2d(
        segments,       # float64[n_segs, 4] — (x1, y1, x2, y2)
        delta,          # float64
        inv_delta,      # float64
        domain_min_x, domain_min_y,
        nx, ny,
        box_counts,     # int32[nx * ny] — atomicAdd
    ):
        """
        CUDA kernel: one thread per segment.

        For each segment, compute bounding box -> candidate box range,
        run Liang-Barsky test on each candidate, atomicAdd to box_counts.
        """
        tid = _cuda.grid(1)
        n_segs = segments.shape[0]
        if tid >= n_segs:
            return

        x1 = segments[tid, 0]
        y1 = segments[tid, 1]
        x2 = segments[tid, 2]
        y2 = segments[tid, 3]

        # Segment bounding box -> box index range
        if x1 < x2:
            s_min_x = x1
            s_max_x = x2
        else:
            s_min_x = x2
            s_max_x = x1

        if y1 < y2:
            s_min_y = y1
            s_max_y = y2
        else:
            s_min_y = y2
            s_max_y = y1

        # Expand candidate range by 1 to handle segments exactly on box boundaries
        i_min = int((s_min_x - domain_min_x) * inv_delta) - 1
        i_max = int((s_max_x - domain_min_x) * inv_delta) + 1
        j_min = int((s_min_y - domain_min_y) * inv_delta) - 1
        j_max = int((s_max_y - domain_min_y) * inv_delta) + 1

        # Clamp to grid bounds
        if i_min < 0:
            i_min = 0
        if j_min < 0:
            j_min = 0
        if i_max >= nx:
            i_max = nx - 1
        if j_max >= ny:
            j_max = ny - 1

        # Test each candidate box
        for i in range(i_min, i_max + 1):
            box_xmin = domain_min_x + i * delta
            box_xmax = domain_min_x + (i + 1) * delta
            for j in range(j_min, j_max + 1):
                box_ymin = domain_min_y + j * delta
                box_ymax = domain_min_y + (j + 1) * delta

                if _liang_barsky_2d(x1, y1, x2, y2,
                                    box_xmin, box_ymin, box_xmax, box_ymax):
                    idx = i * ny + j
                    _cuda.atomic.add(box_counts, idx, 1)

    @_cuda.jit
    def _occupied_boxes_kernel_2d(
        segments,       # float64[n_segs, 4]
        delta,          # float64
        inv_delta,      # float64
        domain_min_x, domain_min_y,
        nx, ny,
        occupied,       # uint8[nx * ny] — idempotent write of 1
    ):
        """
        CUDA kernel for monofractal box counting (occupied/not).

        Multiple threads writing 1 to the same cell is idempotent.
        """
        tid = _cuda.grid(1)
        n_segs = segments.shape[0]
        if tid >= n_segs:
            return

        x1 = segments[tid, 0]
        y1 = segments[tid, 1]
        x2 = segments[tid, 2]
        y2 = segments[tid, 3]

        if x1 < x2:
            s_min_x = x1
            s_max_x = x2
        else:
            s_min_x = x2
            s_max_x = x1

        if y1 < y2:
            s_min_y = y1
            s_max_y = y2
        else:
            s_min_y = y2
            s_max_y = y1

        # Expand candidate range by 1 to handle segments exactly on box boundaries
        i_min = int((s_min_x - domain_min_x) * inv_delta) - 1
        i_max = int((s_max_x - domain_min_x) * inv_delta) + 1
        j_min = int((s_min_y - domain_min_y) * inv_delta) - 1
        j_max = int((s_max_y - domain_min_y) * inv_delta) + 1

        if i_min < 0:
            i_min = 0
        if j_min < 0:
            j_min = 0
        if i_max >= nx:
            i_max = nx - 1
        if j_max >= ny:
            j_max = ny - 1

        for i in range(i_min, i_max + 1):
            box_xmin = domain_min_x + i * delta
            box_xmax = domain_min_x + (i + 1) * delta
            for j in range(j_min, j_max + 1):
                box_ymin = domain_min_y + j * delta
                box_ymax = domain_min_y + (j + 1) * delta

                if _liang_barsky_2d(x1, y1, x2, y2,
                                    box_xmin, box_ymin, box_xmax, box_ymax):
                    idx = i * ny + j
                    occupied[idx] = 1


def _prepare_segments_array(segments):
    """
    Convert list of ((x1,y1),(x2,y2)) tuples to float64 array [n, 4].
    """
    n = len(segments)
    arr = np.empty((n, 4), dtype=np.float64)
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        arr[i, 0] = x1
        arr[i, 1] = y1
        arr[i, 2] = x2
        arr[i, 3] = y2
    return arr


def count_segments_per_box_gpu(
    segments,
    box_size: float,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """
    GPU-accelerated segment counting per 2D box.

    Args:
        segments: Either list of ((x1,y1),(x2,y2)) or float64 array [n, 4]
        box_size: Box side length
        domain_min: (2,) array [xmin, ymin]
        domain_max: (2,) array [xmax, ymax]

    Returns:
        (box_counts_2d, nx, ny): 2D array of counts and grid dimensions
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available")

    if isinstance(segments, list):
        seg_arr = _prepare_segments_array(segments)
    else:
        seg_arr = np.ascontiguousarray(segments, dtype=np.float64)

    nx = max(1, int(np.ceil((domain_max[0] - domain_min[0]) / box_size)))
    ny = max(1, int(np.ceil((domain_max[1] - domain_min[1]) / box_size)))
    grid_size = nx * ny

    inv_delta = 1.0 / box_size
    n_segs = seg_arr.shape[0]

    d_segs = cuda.to_device(seg_arr)
    d_counts = cuda.to_device(np.zeros(grid_size, dtype=np.int32))

    threads_per_block = 256
    blocks = (n_segs + threads_per_block - 1) // threads_per_block

    _count_segments_kernel_2d[blocks, threads_per_block](
        d_segs, box_size, inv_delta,
        domain_min[0], domain_min[1],
        nx, ny,
        d_counts
    )

    counts_flat = d_counts.copy_to_host()
    box_counts_2d = counts_flat.reshape(nx, ny)

    return box_counts_2d, nx, ny


def box_counting_gpu(
    segments,
    box_size: float,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
) -> int:
    """
    GPU-accelerated monofractal box counting (occupied count).

    Args:
        segments: Either list of ((x1,y1),(x2,y2)) or float64 array [n, 4]
        box_size: Box side length
        domain_min: (2,) array [xmin, ymin]
        domain_max: (2,) array [xmax, ymax]

    Returns:
        Number of occupied boxes
    """
    if not HAS_CUDA:
        raise RuntimeError("CUDA not available")

    if isinstance(segments, list):
        seg_arr = _prepare_segments_array(segments)
    else:
        seg_arr = np.ascontiguousarray(segments, dtype=np.float64)

    nx = max(1, int(np.ceil((domain_max[0] - domain_min[0]) / box_size)))
    ny = max(1, int(np.ceil((domain_max[1] - domain_min[1]) / box_size)))
    grid_size = nx * ny

    inv_delta = 1.0 / box_size
    n_segs = seg_arr.shape[0]

    d_segs = cuda.to_device(seg_arr)
    d_occupied = cuda.to_device(np.zeros(grid_size, dtype=np.uint8))

    threads_per_block = 256
    blocks = (n_segs + threads_per_block - 1) // threads_per_block

    _occupied_boxes_kernel_2d[blocks, threads_per_block](
        d_segs, box_size, inv_delta,
        domain_min[0], domain_min[1],
        nx, ny,
        d_occupied
    )

    occupied = d_occupied.copy_to_host()
    return int(occupied.sum())


def check_cuda_available() -> bool:
    """Check if CUDA is available for GPU acceleration."""
    return HAS_CUDA
