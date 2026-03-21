"""
Multifractal Analysis Module

This module provides advanced multifractal analysis capabilities for 
interface characterization, extracted and modernized from rt_analyzer.py.

Classes:
    MultifractalAnalyzer: Core multifractal spectrum analysis
    
Functions:
    compute_singularity_spectrum: Convert tau(q) to f(alpha) spectrum
    plot_multifractal_spectrum: Visualization utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from scipy import stats
import time
import os
from collections import defaultdict

# GPU acceleration for multifractal box counting
try:
    from rt_analyzer.fast_counting_gpu_2d import HAS_CUDA, count_segments_per_box_gpu, _prepare_segments_array
except ImportError:
    try:
        from .fast_counting_gpu_2d import HAS_CUDA, count_segments_per_box_gpu, _prepare_segments_array
    except ImportError:
        try:
            from fast_counting_gpu_2d import HAS_CUDA, count_segments_per_box_gpu, _prepare_segments_array
        except ImportError:
            HAS_CUDA = False

class MultifractalAnalyzer:
    """
    Advanced multifractal analysis for interface characterization.
    
    Extracted from rt_analyzer.py and modernized for the FractalAnalyzer framework.
    """
    
    def __init__(self, debug: bool = False, use_spatial_index: bool = True):
        """
        Initialize MultifractalAnalyzer.
        
        Args:
            debug: Enable debug output
            use_spatial_index: Use spatial indexing for performance
        """
        self.debug = debug
        self.use_spatial_index = use_spatial_index
        
    def create_spatial_index(self, segments, min_x, min_y, max_x, max_y, grid_size):
        """Create spatial index for fast segment lookup."""
        grid_width = int(np.ceil((max_x - min_x) / grid_size))
        grid_height = int(np.ceil((max_y - min_y) / grid_size))
        
        segment_grid = defaultdict(list)
        
        for idx, ((x1, y1), (x2, y2)) in enumerate(segments):
            # Find grid cells this segment might touch
            seg_min_x, seg_max_x = min(x1, x2), max(x1, x2)
            seg_min_y, seg_max_y = min(y1, y2), max(y1, y2)
            
            min_cell_x = max(0, int((seg_min_x - min_x) / grid_size))
            max_cell_x = min(int((seg_max_x - min_x) / grid_size) + 1, grid_width)
            min_cell_y = max(0, int((seg_min_y - min_y) / grid_size))
            max_cell_y = min(int((seg_max_y - min_y) / grid_size) + 1, grid_height)
            
            for cell_x in range(min_cell_x, max_cell_x):
                for cell_y in range(min_cell_y, max_cell_y):
                    segment_grid[(cell_x, cell_y)].append(idx)
        
        return segment_grid, grid_width, grid_height

    def liang_barsky_line_box_intersection(self, x1, y1, x2, y2, xmin, ymin, xmax, ymax):
        """Test if line segment intersects with box using Liang-Barsky algorithm."""
        dx = x2 - x1
        dy = y2 - y1
        
        # Parametric line: P = P1 + t*(P2-P1) where t ∈ [0,1]
        p = [-dx, dx, -dy, dy]
        q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
        
        t_min, t_max = 0.0, 1.0
        
        for i in range(4):
            if p[i] == 0:
                # Line is parallel to this boundary
                if q[i] < 0:
                    return False  # Line is outside
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    t_min = max(t_min, t)
                else:
                    t_max = min(t_max, t)
                
                if t_min > t_max:
                    return False
        
        return True
        
    def _get_grid_offsets(self, box_size):
        """Get adaptive grid offset fractions based on box size (matches FractalAnalyzer)."""
        if box_size < 0.005:
            return np.linspace(0, 0.75, 4)  # 4x4 = 16 tests
        elif box_size < 0.02:
            return np.linspace(0, 0.5, 3)   # 3x3 = 9 tests
        else:
            return np.linspace(0, 0.5, 2)   # 2x2 = 4 tests

    def _count_boxes_per_segment_with_offset_cpu(self, segments, box_size,
                                                  offset_x, offset_y, max_x, max_y,
                                                  segment_grid, grid_width, grid_height,
                                                  cell_size, spatial_min_x, spatial_min_y):
        """Count segments per box with a specific grid offset (CPU path).

        Returns:
            (box_counts_2d, num_boxes_x, num_boxes_y)
        """
        num_boxes_x = int(np.ceil((max_x - offset_x) / box_size))
        num_boxes_y = int(np.ceil((max_y - offset_y) / box_size))
        box_counts = np.zeros((num_boxes_x, num_boxes_y))

        for i in range(num_boxes_x):
            for j in range(num_boxes_y):
                box_xmin = offset_x + i * box_size
                box_ymin = offset_y + j * box_size
                box_xmax = box_xmin + box_size
                box_ymax = box_ymin + box_size

                min_cell_x = max(0, int((box_xmin - spatial_min_x) / cell_size))
                max_cell_x = min(grid_width - 1, int((box_xmax - spatial_min_x) / cell_size))
                min_cell_y = max(0, int((box_ymin - spatial_min_y) / cell_size))
                max_cell_y = min(grid_height - 1, int((box_ymax - spatial_min_y) / cell_size))

                segments_to_check = set()
                for cell_x in range(min_cell_x, max_cell_x + 1):
                    for cell_y in range(min_cell_y, max_cell_y + 1):
                        segments_to_check.update(segment_grid.get((cell_x, cell_y), []))

                count = 0
                for seg_idx in segments_to_check:
                    (x1, y1), (x2, y2) = segments[seg_idx]
                    if self.liang_barsky_line_box_intersection(
                            x1, y1, x2, y2, box_xmin, box_ymin, box_xmax, box_ymax):
                        count += 1

                box_counts[i, j] = count

        return box_counts, num_boxes_x, num_boxes_y

    def _count_with_grid_optimization(self, segments, box_size,
                                       min_x, min_y, max_x, max_y,
                                       use_gpu, seg_arr, domain_max,
                                       segment_grid, grid_width, grid_height, cell_size):
        """Count per-box segments with grid optimization — returns distribution for best offset.

        Tests multiple grid offsets and selects the one with minimum occupied box count,
        matching the validated monofractal approach (Theiler 1990).

        Returns:
            (best_occupied_boxes, best_n_occupied, grid_tests)
            where best_occupied_boxes is the 1D array of non-zero per-box segment counts
        """
        offset_fractions = self._get_grid_offsets(box_size)

        best_n_occupied = float('inf')
        best_occupied = None
        grid_tests = 0

        for dx_frac in offset_fractions:
            for dy_frac in offset_fractions:
                grid_tests += 1
                offset_x = min_x + dx_frac * box_size
                offset_y = min_y + dy_frac * box_size

                if use_gpu:
                    d_min = np.array([offset_x, offset_y], dtype=np.float64)
                    box_counts, nx, ny = count_segments_per_box_gpu(
                        seg_arr, box_size, d_min, domain_max)
                else:
                    box_counts, nx, ny = self._count_boxes_per_segment_with_offset_cpu(
                        segments, box_size, offset_x, offset_y, max_x, max_y,
                        segment_grid, grid_width, grid_height, cell_size, min_x, min_y)

                n_occupied = np.count_nonzero(box_counts)

                if n_occupied < best_n_occupied:
                    best_n_occupied = n_occupied
                    best_occupied = box_counts[box_counts > 0].flatten().copy()

        return best_occupied if best_occupied is not None else np.array([]), best_n_occupied, grid_tests

    def _enhanced_boundary_removal(self, box_sizes, occupied_counts):
        """Enhanced boundary artifact detection and removal for scale data.

        Matches FractalAnalyzer.enhanced_boundary_removal() logic.
        Returns boolean mask of scales to keep.

        Args:
            box_sizes: array of box sizes
            occupied_counts: array of occupied box counts (one per box size)

        Returns:
            mask: boolean array, True for scales to keep
        """
        n = len(box_sizes)
        mask = np.ones(n, dtype=bool)

        if n <= 8:
            return mask

        log_sizes = np.log(box_sizes)
        log_counts = np.log(occupied_counts.astype(float))

        segment_size = max(3, n // 4)

        if n < 3 * segment_size:
            return mask

        try:
            slope_first, _, r_first, _, _ = stats.linregress(
                log_sizes[:segment_size], log_counts[:segment_size])
            slope_middle, _, r_middle, _, _ = stats.linregress(
                log_sizes[segment_size:3*segment_size], log_counts[segment_size:3*segment_size])
            slope_last, _, r_last, _, _ = stats.linregress(
                log_sizes[-segment_size:], log_counts[-segment_size:])

            r2_first = r_first ** 2
            r2_last = r_last ** 2

            trim_start = 0
            trim_end = 0

            SLOPE_DEVIATION_THRESHOLD = 0.12
            MIN_R_SQUARED_THRESHOLD = 0.99

            first_slope_dev = abs(slope_first - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
            if first_slope_dev > SLOPE_DEVIATION_THRESHOLD or r2_first < MIN_R_SQUARED_THRESHOLD:
                trim_start = 1
                print(f"  Boundary artifact at start: slope deviation {first_slope_dev:.3f}, R² {r2_first:.3f}")

            last_slope_dev = abs(slope_last - slope_middle) / abs(slope_middle) if slope_middle != 0 else 0
            if last_slope_dev > SLOPE_DEVIATION_THRESHOLD or r2_last < MIN_R_SQUARED_THRESHOLD:
                trim_end = 1
                print(f"  Boundary artifact at end: slope deviation {last_slope_dev:.3f}, R² {r2_last:.3f}")

            remaining = n - trim_start - trim_end
            if remaining >= 4:
                if trim_start > 0:
                    mask[:trim_start] = False
                if trim_end > 0:
                    mask[-trim_end:] = False

                trimmed_sizes = log_sizes[mask]
                trimmed_counts = log_counts[mask]
                _, _, r_new, _, _ = stats.linregress(trimmed_sizes, trimmed_counts)
                print(f"  R² after boundary removal: {r_new**2:.4f}")
            else:
                mask[:] = True  # not enough points, keep all

        except Exception as e:
            print(f"  Warning: boundary detection failed: {e}")

        return mask

    def compute_multifractal_spectrum(self, segments: List[Tuple],
                                    min_box_size: Optional[float] = None,
                                    q_values: Optional[List[float]] = None,
                                    output_dir: Optional[str] = None,
                                    time_value: Optional[float] = None,
                                    rt_physics=None,
                                    box_sizes: Optional[np.ndarray] = None) -> Dict:
        """
        Compute multifractal spectrum from interface segments.

        Uses grid-optimized box counting (multiple grid offsets, take minimum
        occupied count) to match the validated monofractal approach. Applies
        enhanced boundary removal before fitting.

        Args:
            segments: List of line segments as ((x1,y1), (x2,y2)) tuples
            min_box_size: Minimum box size for analysis (default: auto-estimate)
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            output_dir: Directory to save results (default: None)
            time_value: Time value for labeling plots (default: None)
            rt_physics: Optional RTPhysics instance for nondimensional box sizes

        Returns:
            dict: Multifractal spectrum results
        """
        if not segments:
            print("No interface segments provided. Skipping multifractal analysis.")
            return None

        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        q_values = np.array(q_values)

        print(f"Performing multifractal analysis with {len(q_values)} q-values")
        print(f"Using {len(segments)} interface segments")

        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Calculate extent for max box size
        min_x = min(min(s[0][0], s[1][0]) for s in segments)
        max_x = max(max(s[0][0], s[1][0]) for s in segments)
        min_y = min(min(s[0][1], s[1][1]) for s in segments)
        max_y = max(max(s[0][1], s[1][1]) for s in segments)

        extent = max(max_x - min_x, max_y - min_y)
        max_box_size = extent / 2

        # Auto-estimate min_box_size if not provided
        if min_box_size is None:
            lengths = [np.sqrt((s[1][0]-s[0][0])**2 + (s[1][1]-s[0][1])**2) for s in segments]
            avg_length = np.mean(lengths)
            min_box_size = avg_length * 2
            print(f"Auto-estimated min_box_size: {min_box_size:.6f}")

        if box_sizes is not None:
            # Use pre-computed scales (e.g. from Phase 1 monofractal analysis)
            box_sizes = np.sort(np.asarray(box_sizes, dtype=np.float64))[::-1]
            num_box_sizes = len(box_sizes)
            print(f"Using {num_box_sizes} pre-computed box sizes "
                  f"({box_sizes[0]:.6f} to {box_sizes[-1]:.6f})")
        else:
            print(f"Box size range: {min_box_size:.6f} to {max_box_size:.6f}")

            # Generate box sizes
            box_sizes_list = []
            current_size = max_box_size
            box_size_factor = 1.5

            while current_size >= min_box_size:
                box_sizes_list.append(current_size)
                current_size /= box_size_factor

            box_sizes = np.array(box_sizes_list)
            num_box_sizes = len(box_sizes)

        print(f"Using {num_box_sizes} box sizes for analysis")

        # Add small margin to bounding box
        margin = extent * 0.01
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        # Initialize data structures for box counting
        all_box_counts = []
        all_probabilities = []
        occupied_count_per_scale = []  # for boundary removal

        domain_min = np.array([min_x, min_y], dtype=np.float64)
        domain_max = np.array([max_x, max_y], dtype=np.float64)

        use_gpu = HAS_CUDA
        if use_gpu:
            print("Using GPU-accelerated box counting (CUDA) with grid optimization")
            seg_arr = _prepare_segments_array(segments)
        else:
            print("Using CPU box counting with spatial index and grid optimization")
            start_time = time.time()
            grid_size = min_box_size * 2
            segment_grid, grid_width, grid_height = self.create_spatial_index(
                segments, min_x, min_y, max_x, max_y, grid_size)
            print(f"Spatial index created in {time.time() - start_time:.2f} seconds")

        # Analyze each box size with grid optimization
        print("Box counting with grid optimization:")
        print("  Box size    |  Min count |  Grid tests | Time (s)")
        print("  " + "-" * 55)

        for box_idx, box_size in enumerate(box_sizes):
            box_start_time = time.time()

            if not use_gpu:
                best_occupied, best_n, grid_tests = self._count_with_grid_optimization(
                    segments, box_size, min_x, min_y, max_x, max_y,
                    False, None, domain_max,
                    segment_grid, grid_width, grid_height, grid_size)
            else:
                best_occupied, best_n, grid_tests = self._count_with_grid_optimization(
                    segments, box_size, min_x, min_y, max_x, max_y,
                    True, seg_arr, domain_max,
                    None, 0, 0, 0)

            # Calculate probabilities from the best-offset distribution
            total_segments = best_occupied.sum() if len(best_occupied) > 0 else 0
            if total_segments > 0:
                probabilities = best_occupied / total_segments
            else:
                probabilities = np.array([])

            all_box_counts.append(best_occupied)
            all_probabilities.append(probabilities)
            occupied_count_per_scale.append(len(best_occupied))

            elapsed = time.time() - box_start_time
            print(f"  {box_size:.6f}  |  {len(best_occupied):8d}  |  {grid_tests:10d}  |  {elapsed:.2f}")

        # No boundary removal — matches Phase 1 monofractal fitting (all scales used).
        # Boundary removal is available via _enhanced_boundary_removal() if needed.
        occupied_count_per_scale = np.array(occupied_count_per_scale)
        scale_mask = np.ones(num_box_sizes, dtype=bool)

        # Calculate multifractal properties
        print("Calculating multifractal spectrum...")
        
        taus = np.zeros(len(q_values))
        Dqs = np.zeros(len(q_values))
        r_squared = np.zeros(len(q_values))

        # Use only non-boundary scales for fitting
        fit_box_sizes = box_sizes[scale_mask]
        fit_probabilities = [all_probabilities[i] for i in range(num_box_sizes) if scale_mask[i]]
        num_fit_sizes = len(fit_box_sizes)

        for q_idx, q in enumerate(q_values):
            print(f"Processing q = {q:.1f}")

            # tau(1) = 0 exactly (Z_1 = sum(p_i) = 1 for all scales)
            if abs(q - 1.0) < 1e-6:
                taus[q_idx] = 0.0
                r_squared[q_idx] = 1.0
                print(f"  τ(1) = 0.0000 (exact)")
                continue

            # Calculate partition function for each box size
            Z_q = np.zeros(num_fit_sizes)

            for i, probabilities in enumerate(fit_probabilities):
                if len(probabilities) > 0:
                    Z_q[i] = np.sum(probabilities ** q)
                else:
                    Z_q[i] = np.nan

            # Remove NaN values
            valid = ~np.isnan(Z_q)
            if np.sum(valid) < 3:
                print(f"Warning: Not enough valid points for q={q}")
                taus[q_idx] = np.nan
                Dqs[q_idx] = np.nan
                r_squared[q_idx] = np.nan
                continue

            log_eps = np.log(fit_box_sizes[valid])
            log_Z_q = np.log(Z_q[valid])

            # Linear regression to find tau(q)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_Z_q)

            taus[q_idx] = slope
            r_squared[q_idx] = r_value ** 2

            print(f"  τ({q}) = {taus[q_idx]:.4f}, R² = {r_squared[q_idx]:.4f}")

        # Enforce convexity on tau(q) to ensure D_0 >= D_1 >= D_2 (Renyi inequality).
        # Individual regressions can violate convexity due to fitting noise.
        # Project tau(q) onto nearest convex function via constrained least-squares.
        from scipy.optimize import minimize as sp_minimize
        from scipy.interpolate import UnivariateSpline

        valid_tau = ~np.isnan(taus)
        n_valid = np.sum(valid_tau)

        if n_valid >= 4:
            # Enforce convexity only for q >= 0 where partition functions are stable.
            # Negative-q values are sensitive to near-zero probabilities and often
            # produce wildly non-convex tau(q); including them collapses the optimizer.
            mask_pos = (q_values >= -1e-10) & valid_tau
            q_pos = q_values[mask_pos]
            tau_pos_raw = taus[mask_pos].copy()
            n_pos = len(q_pos)

            if n_pos >= 3:
                q0_pos_idx = np.argmin(np.abs(q_pos - 0.0))
                has_q0 = abs(q_pos[q0_pos_idx]) < 1e-6
                q1_pos_idx = np.argmin(np.abs(q_pos - 1.0))
                has_q1 = abs(q_pos[q1_pos_idx] - 1.0) < 1e-6

                def _objective(tau):
                    return np.sum((tau - tau_pos_raw) ** 2)

                constraints = []
                for k in range(n_pos - 2):
                    dq0 = q_pos[k + 1] - q_pos[k]
                    dq1 = q_pos[k + 2] - q_pos[k + 1]
                    def _conv(tau, k=k, dq0=dq0, dq1=dq1):
                        return (tau[k + 1] - tau[k]) / dq0 - (tau[k + 2] - tau[k + 1]) / dq1
                    constraints.append({'type': 'ineq', 'fun': _conv})

                # Pin tau(0) to raw value (preserves D0 = Phase 1 box-counting D)
                if has_q0:
                    raw_tau0 = float(tau_pos_raw[q0_pos_idx])
                    constraints.append({'type': 'eq',
                                        'fun': lambda tau, idx=q0_pos_idx, val=raw_tau0: tau[idx] - val})

                if has_q1:
                    constraints.append({'type': 'eq',
                                        'fun': lambda tau, idx=q1_pos_idx: tau[idx]})

                try:
                    result = sp_minimize(_objective, tau_pos_raw, constraints=constraints,
                                         method='SLSQP', options={'maxiter': 500, 'ftol': 1e-14})
                    if result.success:
                        tau_pos = result.x
                        adj = np.max(np.abs(tau_pos - tau_pos_raw))
                        taus[mask_pos] = tau_pos
                        if adj > 1e-6:
                            print(f"  Convexity adjustment (q>=0): max |Δτ| = {adj:.6f}")
                    else:
                        print(f"  Warning: convexity optimization did not converge, using raw tau")
                except Exception as e:
                    print(f"  Warning: convexity enforcement failed ({e}), using raw tau")

            # Compute D_q from convex tau
            for i, q_val in enumerate(q_values):
                if np.isnan(taus[i]):
                    Dqs[i] = np.nan
                    continue
                if abs(q_val - 1.0) < 1e-6:
                    # D_1 = d(tau)/dq at q=1 via central finite difference
                    if i > 0 and i < len(q_values) - 1:
                        if not np.isnan(taus[i - 1]) and not np.isnan(taus[i + 1]):
                            Dqs[i] = (taus[i + 1] - taus[i - 1]) / \
                                      (q_values[i + 1] - q_values[i - 1])
                else:
                    Dqs[i] = taus[i] / (q_val - 1)

            for i, q_val in enumerate(q_values):
                if not np.isnan(Dqs[i]):
                    print(f"  D({q_val:.1f}) = {Dqs[i]:.4f}")

            # Legendre transform for f(alpha) using spline on convex tau
            alpha = np.full(len(q_values), np.nan)
            f_alpha = np.full(len(q_values), np.nan)

            print("Calculating multifractal spectrum f(α)...")

            q_valid = q_values[valid_tau]
            tau_valid = taus[valid_tau]
            try:
                spline = UnivariateSpline(q_valid, tau_valid, k=3, s=0)
                dtau_dq = spline.derivative()(q_valid)
                j = 0
                for i in range(len(q_values)):
                    if valid_tau[i]:
                        alpha[i] = dtau_dq[j]
                        f_alpha[i] = q_values[i] * alpha[i] - taus[i]
                        j += 1
                        print(f"  q = {q_values[i]:.1f}, α = {alpha[i]:.4f}, f(α) = {f_alpha[i]:.4f}")
            except Exception:
                for i, q_val in enumerate(q_values):
                    if np.isnan(taus[i]):
                        continue
                    if i > 0 and i < len(q_values) - 1:
                        if not np.isnan(taus[i - 1]) and not np.isnan(taus[i + 1]):
                            alpha[i] = (taus[i + 1] - taus[i - 1]) / \
                                        (q_values[i + 1] - q_values[i - 1])
                    elif i == 0 and not np.isnan(taus[i + 1]):
                        alpha[i] = (taus[i + 1] - taus[i]) / (q_values[i + 1] - q_values[i])
                    elif i == len(q_values) - 1 and not np.isnan(taus[i - 1]):
                        alpha[i] = (taus[i] - taus[i - 1]) / (q_values[i] - q_values[i - 1])
                    if not np.isnan(alpha[i]):
                        f_alpha[i] = q_val * alpha[i] - taus[i]
                    print(f"  q = {q_val:.1f}, α = {alpha[i]:.4f}, f(α) = {f_alpha[i]:.4f}")
        else:
            alpha = np.full(len(q_values), np.nan)
            f_alpha = np.full(len(q_values), np.nan)
            for i, q_val in enumerate(q_values):
                if np.isnan(taus[i]) or abs(q_val - 1.0) < 1e-6:
                    Dqs[i] = np.nan
                    continue
                Dqs[i] = taus[i] / (q_val - 1)
        
        # Calculate multifractal parameters
        valid_idx = ~np.isnan(Dqs)
        if np.sum(valid_idx) >= 3:
            D0 = Dqs[np.searchsorted(q_values, 0)] if 0 in q_values else np.nan
            D1 = Dqs[np.searchsorted(q_values, 1)] if 1 in q_values else np.nan
            D2 = Dqs[np.searchsorted(q_values, 2)] if 2 in q_values else np.nan
            
            # Width of multifractal spectrum
            valid = ~np.isnan(alpha)
            if np.sum(valid) >= 2:
                alpha_width = np.max(alpha[valid]) - np.min(alpha[valid])
            else:
                alpha_width = np.nan

            # Calculate degree of multifractality using available q-values
            valid_idx = ~np.isnan(Dqs)
            if np.sum(valid_idx) >= 3:
                # Use extreme available q-values
                q_min_idx = np.where(q_values == np.min(q_values[valid_idx]))[0][0]
                q_max_idx = np.where(q_values == np.max(q_values[valid_idx]))[0][0]
                degree_multifractality = Dqs[q_min_idx] - Dqs[q_max_idx]
                print(f"  Degree of multifractality (D({q_values[q_min_idx]}) - D({q_values[q_max_idx]})): {degree_multifractality:.4f}")
            else:
                degree_multifractality = np.nan
                print("  Warning: Insufficient valid q-values for degree calculation")
            
            print(f"Multifractal parameters:")
            print(f"  D(0) = {D0:.4f} (capacity dimension)")
            print(f"  D(1) = {D1:.4f} (information dimension)")
            print(f"  D(2) = {D2:.4f} (correlation dimension)")
            print(f"  α width = {alpha_width:.4f}")
            print(f"  Degree of multifractality = {degree_multifractality:.4f}")
        else:
            D0 = D1 = D2 = alpha_width = degree_multifractality = np.nan
            print("Warning: Not enough valid points to calculate multifractal parameters")
        
        # Plot results if output directory provided
        if output_dir:
            self._create_multifractal_plots(q_values, Dqs, alpha, f_alpha, r_squared, 
                                          D0, output_dir, time_value)
            self._save_multifractal_results(q_values, taus, Dqs, alpha, f_alpha, r_squared,
                                          D0, D1, D2, alpha_width, degree_multifractality,
                                          output_dir, time_value)
        
        # Return results
        results = {
            'q_values': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared,
            'D0': D0,
            'D1': D1,
            'D2': D2,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_multifractality,
            'time': time_value,
            'box_sizes': box_sizes,
        }

        if rt_physics is not None:
            results['box_sizes_nondim'] = rt_physics.nondim_box_size(box_sizes)
            if time_value is not None:
                results['dimensionless_time'] = float(rt_physics.nondim_time(time_value))

        return results

    def _create_multifractal_plots(self, q_values, Dqs, alpha, f_alpha, r_squared, 
                                 D0, output_dir, time_value):
        """Create multifractal analysis plots."""
        time_str = f" at t = {time_value:.2f}" if time_value is not None else ""
        
        # Plot D(q) vs q
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(Dqs)
        plt.plot(q_values[valid], Dqs[valid], 'bo-', markersize=4)
        
        if not np.isnan(D0):
            plt.axhline(y=D0, color='r', linestyle='--', 
                       label=f"D(0) = {D0:.4f}")
        
        plt.xlabel('q')
        plt.ylabel('D(q)')
        plt.title(f'Generalized Dimensions D(q){time_str}')
        plt.grid(True)
        if not np.isnan(D0):
            plt.legend()
        plt.savefig(os.path.join(output_dir, "multifractal_dimensions.png"), dpi=300)
        plt.close()
        
        # Plot f(alpha) vs alpha (multifractal spectrum)
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(alpha) & ~np.isnan(f_alpha)
        plt.plot(alpha[valid], f_alpha[valid], 'bo-', markersize=4)
        
        # Add selected q values as annotations
        q_to_highlight = [-5, -2, 0, 2, 5]
        for q_val in q_to_highlight:
            if q_val in q_values:
                idx = np.searchsorted(q_values, q_val)
                if idx < len(q_values) and valid[idx]:
                    plt.annotate(f"q={q_values[idx]}", 
                                (alpha[idx], f_alpha[idx]),
                                xytext=(5, 0), textcoords='offset points')
        
        plt.xlabel('α')
        plt.ylabel('f(α)')
        plt.title(f'Multifractal Spectrum f(α){time_str}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "multifractal_spectrum.png"), dpi=300)
        plt.close()
        
        # Plot R² values
        plt.figure(figsize=(10, 6))
        valid = ~np.isnan(r_squared)
        plt.plot(q_values[valid], r_squared[valid], 'go-', markersize=4)
        plt.xlabel('q')
        plt.ylabel('R²')
        plt.title(f'Fit Quality for Different q Values{time_str}')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "multifractal_r_squared.png"), dpi=300)
        plt.close()

    def _save_multifractal_results(self, q_values, taus, Dqs, alpha, f_alpha, r_squared,
                                 D0, D1, D2, alpha_width, degree_multifractality,
                                 output_dir, time_value):
        """Save multifractal results to CSV files."""
        # Save detailed results
        results_df = pd.DataFrame({
            'q': q_values,
            'tau': taus,
            'Dq': Dqs,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'r_squared': r_squared
        })
        results_df.to_csv(os.path.join(output_dir, "multifractal_results.csv"), index=False)
        
        # Save multifractal parameters
        params_df = pd.DataFrame({
            'Parameter': ['Time', 'D0', 'D1', 'D2', 'alpha_width', 'degree_multifractality'],
            'Value': [time_value if time_value is not None else np.nan, 
                     D0, D1, D2, alpha_width, degree_multifractality]
        })
        params_df.to_csv(os.path.join(output_dir, "multifractal_parameters.csv"), index=False)

    def print_multifractal_summary(self, mf_results):
        """Print a nice summary of multifractal results."""
        if not mf_results:
            print("No multifractal results to display")
            return
            
        print(f"\n📊 MULTIFRACTAL ANALYSIS SUMMARY")
        print(f"=" * 50)
        if mf_results.get('time') is not None:
            print(f"Time: {mf_results['time']:.3f}")
        print(f"")
        print(f"Generalized Dimensions:")
        print(f"  D(0) = {mf_results['D0']:.4f} (Capacity dimension)")
        print(f"  D(1) = {mf_results['D1']:.4f} (Information dimension)")  
        print(f"  D(2) = {mf_results['D2']:.4f} (Correlation dimension)")
        print(f"")
        print(f"Multifractal Properties:")
        print(f"  α width = {mf_results['alpha_width']:.4f}")
        print(f"  Degree of multifractality = {mf_results['degree_multifractality']:.4f}")
        print(f"")
        
        # Interpretation
        if mf_results['degree_multifractality'] > 0.1:
            print(f"  🔍 Interface shows multifractal behavior")
        else:
            print(f"  📏 Interface appears monofractal")
            
        if mf_results['D0'] > 1.8:
            print(f"  🌊 Highly complex, space-filling interface")
        elif mf_results['D0'] > 1.5:
            print(f"  🌀 Moderately complex interface")
        else:
            print(f"  📐 Relatively smooth interface")

    def analyze_multifractal_evolution(self, segments_data: Dict, output_dir: Optional[str] = None,
                                     q_values: Optional[List[float]] = None,
                                     box_sizes_data: Optional[Dict] = None) -> List[Dict]:
        """
        Analyze how multifractal properties evolve over time or across resolutions.
        
        Args:
            segments_data: Dict mapping times/resolutions to segments data
                         e.g. {0.1: segments_list, 0.2: segments_list} for time series
                         or {100: segments_list, 200: segments_list} for resolutions
            output_dir: Directory to save results
            q_values: List of q moments to analyze (default: -5 to 5 in 0.5 steps)
            
        Returns:
            List[Dict]: Multifractal evolution results
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set default q values if not provided
        if q_values is None:
            q_values = np.arange(-5, 5.1, 0.5)
        
        # Determine type of analysis (time or resolution)
        keys = list(segments_data.keys())
        is_time_series = all(isinstance(k, (int, float)) for k in keys)
        
        if is_time_series:
            print(f"Analyzing multifractal evolution over time/resolution series: {sorted(keys)}")
            x_label = 'Time/Resolution'
            series_name = "parameter"
        else:
            print(f"Analyzing multifractal evolution across parameters: {sorted(keys)}")
            x_label = 'Parameter'
            series_name = "parameter"
        
        # Initialize results storage
        results = []
        
        # Process each segments dataset
        for key, segments in sorted(segments_data.items()):
            print(f"\nProcessing {series_name} = {key}")
            
            try:
                # Create subdirectory for this point
                if output_dir:
                    point_dir = os.path.join(output_dir, f"{series_name}_{key}")
                    os.makedirs(point_dir, exist_ok=True)
                else:
                    point_dir = None
                
                # Perform multifractal analysis
                key_boxes = None
                if box_sizes_data is not None and key in box_sizes_data:
                    key_boxes = box_sizes_data[key]
                mf_results = self.compute_multifractal_spectrum(
                    segments, q_values=q_values, output_dir=point_dir, time_value=key,
                    box_sizes=key_boxes
                )
                
                if mf_results:
                    # Store results with the key (time or resolution)
                    mf_results[series_name] = key
                    results.append(mf_results)
                
            except Exception as e:
                print(f"Error processing {series_name}={key}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create summary plots
        if results and output_dir:
            self._create_evolution_plots(results, series_name, x_label, output_dir, q_values)
            self._save_evolution_summary(results, series_name, output_dir)
        
        return results
    
    def _create_evolution_plots(self, results: List[Dict], series_name: str, 
                              x_label: str, output_dir: str, q_values: np.ndarray):
        """Create evolution analysis plots."""
        # Extract evolution of key parameters
        x_values = [res[series_name] for res in results]
        D0_values = [res['D0'] for res in results]
        D1_values = [res['D1'] for res in results]
        D2_values = [res['D2'] for res in results]
        alpha_width = [res['alpha_width'] for res in results]
        degree_mf = [res['degree_multifractality'] for res in results]
        
        # Plot generalized dimensions evolution
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, D0_values, 'bo-', label='D(0) - Capacity dimension')
        plt.plot(x_values, D1_values, 'ro-', label='D(1) - Information dimension')
        plt.plot(x_values, D2_values, 'go-', label='D(2) - Correlation dimension')
        plt.xlabel(x_label)
        plt.ylabel('Generalized Dimensions')
        plt.title(f'Evolution of Generalized Dimensions with {x_label}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "dimensions_evolution.png"), dpi=300)
        plt.close()
        
        # Plot multifractal parameters evolution
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, alpha_width, 'ms-', label='α width')
        plt.plot(x_values, degree_mf, 'cd-', label='Degree of multifractality')
        plt.xlabel(x_label)
        plt.ylabel('Parameter Value')
        plt.title(f'Evolution of Multifractal Parameters with {x_label}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, "multifractal_params_evolution.png"), dpi=300)
        plt.close()
        
        # Create 3D surface plot of D(q) evolution if possible
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Prepare data for 3D plot
            X, Y = np.meshgrid(x_values, q_values)
            Z = np.zeros((len(q_values), len(x_values)))
            
            for i, result in enumerate(results):
                for j, q in enumerate(q_values):
                    q_idx = np.where(result['q_values'] == q)[0]
                    if len(q_idx) > 0:
                        Z[j, i] = result['Dq'][q_idx[0]]
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('q')
            ax.set_zlabel('D(q)')
            ax.set_title(f'Evolution of D(q) Spectrum with {x_label}')
            
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='D(q)')
            plt.savefig(os.path.join(output_dir, "Dq_evolution_3D.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating 3D plot: {str(e)}")
    
    def _save_evolution_summary(self, results: List[Dict], series_name: str, output_dir: str):
        """Save evolution analysis summary."""
        x_values = [res[series_name] for res in results]
        D0_values = [res['D0'] for res in results]
        D1_values = [res['D1'] for res in results]
        D2_values = [res['D2'] for res in results]
        alpha_width = [res['alpha_width'] for res in results]
        degree_mf = [res['degree_multifractality'] for res in results]
        
        summary_df = pd.DataFrame({
            series_name: x_values,
            'D0': D0_values,
            'D1': D1_values,
            'D2': D2_values,
            'alpha_width': alpha_width,
            'degree_multifractality': degree_mf
        })
        summary_df.to_csv(os.path.join(output_dir, "multifractal_evolution_summary.csv"), index=False)
