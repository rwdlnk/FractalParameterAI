#!/usr/bin/env python3
"""
RT Interface Extraction

Extracts interfaces from RT simulation VTK data using various contour methods.
Currently implements scikit-image contours, with provisions for future PLIC/CONREC.

This module is part of the FractalParameterAI framework.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time

try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️  scikit-image not available - interface extraction limited")

# Import CONREC and PLIC extractors from integration directory
try:
    from integration.conrec_extractor import CONRECExtractor
    CONREC_AVAILABLE = True
except ImportError:
    CONREC_AVAILABLE = False
    print("⚠️  CONREC extractor not available")

try:
    from integration.plic_extractor import PLICExtractor
    PLIC_AVAILABLE = True
except ImportError:
    PLIC_AVAILABLE = False
    print("⚠️  PLIC extractor not available")


@dataclass
class InterfaceData:
    """Container for extracted interface data."""
    points: List[Tuple[float, float]]  # Interface points as (x, y) tuples
    segments: np.ndarray  # Interface segments as (n, 4) array [x1, y1, x2, y2]
    extraction_method: str  # Method used for extraction
    extraction_time: float  # Time taken for extraction (seconds)
    metadata: Dict  # Additional metadata


class InterfaceExtractor:
    """
    Extracts RT interfaces from VTK scalar field data.

    Methods:
    - scikit-image contours (default)
    - PLIC (future)
    - CONREC (future)
    """

    def __init__(self, method: str = "skimage", debug: bool = False):
        """
        Initialize interface extractor.

        Args:
            method: Extraction method ("skimage", "plic", "conrec")
            debug: Enable debug output
        """
        self.method = method
        self.debug = debug

        # Check method availability
        if method == "skimage" and not SKIMAGE_AVAILABLE:
            raise RuntimeError("scikit-image not available but required for skimage method")
        elif method == "plic" and not PLIC_AVAILABLE:
            raise RuntimeError("PLIC extractor not available but required for plic method")
        elif method == "conrec" and not CONREC_AVAILABLE:
            raise RuntimeError("CONREC extractor not available but required for conrec method")

        # Initialize method-specific extractors
        self.plic_extractor = None
        self.conrec_extractor = None

        if method == "plic" and PLIC_AVAILABLE:
            self.plic_extractor = PLICExtractor(debug=debug)
        elif method == "conrec" and CONREC_AVAILABLE:
            self.conrec_extractor = CONRECExtractor(debug=debug)

    def extract_interface(self, f_grid: np.ndarray, x_grid: np.ndarray,
                         y_grid: np.ndarray, level: float = 0.5) -> Optional[InterfaceData]:
        """
        Extract interface from VOF field at specified level.

        Args:
            f_grid: Volume fraction field (2D array)
            x_grid: X-coordinate grid (2D array)
            y_grid: Y-coordinate grid (2D array)
            level: Contour level (default 0.5 for interface)

        Returns:
            InterfaceData object or None on failure
        """
        start_time = time.time()

        if self.method == "skimage":
            result = self._extract_skimage(f_grid, x_grid, y_grid, level)
        elif self.method == "plic":
            result = self._extract_plic(f_grid, x_grid, y_grid, level)
        elif self.method == "conrec":
            result = self._extract_conrec(f_grid, x_grid, y_grid, level)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")

        if result is not None:
            extraction_time = time.time() - start_time
            result.extraction_time = extraction_time
            result.extraction_method = self.method

        return result

    def _extract_skimage(self, f_grid: np.ndarray, x_grid: np.ndarray,
                        y_grid: np.ndarray, level: float) -> Optional[InterfaceData]:
        """
        Extract interface using scikit-image marching squares.

        Args:
            f_grid: Volume fraction field
            x_grid: X-coordinate grid
            y_grid: Y-coordinate grid
            level: Contour level

        Returns:
            InterfaceData or None
        """
        if not SKIMAGE_AVAILABLE:
            return None

        try:
            # Find contours at specified level
            contours = measure.find_contours(f_grid.T, level)

            if not contours:
                if self.debug:
                    print(f"   ⚠️  No contours found at level {level}")
                return None

            # Take the longest contour (main interface)
            main_contour = max(contours, key=len)

            if self.debug:
                print(f"   Found {len(contours)} contours, using longest ({len(main_contour)} points)")

            # Convert contour indices to physical coordinates
            # Contour indices are in (j, i) format where j=y-index, i=x-index
            interface_points = []
            for j, i in main_contour:
                # Convert float indices to integer indices for grid lookup
                i_int = int(np.floor(i))
                j_int = int(np.floor(j))

                # Bounds checking
                i_int = max(0, min(i_int, x_grid.shape[0] - 1))
                j_int = max(0, min(j_int, x_grid.shape[1] - 1))

                # Get physical coordinates
                x = x_grid[i_int, j_int]
                y = y_grid[i_int, j_int]
                interface_points.append((x, y))

            # Convert to segments
            segments = self._points_to_segments(interface_points)

            # Create metadata
            metadata = {
                'n_contours': len(contours),
                'n_points': len(interface_points),
                'n_segments': len(segments),
                'level': level,
                'bounds': self._compute_bounds(interface_points)
            }

            return InterfaceData(
                points=interface_points,
                segments=segments,
                extraction_method="skimage",
                extraction_time=0.0,  # Will be set by caller
                metadata=metadata
            )

        except Exception as e:
            if self.debug:
                print(f"   ❌ scikit-image extraction failed: {e}")
            return None

    def _extract_plic(self, f_grid: np.ndarray, x_grid: np.ndarray,
                     y_grid: np.ndarray, level: float) -> Optional[InterfaceData]:
        """
        Extract interface using PLIC (Piecewise Linear Interface Calculation).

        Args:
            f_grid: Volume fraction field
            x_grid: X-coordinate grid
            y_grid: Y-coordinate grid
            level: Contour level (ignored for PLIC, uses VOF directly)

        Returns:
            InterfaceData or None
        """
        if not PLIC_AVAILABLE or self.plic_extractor is None:
            return None

        try:
            # PLIC doesn't use a contour level - it reconstructs based on VOF directly
            segment_list = self.plic_extractor.extract_interface_plic(f_grid, x_grid, y_grid)

            if not segment_list:
                if self.debug:
                    print(f"   ⚠️  No segments extracted by PLIC")
                return None

            # Convert segment list format to our format
            # PLIC returns: [((x1, y1), (x2, y2)), ...]
            # We need: array of [x1, y1, x2, y2]
            segments = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in segment_list])

            # Convert segments to points (for metadata)
            interface_points = []
            for (x1, y1), (x2, y2) in segment_list:
                interface_points.append((x1, y1))
            if segment_list:
                interface_points.append(segment_list[-1][1])  # Add last endpoint

            # Create metadata
            metadata = {
                'n_points': len(interface_points),
                'n_segments': len(segments),
                'level': 'VOF-based',
                'bounds': self._compute_bounds(interface_points)
            }

            return InterfaceData(
                points=interface_points,
                segments=segments,
                extraction_method="plic",
                extraction_time=0.0,  # Will be set by caller
                metadata=metadata
            )

        except Exception as e:
            if self.debug:
                print(f"   ❌ PLIC extraction failed: {e}")
            return None

    def _extract_conrec(self, f_grid: np.ndarray, x_grid: np.ndarray,
                       y_grid: np.ndarray, level: float) -> Optional[InterfaceData]:
        """
        Extract interface using CONREC contouring algorithm.

        Args:
            f_grid: Volume fraction field
            x_grid: X-coordinate grid
            y_grid: Y-coordinate grid
            level: Contour level

        Returns:
            InterfaceData or None
        """
        if not CONREC_AVAILABLE or self.conrec_extractor is None:
            return None

        try:
            # CONREC returns segment list format
            segment_list = self.conrec_extractor.extract_interface_conrec(
                f_grid, x_grid, y_grid, level
            )

            if not segment_list:
                if self.debug:
                    print(f"   ⚠️  No segments extracted by CONREC")
                return None

            # Convert segment list format to our format
            segments = np.array([[x1, y1, x2, y2] for (x1, y1), (x2, y2) in segment_list])

            # Convert segments to points (for metadata)
            interface_points = []
            for (x1, y1), (x2, y2) in segment_list:
                interface_points.append((x1, y1))
            if segment_list:
                interface_points.append(segment_list[-1][1])  # Add last endpoint

            # Create metadata
            metadata = {
                'n_points': len(interface_points),
                'n_segments': len(segments),
                'level': level,
                'bounds': self._compute_bounds(interface_points)
            }

            return InterfaceData(
                points=interface_points,
                segments=segments,
                extraction_method="conrec",
                extraction_time=0.0,  # Will be set by caller
                metadata=metadata
            )

        except Exception as e:
            if self.debug:
                print(f"   ❌ CONREC extraction failed: {e}")
            return None

    def _points_to_segments(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Convert list of points to segment array.

        Args:
            points: List of (x, y) tuples

        Returns:
            numpy array of shape (n_segments, 4) with [x1, y1, x2, y2]
        """
        if len(points) < 2:
            return np.array([])

        segments = []
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            segments.append([x1, y1, x2, y2])

        return np.array(segments)

    def _compute_bounds(self, points: List[Tuple[float, float]]) -> Dict:
        """Compute bounding box of interface points."""
        if not points:
            return {'x_min': 0, 'x_max': 0, 'y_min': 0, 'y_max': 0,
                   'width': 0, 'height': 0}

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }


def extract_multiple_levels(f_grid: np.ndarray, x_grid: np.ndarray,
                           y_grid: np.ndarray,
                           levels: List[float] = [0.05, 0.5, 0.95],
                           method: str = "skimage",
                           debug: bool = False) -> Dict[str, InterfaceData]:
    """
    Extract interfaces at multiple VOF levels.

    Useful for mixing zone analysis:
    - 0.05: Lower boundary of mixing zone
    - 0.5: Main interface
    - 0.95: Upper boundary of mixing zone

    Args:
        f_grid: Volume fraction field
        x_grid: X-coordinate grid
        y_grid: Y-coordinate grid
        levels: List of contour levels
        method: Extraction method
        debug: Enable debug output

    Returns:
        Dictionary mapping level names to InterfaceData
    """
    level_names = {
        0.05: 'lower_boundary',
        0.5: 'interface',
        0.95: 'upper_boundary'
    }

    extractor = InterfaceExtractor(method=method, debug=debug)
    results = {}

    for level in levels:
        level_name = level_names.get(level, f'level_{level:.2f}')

        if debug:
            print(f"   Extracting {level_name} (F={level:.2f})...")

        interface_data = extractor.extract_interface(f_grid, x_grid, y_grid, level)

        if interface_data:
            results[level_name] = interface_data
            if debug:
                print(f"      ✅ {interface_data.metadata['n_points']} points")
        else:
            if debug:
                print(f"      ❌ Extraction failed")

    return results


def main():
    """Test interface extraction."""
    import argparse
    import sys
    import os

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from integration.rt_vtk_parser import VTKParser

    parser = argparse.ArgumentParser(description='Test interface extraction')
    parser.add_argument('vtk_file', help='Path to VTK file')
    parser.add_argument('--levels', nargs='+', type=float, default=[0.5],
                       help='Contour levels to extract')
    parser.add_argument('--method', choices=['skimage'], default='skimage',
                       help='Extraction method')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()

    # Parse VTK file
    print(f"📂 Loading VTK file...")
    vtk_parser = VTKParser(debug=args.debug)
    vtk_data = vtk_parser.parse_vtk_file(args.vtk_file)

    if not vtk_data:
        print(f"❌ Failed to parse VTK file")
        return

    if 'f' not in vtk_data.scalar_fields and 'F' not in vtk_data.scalar_fields:
        print(f"❌ No VOF field found in VTK file")
        return

    f_grid = vtk_data['f']
    x_grid = vtk_data.x_grid
    y_grid = vtk_data.y_grid

    # Extract interfaces
    print(f"\n🔍 Extracting interfaces...")
    extractor = InterfaceExtractor(method=args.method, debug=args.debug)

    for level in args.levels:
        print(f"\n   Level {level:.2f}:")
        interface_data = extractor.extract_interface(f_grid, x_grid, y_grid, level)

        if interface_data:
            print(f"      ✅ Extracted {interface_data.metadata['n_points']} points")
            print(f"      ⏱️  Extraction time: {interface_data.extraction_time:.3f}s")
            bounds = interface_data.metadata['bounds']
            print(f"      📏 Bounds: x=[{bounds['x_min']:.4f}, {bounds['x_max']:.4f}], "
                  f"y=[{bounds['y_min']:.4f}, {bounds['y_max']:.4f}]")
            print(f"      📐 Size: {bounds['width']:.4f} × {bounds['height']:.4f}")
        else:
            print(f"      ❌ Extraction failed")


if __name__ == "__main__":
    main()
