#!/usr/bin/env python3
"""
Interface Feature Extraction for AI-Enhanced Parameter Selection
Extracts geometric and statistical features from segment data to "teach" the system
what different interface types look like.

Now includes feedback learning integration for adaptive parameter selection.
"""

import numpy as np
import re
from scipy import stats
from .feedback_system import AdaptiveParameterSystem


def parse_segment_file(filename):
    """Parse file with segment coordinates."""
    segments = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
            if len(numbers) >= 4:
                x1, y1, x2, y2 = map(float, numbers[:4])
                segments.append([x1, y1, x2, y2])
    return np.array(segments)


def compute_segment_lengths(segments):
    """Compute length of each segment."""
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    return np.sqrt(dx**2 + dy**2)


def compute_segment_angles(segments):
    """Compute angle of each segment relative to horizontal."""
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    return np.arctan2(dy, dx)


def compute_bounding_box(segments):
    """Compute bounding box of all segments."""
    all_x = np.concatenate([segments[:, 0], segments[:, 2]])
    all_y = np.concatenate([segments[:, 1], segments[:, 3]])
    return {
        'min_x': np.min(all_x),
        'max_x': np.max(all_x),
        'min_y': np.min(all_y),
        'max_y': np.max(all_y),
        'width': np.max(all_x) - np.min(all_x),
        'height': np.max(all_y) - np.min(all_y)
    }


def estimate_interface_roughness(segments):
    """Estimate interface roughness/complexity."""
    angles = compute_segment_angles(segments)

    # Measure angular variation
    angle_std = np.std(angles)
    angle_range = np.max(angles) - np.min(angles)

    # Measure direction changes (how often direction changes)
    angle_diffs = np.diff(angles)
    # Handle angle wraparound
    angle_diffs = np.abs(angle_diffs)
    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)

    direction_changes = np.sum(angle_diffs > np.pi/6)  # Changes > 30 degrees
    direction_change_rate = direction_changes / len(segments)

    return {
        'angle_std': angle_std,
        'angle_range': angle_range,
        'direction_changes': direction_changes,
        'direction_change_rate': direction_change_rate
    }


def estimate_fractal_dimension_heuristic(segments):
    """Quick heuristic estimate of fractal dimension."""
    bbox = compute_bounding_box(segments)
    total_length = np.sum(compute_segment_lengths(segments))

    # Euclidean distance between start and end
    start_point = segments[0, :2]
    end_point = segments[-1, 2:]
    euclidean_length = np.sqrt(np.sum((end_point - start_point)**2))

    # If interface is nearly straight, euclidean might be ~ total length
    if euclidean_length < 1e-6:
        euclidean_length = bbox['width']  # Use bounding box width instead

    # Tortuosity: ratio of actual length to straight-line length
    tortuosity = total_length / max(euclidean_length, 1e-6)

    # Heuristic dimension estimate (1 = straight line, >1 = fractal)
    dimension_estimate = 1.0 + min(np.log(tortuosity) / np.log(2), 1.0)

    return {
        'tortuosity': tortuosity,
        'dimension_estimate': dimension_estimate,
        'total_length': total_length,
        'euclidean_length': euclidean_length
    }


def compute_straightness_metrics(segments):
    """Compute metrics related to how straight the interface is."""
    bbox = compute_bounding_box(segments)
    lengths = compute_segment_lengths(segments)

    # Length uniformity
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)
    length_cv = length_std / (length_mean + 1e-10)  # Coefficient of variation

    # Aspect ratio of bounding box
    aspect_ratio = bbox['height'] / max(bbox['width'], 1e-10)

    # Linear fit quality to all endpoints
    all_x = np.concatenate([segments[:, 0], segments[:, 2]])
    all_y = np.concatenate([segments[:, 1], segments[:, 3]])

    if len(np.unique(all_x)) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_y)
        linearity_r_squared = r_value**2
    else:
        linearity_r_squared = 1.0  # Vertical line

    return {
        'length_mean': length_mean,
        'length_std': length_std,
        'length_cv': length_cv,
        'aspect_ratio': aspect_ratio,
        'linearity_r_squared': linearity_r_squared
    }


def compute_connectivity(segments):
    """
    Compute connectivity ratio - what fraction of segments are connected end-to-end.
    Returns value between 0 (completely disconnected) and 1 (fully connected curve).
    """
    n = len(segments)
    if n <= 1:
        return 1.0  # Single segment is "connected" by definition

    connected_count = 0
    for i in range(n - 1):
        end_point = segments[i, 2:4]
        next_start = segments[i+1, 0:2]
        distance = np.sqrt(np.sum((end_point - next_start)**2))

        # Consider connected if distance is very small (tolerance for floating point)
        if distance < 0.001:
            connected_count += 1

    connectivity_ratio = connected_count / (n - 1)
    return connectivity_ratio


def extract_interface_features(segments):
    """
    Extract comprehensive features from interface segments.
    These features will be used to train AI models for parameter prediction.
    """

    n_segments = len(segments)
    bbox = compute_bounding_box(segments)
    roughness = estimate_interface_roughness(segments)
    fractal_props = estimate_fractal_dimension_heuristic(segments)
    straightness = compute_straightness_metrics(segments)
    connectivity = compute_connectivity(segments)

    # Spatial scale features
    characteristic_length = np.sqrt(bbox['width'] * bbox['height'])
    min_feature_size = np.min(compute_segment_lengths(segments))
    max_feature_size = np.max(compute_segment_lengths(segments))

    return {
        # Basic geometry
        'n_segments': n_segments,
        'bbox_width': bbox['width'],
        'bbox_height': bbox['height'],
        'bbox_area': bbox['width'] * bbox['height'],
        'characteristic_length': characteristic_length,

        # Segment properties
        'min_segment_length': min_feature_size,
        'max_segment_length': max_feature_size,
        'mean_segment_length': straightness['length_mean'],
        'segment_length_std': straightness['length_std'],
        'segment_length_cv': straightness['length_cv'],

        # Complexity/roughness
        'angle_std': roughness['angle_std'],
        'angle_range': roughness['angle_range'],
        'direction_changes': roughness['direction_changes'],
        'direction_change_rate': roughness['direction_change_rate'],

        # Fractal properties
        'tortuosity': fractal_props['tortuosity'],
        'dimension_estimate': fractal_props['dimension_estimate'],
        'total_length': fractal_props['total_length'],
        'euclidean_length': fractal_props['euclidean_length'],

        # Straightness
        'aspect_ratio': straightness['aspect_ratio'],
        'linearity_r_squared': straightness['linearity_r_squared'],

        # Derived features for parameter prediction
        'complexity_score': roughness['direction_change_rate'] * roughness['angle_std'],
        'scale_ratio': max_feature_size / (min_feature_size + 1e-10),
        'density': n_segments / (bbox['width'] * bbox['height'] + 1e-10),

        # Connectivity
        'connectivity_ratio': connectivity
    }


def classify_interface_type(features):
    """
    Classify interface type based on features.
    Enhanced to detect:
    - High-turning fractals like Dragon curves
    - Disconnected fracture networks
    """

    # NEW: Detect fracture networks first (disconnected segments)
    # Key insight: fracture networks have very low connectivity
    if features['connectivity_ratio'] < 0.1 and features['n_segments'] > 20:
        return 'fracture_network'  # Disconnected segments (geologic fractures, etc)

    # Special case: perfectly straight line (horizontal or vertical)
    if (features['segment_length_cv'] < 0.001 and
        features['direction_change_rate'] == 0.0 and
        features['angle_std'] < 0.001):
        return 'straight_line'

    # High linearity and low variation = straight line
    if features['linearity_r_squared'] > 0.99 and features['direction_change_rate'] < 0.1:
        return 'straight_line'

    # ENHANCED: Dragon curve detection - extreme turning fractals
    # Key finding: Dragon curves have >90% directional change rate
    elif (features['direction_change_rate'] > 0.85 and
          features['tortuosity'] > 3.0 and  # Lower tortuosity threshold for L3
          features['complexity_score'] > 1.5):
        return 'extreme_turning_fractal'  # Dragon curves, complex spirals

    # High-turning fractals (less extreme than Dragon curves)
    elif (features['direction_change_rate'] > 0.7 and
          features['tortuosity'] > 3.0):
        return 'high_turning_fractal'

    # Koch curve characteristics: many direction changes, high tortuosity
    elif features['direction_change_rate'] > 0.5 and features['tortuosity'] > 1.2:
        return 'koch_curve'

    # Complex fractal: high complexity score
    elif features['complexity_score'] > 0.3:
        return 'complex_fractal'

    # Moderate fractal: some tortuosity
    elif features['tortuosity'] > 1.1:
        return 'moderate_fractal'

    else:
        return 'simple_curve'


def suggest_optimal_parameters(features, enable_rotation=None):
    """
    Suggest optimal parameters based on your empirical findings.
    This is the "pre-AI" heuristic version that will later be replaced by ML models.

    Args:
        features: Interface feature dictionary
        enable_rotation: None (auto-detect), True/False (force enable/disable), or list of angles
    """

    interface_type = classify_interface_type(features)

    # Determine rotation optimization strategy
    def get_rotation_parameters(interface_type, enable_rotation):
        """Get rotation angles and settings based on interface type and user preference."""
        import numpy as np

        if enable_rotation is False:
            return [], False
        elif isinstance(enable_rotation, list):
            return enable_rotation, True
        elif enable_rotation is True:
            # Force rotation for all types
            return list(np.arange(0, 91, 5)), True
        else:
            # Auto-detect based on interface type
            if interface_type in ['extreme_turning_fractal', 'high_turning_fractal']:
                # High-turning fractals benefit most from rotation optimization
                return list(np.arange(0, 91, 5)), True
            elif interface_type in ['koch_curve', 'complex_fractal']:
                # Moderate benefit - test key angles
                return [0, 15, 30, 45, 60, 75, 90], True
            elif interface_type == 'straight_line':
                # Straight lines typically don't benefit from rotation
                return [0], False
            else:
                # Default: test basic angles
                return [0, 30, 45, 60, 90], True

    rotation_angles, rotation_enabled = get_rotation_parameters(interface_type, enable_rotation)

    if interface_type == 'fracture_network':
        # Disconnected fracture networks: need fine resolution relative to segment size
        # Use mean segment length as reference, not domain size
        # Note: Use conservative parameters to avoid excessive computation with many segments
        return {
            'initial_delta': features['mean_segment_length'] * 3.0,  # Start at ~3x segment size
            'delta_factor': 2.0,  # Larger scaling to avoid tiny boxes
            'num_steps': 10,      # Limited steps for computational efficiency
            'confidence': 'medium',
            'reason': 'fracture network - balanced resolution for disconnected segments',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    elif interface_type == 'straight_line':
        # Your empirical findings for straight lines
        return {
            'initial_delta': features['mean_segment_length'] * 8,  # 8x feature size
            'delta_factor': 1.8,
            'num_steps': 25,
            'confidence': 'high',
            'reason': 'straight line - use fine resolution',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    elif interface_type == 'extreme_turning_fractal':
        # Dragon curves: ultra-fine resolution needed for 96% directional change rate
        return {
            'initial_delta': features['characteristic_length'] * 0.03,  # 3x finer than Koch
            'delta_factor': 1.3,  # Smaller steps to capture fine structure
            'num_steps': 35,      # Extended range for complex scaling
            'confidence': 'high',
            'reason': 'extreme turning fractal (Dragon curve) - ultra-fine resolution',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    elif interface_type == 'high_turning_fractal':
        # Complex turning fractals (less extreme than Dragon curves)
        return {
            'initial_delta': features['characteristic_length'] * 0.05,  # 2x finer than Koch
            'delta_factor': 1.4,  # Smaller steps
            'num_steps': 30,      # Extended range
            'confidence': 'high',
            'reason': 'high turning fractal - fine resolution for complex structure',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    elif interface_type == 'koch_curve':
        # Koch curves need good coverage of scaling range
        return {
            'initial_delta': features['characteristic_length'] * 0.1,
            'delta_factor': 1.8,
            'num_steps': 30,
            'confidence': 'high',
            'reason': 'koch curve - extended range needed',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    elif interface_type in ['complex_fractal', 'moderate_fractal']:
        # RT interfaces and other complex fractals
        return {
            'initial_delta': features['characteristic_length'] * 0.2,
            'delta_factor': 1.8,
            'num_steps': 20,
            'confidence': 'medium',
            'reason': 'complex interface - balanced approach',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }

    else:
        # Default parameters
        return {
            'initial_delta': features['characteristic_length'] * 0.3,
            'delta_factor': 1.5,
            'num_steps': 15,
            'confidence': 'low',
            'reason': 'unknown interface type - conservative parameters',
            'rotation_angles': rotation_angles,
            'rotation_enabled': rotation_enabled
        }


# Global adaptive system instance
_adaptive_system = None

def get_adaptive_system():
    """Get the global adaptive parameter system."""
    global _adaptive_system
    if _adaptive_system is None:
        _adaptive_system = AdaptiveParameterSystem()
    return _adaptive_system


def suggest_optimal_parameters_adaptive(features, implementation="basic_box_counting", enable_rotation=None):
    """
    Suggest optimal parameters using adaptive learning from feedback.
    This replaces the original heuristic approach with feedback-based learning.

    Args:
        features: Interface feature dictionary
        implementation: Box counting implementation type
        enable_rotation: None (auto-detect), True/False (force enable/disable), or list of angles
    """
    interface_type = classify_interface_type(features)
    adaptive_system = get_adaptive_system()

    # Get rotation parameters using same logic as heuristic function
    def get_rotation_parameters(interface_type, enable_rotation):
        """Get rotation angles and settings based on interface type and user preference."""
        import numpy as np

        if enable_rotation is False:
            return [], False
        elif isinstance(enable_rotation, list):
            return enable_rotation, True
        elif enable_rotation is True:
            # Force rotation for all types
            return list(np.arange(0, 91, 5)), True
        else:
            # Auto-detect based on interface type
            if interface_type in ['extreme_turning_fractal', 'high_turning_fractal']:
                # High-turning fractals benefit most from rotation optimization
                return list(np.arange(0, 91, 5)), True
            elif interface_type in ['koch_curve', 'complex_fractal']:
                # Moderate benefit - test key angles
                return [0, 15, 30, 45, 60, 75, 90], True
            elif interface_type == 'straight_line':
                # Straight lines typically don't benefit from rotation
                return [0], False
            else:
                # Default: test basic angles
                return [0, 30, 45, 60, 90], True

    rotation_angles, rotation_enabled = get_rotation_parameters(interface_type, enable_rotation)

    # Get learned parameters
    learned_params = adaptive_system.suggest_parameters_with_learning(
        interface_type, features, implementation
    )

    # Add metadata and rotation parameters
    learned_params.update({
        'interface_type': interface_type,
        'implementation': implementation,
        'source': 'adaptive_learning',
        'rotation_angles': rotation_angles,
        'rotation_enabled': rotation_enabled
    })

    return learned_params


def record_parameter_feedback(features, suggested_parameters, dimension_result,
                            theoretical_dimension=None, r_squared=None,
                            computation_time=None, implementation="basic_box_counting"):
    """
    Record feedback from parameter usage for learning.
    Call this after running dimension calculation with suggested parameters.
    """
    interface_type = classify_interface_type(features)
    adaptive_system = get_adaptive_system()

    return adaptive_system.record_result(
        interface_type, features, suggested_parameters, dimension_result,
        theoretical_dimension, r_squared, computation_time, implementation
    )


def get_learning_statistics():
    """Get current learning progress and statistics."""
    adaptive_system = get_adaptive_system()
    return adaptive_system.get_learning_status()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python interface_features.py <segment_file>")
        sys.exit(1)

    filename = sys.argv[1]

    print("="*60)
    print("INTERFACE FEATURE EXTRACTION")
    print("="*60)
    print(f"File: {filename}")

    try:
        segments = parse_segment_file(filename)
        print(f"Loaded {len(segments)} segments")

        features = extract_interface_features(segments)
        interface_type = classify_interface_type(features)
        suggested_params = suggest_optimal_parameters(features)

        print(f"\nInterface Type: {interface_type}")
        print("\nFeatures:")
        print("-" * 40)
        for key, value in features.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:12.6f}")
            else:
                print(f"{key:25s}: {value:12}")

        print(f"\nSuggested Parameters:")
        print("-" * 40)
        for key, value in suggested_params.items():
            print(f"{key:15s}: {value}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)