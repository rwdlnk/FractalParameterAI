#!/usr/bin/env python3
"""
AI Parameter Learning Feedback System

Implements feedback loops to improve parameter selection based on actual results:
- Collects (features, parameters, outcomes) data
- Analyzes success/failure patterns
- Refines parameter suggestions over time
- Builds training datasets for ML models
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class FeedbackRecord:
    """Single feedback record for parameter learning."""
    timestamp: str
    interface_type: str
    features: Dict
    suggested_parameters: Dict
    dimension_result: Optional[float]
    theoretical_dimension: Optional[float]
    accuracy_score: Optional[float]
    r_squared: Optional[float]
    computation_time: Optional[float]
    success: bool
    failure_mode: Optional[str]
    implementation: str  # 'basic_box_counting', 'wu_methodology', etc.


class FeedbackCollector:
    """Collects and stores feedback data for AI learning."""

    def __init__(self, data_file="feedback_data.jsonl"):
        self.data_file = data_file
        self.ensure_data_file()

    def ensure_data_file(self):
        """Create data file if it doesn't exist."""
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                pass  # Create empty file

    def record_feedback(self,
                       interface_type: str,
                       features: Dict,
                       suggested_parameters: Dict,
                       dimension_result: Optional[float],
                       theoretical_dimension: Optional[float] = None,
                       r_squared: Optional[float] = None,
                       computation_time: Optional[float] = None,
                       implementation: str = "basic_box_counting") -> FeedbackRecord:
        """Record a single feedback instance."""

        # Calculate accuracy score
        accuracy_score = None
        success = False
        failure_mode = None

        if dimension_result is not None and not np.isnan(dimension_result):
            if theoretical_dimension is not None:
                accuracy_score = 1.0 - abs(dimension_result - theoretical_dimension) / theoretical_dimension
                success = accuracy_score > 0.95  # 95% accuracy threshold
            else:
                # No theoretical value - use other quality metrics
                success = (r_squared is not None and r_squared > 0.95)
        else:
            failure_mode = "nan_result" if dimension_result is not None else "no_result"

        # Create record
        record = FeedbackRecord(
            timestamp=datetime.now().isoformat(),
            interface_type=interface_type,
            features=features,
            suggested_parameters=suggested_parameters,
            dimension_result=dimension_result,
            theoretical_dimension=theoretical_dimension,
            accuracy_score=accuracy_score,
            r_squared=r_squared,
            computation_time=computation_time,
            success=success,
            failure_mode=failure_mode,
            implementation=implementation
        )

        # Store to file (convert numpy types to native Python types)
        record_dict = asdict(record)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        record_dict = convert_numpy_types(record_dict)

        with open(self.data_file, 'a') as f:
            f.write(json.dumps(record_dict) + '\n')

        return record

    def load_feedback_history(self) -> List[FeedbackRecord]:
        """Load all feedback records from file."""
        records = []
        try:
            with open(self.data_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        records.append(FeedbackRecord(**data))
        except FileNotFoundError:
            pass
        return records


class ParameterLearner:
    """Learns optimal parameters from feedback data."""

    def __init__(self, feedback_collector: FeedbackCollector):
        self.collector = feedback_collector

    def analyze_success_patterns(self, implementation: str = None) -> Dict:
        """Analyze what parameter patterns lead to success."""
        records = self.collector.load_feedback_history()

        if implementation:
            records = [r for r in records if r.implementation == implementation]

        success_params = []
        failure_params = []

        for record in records:
            if record.success:
                success_params.append(record.suggested_parameters)
            else:
                failure_params.append(record.suggested_parameters)

        analysis = {
            'total_records': len(records),
            'successes': len(success_params),
            'failures': len(failure_params),
            'success_rate': len(success_params) / len(records) if records else 0,
        }

        if success_params:
            # Analyze successful parameter ranges
            for param in ['initial_delta', 'delta_factor', 'num_steps']:
                values = [p.get(param) for p in success_params if p.get(param) is not None]
                if values:
                    analysis[f'successful_{param}'] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

        return analysis

    def get_refined_parameters(self, interface_type: str, features: Dict,
                             implementation: str = "basic_box_counting") -> Dict:
        """Get refined parameters based on learning from feedback with scale normalization."""
        records = self.collector.load_feedback_history()

        # Filter by interface type and implementation
        relevant_records = [
            r for r in records
            if r.interface_type == interface_type and r.implementation == implementation
        ]

        # Also filter out physically invalid results (D < 1.0 or D > 2.0 for 2D embeddings)
        relevant_records = [
            r for r in relevant_records
            if r.dimension_result is None or (1.0 <= r.dimension_result <= 2.0)
        ]

        if not relevant_records:
            # Fall back to original heuristics
            return self._get_fallback_parameters(interface_type, features, implementation)

        # Analyze successful cases
        successful = [r for r in relevant_records if r.success]

        if not successful:
            # No successes yet - use conservative parameters
            return self._get_conservative_parameters(interface_type, features, implementation)

        # Get current domain characteristics for scale normalization
        current_char_length = features.get('characteristic_length', 1.0)
        current_domain_size = min(features.get('bbox_width', 1.0), features.get('bbox_height', 1.0))

        # Extract parameter statistics from successful cases with scale normalization
        normalized_initial_deltas = []
        delta_factors = []
        num_steps_values = []

        for record in successful:
            params = record.suggested_parameters
            record_features = record.features

            # Normalize initial_delta by the characteristic_length at time of recording
            record_char_length = record_features.get('characteristic_length', 1.0)
            if record_char_length > 0:
                # Store as ratio: initial_delta / characteristic_length
                normalized_ratio = params.get('initial_delta', 0) / record_char_length
                normalized_initial_deltas.append(normalized_ratio)

            if params.get('delta_factor') is not None:
                delta_factors.append(params.get('delta_factor'))
            if params.get('num_steps') is not None:
                num_steps_values.append(params.get('num_steps'))

        refined_params = {}

        # Calculate initial_delta from normalized ratios
        if normalized_initial_deltas:
            mean_ratio = np.mean(normalized_initial_deltas)
            # Apply to current characteristic length
            suggested_initial_delta = mean_ratio * current_char_length

            # Apply safety bounds: must be less than domain size
            max_allowed_delta = current_domain_size * 0.8  # 80% of domain
            refined_params['initial_delta'] = min(suggested_initial_delta, max_allowed_delta)

        # Other parameters don't need scale normalization
        if delta_factors:
            refined_params['delta_factor'] = np.mean(delta_factors)
        if num_steps_values:
            refined_params['num_steps'] = np.mean(num_steps_values)

        return refined_params

    def _get_fallback_parameters(self, interface_type: str, features: Dict,
                               implementation: str) -> Dict:
        """Data-driven fallback parameters based on user's empirical discoveries, with scale awareness."""

        # Get domain characteristics
        char_length = features.get('characteristic_length', 1.0)
        domain_size = min(features.get('bbox_width', 1.0), features.get('bbox_height', 1.0))
        max_allowed_delta = domain_size * 0.8  # Safety bound

        # Use your successful manual parameters as intelligent defaults
        # These are expressed as ratios of characteristic_length
        if interface_type == 'fracture_network':
            # Fracture networks: use segment size, not domain size
            mean_segment = features.get('mean_segment_length', char_length * 0.1)
            initial_delta = min(mean_segment * 3.0, max_allowed_delta)
            return {
                'initial_delta': initial_delta,
                'delta_factor': 2.0,
                'num_steps': 10
            }
        elif interface_type == 'straight_line':
            # Your empirical success: used ratio ~0.5 of char_length
            initial_delta = min(0.5 * char_length, max_allowed_delta)
            return {
                'initial_delta': initial_delta,
                'delta_factor': 1.8,
                'num_steps': 15
            }
        elif interface_type in ['complex_fractal', 'moderate_fractal', 'koch_curve']:
            # Use conservative ratio to work across different domain sizes
            # Start with boxes that are 1/3 of domain to ensure good scaling range
            initial_delta = min(0.3 * char_length, max_allowed_delta)
            return {
                'initial_delta': initial_delta,
                'delta_factor': 1.5,
                'num_steps': 15
            }
        else:
            # Conservative middle ground for unknown cases
            initial_delta = min(0.8 * char_length, max_allowed_delta)
            return {
                'initial_delta': initial_delta,
                'delta_factor': 1.5,
                'num_steps': 12
            }

    def _get_conservative_parameters(self, interface_type: str, features: Dict,
                                   implementation: str) -> Dict:
        """Conservative parameters when previous attempts failed."""
        feature_size = features.get('mean_segment_length', 0.1)

        return {
            'initial_delta': min(1.0 * feature_size, 0.25),
            'delta_factor': 1.2,
            'num_steps': 8
        }


class AdaptiveParameterSystem:
    """Complete adaptive parameter system with feedback learning."""

    def __init__(self, data_file="feedback_data.jsonl"):
        self.collector = FeedbackCollector(data_file)
        self.learner = ParameterLearner(self.collector)

    def suggest_parameters_with_learning(self, interface_type: str, features: Dict,
                                       implementation: str = "basic_box_counting") -> Dict:
        """Suggest parameters using learned patterns."""
        return self.learner.get_refined_parameters(interface_type, features, implementation)

    def record_result(self, interface_type: str, features: Dict,
                     suggested_parameters: Dict, dimension_result: Optional[float],
                     theoretical_dimension: Optional[float] = None,
                     r_squared: Optional[float] = None,
                     computation_time: Optional[float] = None,
                     implementation: str = "basic_box_counting") -> FeedbackRecord:
        """Record the result of using suggested parameters."""
        return self.collector.record_feedback(
            interface_type, features, suggested_parameters, dimension_result,
            theoretical_dimension, r_squared, computation_time, implementation
        )

    def get_learning_status(self) -> Dict:
        """Get current learning status and statistics."""
        records = self.collector.load_feedback_history()

        by_implementation = {}
        for impl in set(r.implementation for r in records):
            impl_records = [r for r in records if r.implementation == impl]
            by_implementation[impl] = self.learner.analyze_success_patterns(impl)

        return {
            'total_records': len(records),
            'by_implementation': by_implementation,
            'recent_success_rate': self._calculate_recent_success_rate(records)
        }

    def _calculate_recent_success_rate(self, records: List[FeedbackRecord],
                                     last_n: int = 10) -> float:
        """Calculate success rate for most recent N records."""
        if not records:
            return 0.0

        recent = sorted(records, key=lambda r: r.timestamp)[-last_n:]
        successes = sum(1 for r in recent if r.success)
        return successes / len(recent)