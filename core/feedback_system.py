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
        """Get refined parameters based on learning from feedback."""
        records = self.collector.load_feedback_history()

        # Filter by interface type and implementation
        relevant_records = [
            r for r in records
            if r.interface_type == interface_type and r.implementation == implementation
        ]

        if not relevant_records:
            # Fall back to original heuristics
            return self._get_fallback_parameters(interface_type, features, implementation)

        # Analyze successful cases
        successful = [r for r in relevant_records if r.success]

        if not successful:
            # No successes yet - use conservative parameters
            return self._get_conservative_parameters(interface_type, features, implementation)

        # Extract parameter statistics from successful cases
        successful_params = [r.suggested_parameters for r in successful]

        refined_params = {}
        for param in ['initial_delta', 'delta_factor', 'num_steps']:
            values = [p.get(param) for p in successful_params if p.get(param) is not None]
            if values:
                # Use mean of successful values
                refined_params[param] = np.mean(values)

        return refined_params

    def _get_fallback_parameters(self, interface_type: str, features: Dict,
                               implementation: str) -> Dict:
        """Data-driven fallback parameters based on user's empirical discoveries."""

        # Use your successful manual parameters as intelligent defaults
        if interface_type == 'straight_line':
            # Your empirical success: {0.5, 1.8, 15} → 0.1% error
            return {
                'initial_delta': 0.5,
                'delta_factor': 1.8,
                'num_steps': 15
            }
        elif interface_type in ['complex_fractal', 'moderate_fractal', 'koch_curve']:
            # Your Sierpinski success: {1.0, 1.2, 15} → 2.2% error
            return {
                'initial_delta': 1.0,
                'delta_factor': 1.2,
                'num_steps': 15
            }
        else:
            # Conservative middle ground for unknown cases
            return {
                'initial_delta': 0.8,
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