"""Report generation with time-range consolidation."""

import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

from models.emotion_analyzer import EmotionResult
from models.action_classifier import ActionResult, ActionType
from models.anomaly_detector import Anomaly, AnomalyType
from config import AnalysisConfig


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


@dataclass
class TimeRange:
    """Represents a time range for an observation."""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    
    def to_dict(self) -> dict:
        return {
            'start_time': round(self.start_time, 2),
            'end_time': round(self.end_time, 2),
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'duration': round(self.end_time - self.start_time, 2)
        }


@dataclass
class Observation:
    """Represents a consolidated observation."""
    category: str  # 'positive', 'negative', 'neutral'
    observation_type: str  # 'emotion', 'action'
    value: str  # The emotion or action name
    time_range: TimeRange
    avg_confidence: float
    occurrences: int
    
    def to_dict(self) -> dict:
        return {
            'category': self.category,
            'type': self.observation_type,
            'value': self.value,
            'time_range': self.time_range.to_dict(),
            'average_confidence': round(self.avg_confidence, 2),
            'occurrences': self.occurrences
        }


@dataclass
class AnomalyReport:
    """Represents an anomaly in the report."""
    anomaly_type: str
    severity: float
    time_range: TimeRange
    description: str
    count: int
    
    def to_dict(self) -> dict:
        return {
            'type': self.anomaly_type,
            'severity': round(self.severity, 2),
            'time_range': self.time_range.to_dict(),
            'description': self.description,
            'count': self.count
        }


class ReportGenerator:
    """Generates consolidated analysis reports."""
    
    def __init__(self, config: AnalysisConfig, consolidation_window: float = 0.5):
        """
        Initialize report generator.
        
        Args:
            config: Analysis configuration
            consolidation_window: Time window (seconds) to consolidate same observations
        """
        self.config = config
        self.consolidation_window = consolidation_window
        
        # Raw observation storage
        self._emotion_observations: List[tuple] = []  # (frame, timestamp, emotion, confidence)
        self._action_observations: List[tuple] = []   # (frame, timestamp, action, confidence)
        self._anomalies: List[Anomaly] = []
        
        # Video metadata
        self.video_path: str = ""
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        
    def set_video_info(self, video_path: str, fps: float, total_frames: int):
        """Set video metadata."""
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self.duration = total_frames / fps if fps > 0 else 0
        
    def add_emotion(self, frame: int, timestamp: float, emotion: str, confidence: float):
        """Add an emotion observation."""
        self._emotion_observations.append((frame, timestamp, emotion, confidence))
        
    def add_action(self, frame: int, timestamp: float, action: str, confidence: float):
        """Add an action observation."""
        if action != ActionType.UNKNOWN.value:
            self._action_observations.append((frame, timestamp, action, confidence))
    
    def add_anomaly(self, anomaly: Anomaly):
        """Add an anomaly."""
        self._anomalies.append(anomaly)
        
    def _categorize_emotion(self, emotion: str) -> str:
        """Categorize an emotion as positive, negative, or neutral."""
        if emotion in self.config.positive_emotions:
            return 'positive'
        elif emotion in self.config.negative_emotions:
            return 'negative'
        return 'neutral'
    
    def _categorize_action(self, action: str) -> str:
        """Categorize an action as positive, negative, or neutral."""
        if action in self.config.positive_actions:
            return 'positive'
        elif action in self.config.negative_actions:
            return 'negative'
        return 'neutral'
    
    def _consolidate_observations(self, observations: List[tuple], 
                                   obs_type: str,
                                   categorize_func) -> List[Observation]:
        """
        Consolidate observations into time ranges.
        
        Args:
            observations: List of (frame, timestamp, value, confidence) tuples
            obs_type: 'emotion' or 'action'
            categorize_func: Function to categorize values
            
        Returns:
            List of consolidated Observation objects
        """
        if not observations:
            return []
        
        # Sort by timestamp
        sorted_obs = sorted(observations, key=lambda x: x[1])
        
        consolidated = []
        current_value = None
        current_start_frame = 0
        current_start_time = 0.0
        current_confidences = []
        current_count = 0
        last_timestamp = 0.0
        last_frame = 0
        
        for frame, timestamp, value, confidence in sorted_obs:
            # Check if this continues the current observation
            if (current_value == value and 
                timestamp - last_timestamp <= self.consolidation_window):
                # Continue current observation
                current_confidences.append(confidence)
                current_count += 1
                last_timestamp = timestamp
                last_frame = frame
            else:
                # Save previous observation if exists
                if current_value is not None:
                    consolidated.append(Observation(
                        category=categorize_func(current_value),
                        observation_type=obs_type,
                        value=current_value,
                        time_range=TimeRange(
                            start_time=current_start_time,
                            end_time=last_timestamp,
                            start_frame=current_start_frame,
                            end_frame=last_frame
                        ),
                        avg_confidence=sum(current_confidences) / len(current_confidences),
                        occurrences=current_count
                    ))
                
                # Start new observation
                current_value = value
                current_start_frame = frame
                current_start_time = timestamp
                current_confidences = [confidence]
                current_count = 1
                last_timestamp = timestamp
                last_frame = frame
        
        # Don't forget the last observation
        if current_value is not None:
            consolidated.append(Observation(
                category=categorize_func(current_value),
                observation_type=obs_type,
                value=current_value,
                time_range=TimeRange(
                    start_time=current_start_time,
                    end_time=last_timestamp,
                    start_frame=current_start_frame,
                    end_frame=last_frame
                ),
                avg_confidence=sum(current_confidences) / len(current_confidences),
                occurrences=current_count
            ))
        
        return consolidated
    
    def _consolidate_anomalies(self) -> List[AnomalyReport]:
        """Consolidate anomalies into time ranges."""
        if not self._anomalies:
            return []
        
        # Group by type
        by_type: Dict[AnomalyType, List[Anomaly]] = defaultdict(list)
        for anomaly in self._anomalies:
            by_type[anomaly.anomaly_type].append(anomaly)
        
        consolidated = []
        
        for anomaly_type, anomalies in by_type.items():
            # Sort by timestamp
            sorted_anomalies = sorted(anomalies, key=lambda x: x.timestamp)
            
            current_start = sorted_anomalies[0]
            current_severities = [current_start.severity]
            current_count = 1
            last_anomaly = current_start
            
            for anomaly in sorted_anomalies[1:]:
                if anomaly.timestamp - last_anomaly.timestamp <= self.consolidation_window * 2:
                    # Continue current group
                    current_severities.append(anomaly.severity)
                    current_count += 1
                    last_anomaly = anomaly
                else:
                    # Save and start new group
                    consolidated.append(AnomalyReport(
                        anomaly_type=current_start.anomaly_type.value,
                        severity=sum(current_severities) / len(current_severities),
                        time_range=TimeRange(
                            start_time=current_start.timestamp,
                            end_time=last_anomaly.timestamp,
                            start_frame=current_start.frame_number,
                            end_frame=last_anomaly.frame_number
                        ),
                        description=current_start.description,
                        count=current_count
                    ))
                    
                    current_start = anomaly
                    current_severities = [anomaly.severity]
                    current_count = 1
                    last_anomaly = anomaly
            
            # Don't forget the last group
            consolidated.append(AnomalyReport(
                anomaly_type=current_start.anomaly_type.value,
                severity=sum(current_severities) / len(current_severities),
                time_range=TimeRange(
                    start_time=current_start.timestamp,
                    end_time=last_anomaly.timestamp,
                    start_frame=current_start.frame_number,
                    end_frame=last_anomaly.frame_number
                ),
                description=current_start.description,
                count=current_count
            ))
        
        return sorted(consolidated, key=lambda x: x.time_range.start_time)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate the final analysis report.
        
        Returns:
            Report dictionary
        """
        # Consolidate observations
        emotion_observations = self._consolidate_observations(
            self._emotion_observations, 'emotion', self._categorize_emotion
        )
        action_observations = self._consolidate_observations(
            self._action_observations, 'action', self._categorize_action
        )
        
        # Combine and categorize
        all_observations = emotion_observations + action_observations
        
        positive = [obs.to_dict() for obs in all_observations if obs.category == 'positive']
        negative = [obs.to_dict() for obs in all_observations if obs.category == 'negative']
        neutral = [obs.to_dict() for obs in all_observations if obs.category == 'neutral']
        
        # Consolidate anomalies
        anomaly_reports = self._consolidate_anomalies()
        
        # Calculate statistics
        emotion_counts = defaultdict(int)
        for _, _, emotion, _ in self._emotion_observations:
            emotion_counts[emotion] += 1
            
        action_counts = defaultdict(int)
        for _, _, action, _ in self._action_observations:
            action_counts[action] += 1
        
        # Build report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'video_path': self.video_path,
                'video_duration_seconds': round(self.duration, 2),
                'total_frames': self.total_frames,
                'fps': self.fps,
                'analysis_config': {
                    'face_confidence': self.config.face_confidence,
                    'emotion_confidence': self.config.emotion_confidence,
                    'action_confidence': self.config.action_confidence,
                    'anomaly_sensitivity': self.config.anomaly_sensitivity
                }
            },
            'summary': {
                'total_observations': len(all_observations),
                'positive_count': len(positive),
                'negative_count': len(negative),
                'neutral_count': len(neutral),
                'anomaly_count': len(anomaly_reports),
                'emotions_detected': dict(emotion_counts),
                'actions_detected': dict(action_counts)
            },
            'positive_observations': sorted(positive, key=lambda x: x['time_range']['start_time']),
            'negative_observations': sorted(negative, key=lambda x: x['time_range']['start_time']),
            'neutral_observations': sorted(neutral, key=lambda x: x['time_range']['start_time']),
            'anomalies': [a.to_dict() for a in anomaly_reports]
        }
        
        return report
    
    def save_report(self, output_path: str) -> str:
        """
        Generate and save report to file.
        
        Args:
            output_path: Path to save JSON report
            
        Returns:
            Path to saved report
        """
        report = self.generate_report()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        return str(output_path)
    
    def get_live_stats(self) -> dict:
        """Get live statistics for visualization."""
        emotion_counts = defaultdict(int)
        for _, _, emotion, _ in self._emotion_observations:
            emotion_counts[emotion] += 1
            
        action_counts = defaultdict(int)
        for _, _, action, _ in self._action_observations:
            action_counts[action] += 1
        
        # Count by category
        positive = sum(1 for _, _, e, _ in self._emotion_observations 
                      if e in self.config.positive_emotions)
        positive += sum(1 for _, _, a, _ in self._action_observations 
                       if a in self.config.positive_actions)
        
        negative = sum(1 for _, _, e, _ in self._emotion_observations 
                      if e in self.config.negative_emotions)
        negative += sum(1 for _, _, a, _ in self._action_observations 
                       if a in self.config.negative_actions)
        
        neutral = sum(1 for _, _, e, _ in self._emotion_observations 
                     if e in self.config.neutral_emotions)
        neutral += sum(1 for _, _, a, _ in self._action_observations 
                      if a in self.config.neutral_actions)
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'anomalies': len(self._anomalies),
            'emotions': dict(emotion_counts),
            'actions': dict(action_counts)
        }
    
    def reset(self):
        """Reset all observations."""
        self._emotion_observations.clear()
        self._action_observations.clear()
        self._anomalies.clear()
