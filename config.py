"""Configuration settings for the video analysis system."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AnalysisConfig:
    """Configuration for video analysis parameters."""
    
    # Confidence thresholds
    face_confidence: float = 0.6
    emotion_confidence: float = 0.6
    pose_confidence: float = 0.6
    action_confidence: float = 0.65
    
    # Anomaly detection settings
    anomaly_motion_threshold: float = 0.15  # Normalized movement threshold
    anomaly_emotion_change_frames: int = 5  # Frames to detect rapid emotion change
    anomaly_sensitivity: float = 0.5  # 0.0 = low sensitivity, 1.0 = high sensitivity
    
    # Time range consolidation (seconds)
    consolidation_window: float = 0.5  # Merge detections within this window
    
    # Emotion classifications
    positive_emotions: List[str] = field(default_factory=lambda: ['happy', 'surprise'])
    negative_emotions: List[str] = field(default_factory=lambda: ['angry', 'fear', 'sad', 'disgust'])
    neutral_emotions: List[str] = field(default_factory=lambda: ['neutral'])
    
    # Action classifications
    positive_actions: List[str] = field(default_factory=lambda: ['handshake', 'wave'])
    negative_actions: List[str] = field(default_factory=lambda: [])
    neutral_actions: List[str] = field(default_factory=lambda: ['standing', 'sitting'])
    
    # Visualization settings
    show_landmarks: bool = True
    show_emotions: bool = True
    show_actions: bool = True
    show_anomalies: bool = True
    
    # Colors (BGR format for OpenCV)
    color_positive: tuple = (0, 255, 0)      # Green
    color_negative: tuple = (0, 0, 255)      # Red
    color_neutral: tuple = (255, 255, 0)     # Cyan
    color_anomaly: tuple = (0, 165, 255)     # Orange


@dataclass
class ProcessingConfig:
    """Configuration for processing performance."""
    
    # Frame processing
    process_every_n_frames: int = 2  # Skip frames for performance
    target_width: int = 640  # Resize for processing (None = original)
    
    # GPU settings
    use_gpu: bool = True
    device: str = "cuda"  # or "cpu"
    
    # Buffer sizes
    motion_history_size: int = 10
    emotion_history_size: int = 15
