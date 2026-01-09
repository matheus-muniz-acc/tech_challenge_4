"""Anomaly detection for video analysis."""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from .face_detector import FaceDetection
from .emotion_analyzer import EmotionResult
from .pose_estimator import PoseResult, PoseLandmark


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    JARRING_MOVEMENT = "jarring_movement"
    RAPID_EMOTION_CHANGE = "rapid_emotion_change"
    UNDETECTABLE_FACE = "undetectable_face"
    POSE_DISCONTINUITY = "pose_discontinuity"
    LOW_CONFIDENCE_DETECTION = "low_confidence_detection"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_type: AnomalyType
    severity: float  # 0.0 - 1.0
    description: str
    frame_number: int
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class AnomalyDetector:
    """Detects anomalies in video analysis."""
    
    def __init__(self, 
                 motion_threshold: float = 0.15,
                 emotion_change_frames: int = 5,
                 sensitivity: float = 0.5,
                 history_size: int = 15):
        """
        Initialize anomaly detector.
        
        Args:
            motion_threshold: Threshold for jarring movement detection
            emotion_change_frames: Number of frames to consider for emotion changes
            sensitivity: Overall sensitivity (0.0 - 1.0)
            history_size: Size of history buffers
        """
        self.motion_threshold = motion_threshold
        self.emotion_change_frames = emotion_change_frames
        self.sensitivity = sensitivity
        self.history_size = history_size
        
        # Adjust thresholds based on sensitivity
        self._adjust_thresholds()
        
        # History buffers
        self.pose_history: deque = deque(maxlen=history_size)
        self.emotion_history: deque = deque(maxlen=history_size)
        self.face_history: deque = deque(maxlen=history_size)
        self.landmark_history: deque = deque(maxlen=history_size)
        
        # Frame tracking
        self.frame_count = 0
        self.fps = 30.0  # Will be updated
        
    def _adjust_thresholds(self):
        """Adjust detection thresholds based on sensitivity."""
        # Higher sensitivity = lower thresholds (more sensitive)
        sensitivity_factor = 1.0 - (self.sensitivity * 0.5)
        
        self.adjusted_motion_threshold = self.motion_threshold * sensitivity_factor
        self.adjusted_emotion_frames = max(3, int(self.emotion_change_frames * sensitivity_factor))
        
    def set_fps(self, fps: float):
        """Set video FPS for timestamp calculation."""
        self.fps = fps
        
    def detect(self, 
               frame_number: int,
               faces: List[FaceDetection],
               emotions: List[EmotionResult],
               poses: List[PoseResult]) -> List[Anomaly]:
        """
        Detect anomalies in current frame data.
        
        Args:
            frame_number: Current frame number
            faces: Face detections
            emotions: Emotion results
            poses: Pose results
            
        Returns:
            List of detected anomalies
        """
        self.frame_count = frame_number
        timestamp = frame_number / self.fps
        anomalies = []
        
        # Store current data in history
        self.face_history.append(faces)
        self.emotion_history.append(emotions)
        self.pose_history.append(poses)
        
        # Check for jarring movement
        movement_anomaly = self._detect_jarring_movement(frame_number, timestamp, poses)
        if movement_anomaly:
            anomalies.append(movement_anomaly)
        
        # Check for rapid emotion changes
        emotion_anomaly = self._detect_rapid_emotion_change(frame_number, timestamp, emotions)
        if emotion_anomaly:
            anomalies.append(emotion_anomaly)
        
        # Check for undetectable faces (face was visible but now can't be detected)
        face_anomaly = self._detect_face_discontinuity(frame_number, timestamp, faces)
        if face_anomaly:
            anomalies.append(face_anomaly)
        
        # Check for pose discontinuity
        pose_anomaly = self._detect_pose_discontinuity(frame_number, timestamp, poses)
        if pose_anomaly:
            anomalies.append(pose_anomaly)
        
        return anomalies
    
    def _detect_jarring_movement(self, frame_number: int, timestamp: float,
                                  poses: List[PoseResult]) -> Optional[Anomaly]:
        """Detect sudden, jarring movements."""
        if len(self.pose_history) < 2 or not poses:
            return None
        
        prev_poses = self.pose_history[-2] if len(self.pose_history) >= 2 else []
        
        if not prev_poses:
            return None
        
        # Compare current pose with previous
        current_pose = poses[0]
        prev_pose = prev_poses[0]
        
        # Calculate movement magnitude for key body parts
        key_landmarks = [
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP
        ]
        
        movements = []
        for lm_idx in key_landmarks:
            curr_lm = current_pose.get_landmark(lm_idx)
            prev_lm = prev_pose.get_landmark(lm_idx)
            
            if curr_lm and prev_lm:
                # Both should have reasonable visibility
                if curr_lm[2] > 0.5 and prev_lm[2] > 0.5:
                    dx = curr_lm[0] - prev_lm[0]
                    dy = curr_lm[1] - prev_lm[1]
                    movement = np.sqrt(dx**2 + dy**2)
                    movements.append(movement)
        
        if not movements:
            return None
        
        max_movement = max(movements)
        avg_movement = np.mean(movements)
        
        # Check against threshold
        if max_movement > self.adjusted_motion_threshold:
            severity = min(1.0, max_movement / (self.adjusted_motion_threshold * 2))
            
            return Anomaly(
                anomaly_type=AnomalyType.JARRING_MOVEMENT,
                severity=severity,
                description=f"Sudden movement detected (magnitude: {max_movement:.3f})",
                frame_number=frame_number,
                timestamp=timestamp,
                details={
                    'max_movement': max_movement,
                    'avg_movement': avg_movement,
                    'threshold': self.adjusted_motion_threshold
                }
            )
        
        return None
    
    def _detect_rapid_emotion_change(self, frame_number: int, timestamp: float,
                                      emotions: List[EmotionResult]) -> Optional[Anomaly]:
        """Detect rapid changes in facial expression."""
        if len(self.emotion_history) < self.adjusted_emotion_frames or not emotions:
            return None
        
        current_emotion = emotions[0].dominant_emotion if emotions else None
        
        if not current_emotion:
            return None
        
        # Look at emotion history
        recent_emotions = []
        for hist_emotions in list(self.emotion_history)[-self.adjusted_emotion_frames:]:
            if hist_emotions:
                recent_emotions.append(hist_emotions[0].dominant_emotion)
        
        if len(recent_emotions) < self.adjusted_emotion_frames:
            return None
        
        # Count emotion changes
        changes = 0
        unique_emotions = set()
        for i in range(1, len(recent_emotions)):
            if recent_emotions[i] != recent_emotions[i-1]:
                changes += 1
            unique_emotions.add(recent_emotions[i])
        
        # Rapid change = multiple different emotions in short window
        # Threshold: more than 2 changes in the window
        change_threshold = max(1, int(2 * (1 - self.sensitivity * 0.3)))
        
        if changes > change_threshold and len(unique_emotions) > 2:
            severity = min(1.0, changes / (self.adjusted_emotion_frames * 0.5))
            
            return Anomaly(
                anomaly_type=AnomalyType.RAPID_EMOTION_CHANGE,
                severity=severity,
                description=f"Rapid emotion changes detected ({changes} changes, {len(unique_emotions)} emotions)",
                frame_number=frame_number,
                timestamp=timestamp,
                details={
                    'changes': changes,
                    'unique_emotions': list(unique_emotions),
                    'recent_sequence': recent_emotions
                }
            )
        
        return None
    
    def _detect_face_discontinuity(self, frame_number: int, timestamp: float,
                                    faces: List[FaceDetection]) -> Optional[Anomaly]:
        """Detect when a face suddenly disappears or appears."""
        if len(self.face_history) < 3:
            return None
        
        # Check recent face counts
        recent_counts = [len(f) for f in list(self.face_history)[-5:]]
        current_count = len(faces)
        
        if len(recent_counts) < 3:
            return None
        
        # Calculate average recent count (excluding current)
        avg_recent = np.mean(recent_counts[:-1]) if len(recent_counts) > 1 else 0
        
        # Sudden disappearance: had faces, now none
        if avg_recent > 0.5 and current_count == 0:
            severity = 0.6
            return Anomaly(
                anomaly_type=AnomalyType.UNDETECTABLE_FACE,
                severity=severity,
                description="Face suddenly became undetectable",
                frame_number=frame_number,
                timestamp=timestamp,
                details={
                    'previous_avg_faces': avg_recent,
                    'current_faces': current_count
                }
            )
        
        return None
    
    def _detect_pose_discontinuity(self, frame_number: int, timestamp: float,
                                    poses: List[PoseResult]) -> Optional[Anomaly]:
        """Detect sudden pose estimation failures or discontinuities."""
        if len(self.pose_history) < 3:
            return None
        
        recent_pose_counts = [len(p) for p in list(self.pose_history)[-5:]]
        
        # Had poses but now none
        if len(recent_pose_counts) >= 3:
            avg_recent = np.mean(recent_pose_counts[:-1])
            current_count = len(poses)
            
            if avg_recent > 0.5 and current_count == 0:
                return Anomaly(
                    anomaly_type=AnomalyType.POSE_DISCONTINUITY,
                    severity=0.5,
                    description="Pose estimation suddenly failed",
                    frame_number=frame_number,
                    timestamp=timestamp,
                    details={
                        'previous_avg_poses': avg_recent,
                        'current_poses': current_count
                    }
                )
        
        return None
    
    def reset(self):
        """Reset all history buffers."""
        self.pose_history.clear()
        self.emotion_history.clear()
        self.face_history.clear()
        self.landmark_history.clear()
        self.frame_count = 0
