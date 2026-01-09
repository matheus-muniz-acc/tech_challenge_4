"""Action classification based on pose estimation."""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from collections import deque

from .pose_estimator import PoseResult, PoseLandmark


class ActionType(Enum):
    """Detected action types."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    HANDSHAKE = "handshake"
    WAVE = "wave"
    HAND_RAISED = "hand_raised"


@dataclass
class ActionResult:
    """Represents an action detection result."""
    action: ActionType
    confidence: float
    details: Dict[str, any] = None


class ActionClassifier:
    """Classifies actions based on pose landmarks."""
    
    def __init__(self, min_confidence: float = 0.65, history_size: int = 10):
        """
        Initialize action classifier.
        
        Args:
            min_confidence: Minimum confidence for action detection
            history_size: Number of frames to consider for temporal actions
        """
        self.min_confidence = min_confidence
        self.history_size = history_size
        
        # History buffers for temporal analysis
        self.wrist_history: deque = deque(maxlen=history_size)
        self.pose_history: deque = deque(maxlen=history_size)
        
    def classify(self, poses: List[PoseResult], frame_shape: Tuple[int, int]) -> List[ActionResult]:
        """
        Classify actions from poses.
        
        Args:
            poses: List of PoseResult objects
            frame_shape: (height, width) of the frame
            
        Returns:
            List of ActionResult objects
        """
        results = []
        
        for pose in poses:
            # Store pose in history
            self.pose_history.append(pose)
            
            # Check each action type
            actions = []
            
            # Standing/Sitting detection
            posture = self._detect_posture(pose)
            if posture:
                actions.append(posture)
            
            # Wave detection
            wave = self._detect_wave(pose, frame_shape)
            if wave:
                actions.append(wave)
            
            # Handshake detection (requires specific pose)
            handshake = self._detect_handshake(pose, frame_shape)
            if handshake:
                actions.append(handshake)
            
            # If no specific action detected
            if not actions:
                actions.append(ActionResult(
                    action=ActionType.UNKNOWN,
                    confidence=0.0
                ))
            
            results.extend(actions)
        
        return results
    
    def _detect_posture(self, pose: PoseResult) -> Optional[ActionResult]:
        """Detect standing or sitting posture."""
        # Get relevant landmarks
        left_hip = pose.get_landmark(PoseLandmark.LEFT_HIP)
        right_hip = pose.get_landmark(PoseLandmark.RIGHT_HIP)
        left_knee = pose.get_landmark(PoseLandmark.LEFT_KNEE)
        right_knee = pose.get_landmark(PoseLandmark.RIGHT_KNEE)
        left_ankle = pose.get_landmark(PoseLandmark.LEFT_ANKLE)
        right_ankle = pose.get_landmark(PoseLandmark.RIGHT_ANKLE)
        left_shoulder = pose.get_landmark(PoseLandmark.LEFT_SHOULDER)
        right_shoulder = pose.get_landmark(PoseLandmark.RIGHT_SHOULDER)
        
        # Need hips and knees at minimum
        if not all([left_hip, right_hip, left_knee, right_knee]):
            return None
        
        # Check visibility
        hip_visibility = min(left_hip[2], right_hip[2])
        knee_visibility = min(left_knee[2], right_knee[2])
        
        if hip_visibility < 0.5 or knee_visibility < 0.5:
            return None
        
        # Calculate hip-knee angle to determine posture
        # Sitting: knees are roughly at hip level or higher (y values close)
        # Standing: knees are significantly below hips
        
        hip_y = (left_hip[1] + right_hip[1]) / 2
        knee_y = (left_knee[1] + right_knee[1]) / 2
        
        # Vertical distance between hip and knee (normalized)
        hip_knee_dist = knee_y - hip_y
        
        # Calculate knee angle for additional info
        knee_angle = None
        if all([left_hip, left_knee, left_ankle]):
            knee_angle = self._calculate_angle(
                (left_hip[0], left_hip[1]),
                (left_knee[0], left_knee[1]),
                (left_ankle[0], left_ankle[1])
            )
        
        # Determine posture
        if hip_knee_dist < 0.15:  # Knees close to hip level = sitting
            # Additional check: knee angle < 120 degrees typically indicates sitting
            confidence = 0.7
            if knee_angle and knee_angle < 130:
                confidence = 0.85
            
            return ActionResult(
                action=ActionType.SITTING,
                confidence=confidence,
                details={'hip_knee_distance': hip_knee_dist, 'knee_angle': knee_angle}
            )
        elif hip_knee_dist > 0.2:  # Knees well below hips = standing
            confidence = 0.75
            if knee_angle and knee_angle > 160:
                confidence = 0.9
            
            return ActionResult(
                action=ActionType.STANDING,
                confidence=confidence,
                details={'hip_knee_distance': hip_knee_dist, 'knee_angle': knee_angle}
            )
        
        return None
    
    def _detect_wave(self, pose: PoseResult, frame_shape: Tuple[int, int]) -> Optional[ActionResult]:
        """Detect hand waving gesture."""
        h, w = frame_shape[:2]
        
        # Get arm landmarks
        left_shoulder = pose.get_landmark(PoseLandmark.LEFT_SHOULDER)
        left_elbow = pose.get_landmark(PoseLandmark.LEFT_ELBOW)
        left_wrist = pose.get_landmark(PoseLandmark.LEFT_WRIST)
        
        right_shoulder = pose.get_landmark(PoseLandmark.RIGHT_SHOULDER)
        right_elbow = pose.get_landmark(PoseLandmark.RIGHT_ELBOW)
        right_wrist = pose.get_landmark(PoseLandmark.RIGHT_WRIST)
        
        wave_detected = False
        confidence = 0.0
        waving_hand = None
        
        # Check left arm
        if all([left_shoulder, left_elbow, left_wrist]):
            if left_wrist[2] > 0.5 and left_elbow[2] > 0.5:
                # Check if hand is raised above shoulder
                if left_wrist[1] < left_shoulder[1]:
                    # Check elbow angle (bent arm for waving)
                    elbow_angle = self._calculate_angle(
                        (left_shoulder[0], left_shoulder[1]),
                        (left_elbow[0], left_elbow[1]),
                        (left_wrist[0], left_wrist[1])
                    )
                    
                    # Waving typically has bent elbow (60-150 degrees)
                    if 60 < elbow_angle < 150:
                        # Check for oscillating motion in history
                        motion_score = self._check_waving_motion('left')
                        if motion_score > 0.3:
                            wave_detected = True
                            confidence = min(0.9, 0.6 + motion_score * 0.4)
                            waving_hand = 'left'
        
        # Check right arm
        if all([right_shoulder, right_elbow, right_wrist]) and not wave_detected:
            if right_wrist[2] > 0.5 and right_elbow[2] > 0.5:
                if right_wrist[1] < right_shoulder[1]:
                    elbow_angle = self._calculate_angle(
                        (right_shoulder[0], right_shoulder[1]),
                        (right_elbow[0], right_elbow[1]),
                        (right_wrist[0], right_wrist[1])
                    )
                    
                    if 60 < elbow_angle < 150:
                        motion_score = self._check_waving_motion('right')
                        if motion_score > 0.3:
                            wave_detected = True
                            confidence = min(0.9, 0.6 + motion_score * 0.4)
                            waving_hand = 'right'
        
        # Store wrist positions for motion detection
        if left_wrist and right_wrist:
            self.wrist_history.append({
                'left': (left_wrist[0], left_wrist[1]),
                'right': (right_wrist[0], right_wrist[1])
            })
        
        if wave_detected and confidence >= self.min_confidence:
            return ActionResult(
                action=ActionType.WAVE,
                confidence=confidence,
                details={'hand': waving_hand}
            )
        
        return None
    
    def _check_waving_motion(self, hand: str) -> float:
        """
        Check for oscillating waving motion in history.
        
        Returns:
            Motion score (0.0 - 1.0)
        """
        if len(self.wrist_history) < 5:
            return 0.0
        
        # Get recent wrist positions
        positions = [h.get(hand) for h in self.wrist_history if h.get(hand)]
        
        if len(positions) < 5:
            return 0.0
        
        # Calculate horizontal movement changes (direction reversals)
        x_positions = [p[0] for p in positions]
        
        # Count direction changes
        direction_changes = 0
        for i in range(2, len(x_positions)):
            prev_dir = x_positions[i-1] - x_positions[i-2]
            curr_dir = x_positions[i] - x_positions[i-1]
            if prev_dir * curr_dir < 0:  # Direction changed
                direction_changes += 1
        
        # Calculate total movement amplitude
        amplitude = max(x_positions) - min(x_positions)
        
        # Score based on direction changes and amplitude
        if direction_changes >= 2 and amplitude > 0.05:
            return min(1.0, (direction_changes / 4) * (amplitude / 0.15))
        
        return 0.0
    
    def _detect_handshake(self, pose: PoseResult, frame_shape: Tuple[int, int]) -> Optional[ActionResult]:
        """
        Detect handshake gesture.
        Note: Full handshake detection requires two people, this detects
        the handshake-ready pose (arm extended forward).
        """
        right_shoulder = pose.get_landmark(PoseLandmark.RIGHT_SHOULDER)
        right_elbow = pose.get_landmark(PoseLandmark.RIGHT_ELBOW)
        right_wrist = pose.get_landmark(PoseLandmark.RIGHT_WRIST)
        
        left_shoulder = pose.get_landmark(PoseLandmark.LEFT_SHOULDER)
        left_elbow = pose.get_landmark(PoseLandmark.LEFT_ELBOW)
        left_wrist = pose.get_landmark(PoseLandmark.LEFT_WRIST)
        
        handshake_pose = False
        confidence = 0.0
        extending_hand = None
        
        # Check right arm extended forward
        if all([right_shoulder, right_elbow, right_wrist]):
            if all(lm[2] > 0.5 for lm in [right_shoulder, right_elbow, right_wrist]):
                # Elbow angle should be relatively straight (extended arm)
                elbow_angle = self._calculate_angle(
                    (right_shoulder[0], right_shoulder[1]),
                    (right_elbow[0], right_elbow[1]),
                    (right_wrist[0], right_wrist[1])
                )
                
                # Hand should be roughly at hip to chest level
                shoulder_y = right_shoulder[1]
                wrist_y = right_wrist[1]
                
                # Arm relatively straight and hand at appropriate height
                if elbow_angle > 140 and abs(wrist_y - shoulder_y) < 0.2:
                    # Check if arm is extended (wrist away from shoulder horizontally)
                    horizontal_extension = abs(right_wrist[0] - right_shoulder[0])
                    if horizontal_extension > 0.15:
                        handshake_pose = True
                        confidence = min(0.85, 0.6 + horizontal_extension)
                        extending_hand = 'right'
        
        # Check left arm if right didn't match
        if not handshake_pose and all([left_shoulder, left_elbow, left_wrist]):
            if all(lm[2] > 0.5 for lm in [left_shoulder, left_elbow, left_wrist]):
                elbow_angle = self._calculate_angle(
                    (left_shoulder[0], left_shoulder[1]),
                    (left_elbow[0], left_elbow[1]),
                    (left_wrist[0], left_wrist[1])
                )
                
                shoulder_y = left_shoulder[1]
                wrist_y = left_wrist[1]
                
                if elbow_angle > 140 and abs(wrist_y - shoulder_y) < 0.2:
                    horizontal_extension = abs(left_wrist[0] - left_shoulder[0])
                    if horizontal_extension > 0.15:
                        handshake_pose = True
                        confidence = min(0.85, 0.6 + horizontal_extension)
                        extending_hand = 'left'
        
        if handshake_pose and confidence >= self.min_confidence:
            return ActionResult(
                action=ActionType.HANDSHAKE,
                confidence=confidence,
                details={'hand': extending_hand}
            )
        
        return None
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float],
                         p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def reset_history(self):
        """Reset motion history buffers."""
        self.wrist_history.clear()
        self.pose_history.clear()
