"""Pose estimation using MediaPipe Pose."""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Please install: pip install mediapipe==0.10.9")


class PoseLandmark(IntEnum):
    """MediaPipe pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class PoseResult:
    """Represents pose estimation result."""
    landmarks: Dict[int, Tuple[float, float, float]]  # idx -> (x, y, visibility)
    world_landmarks: Optional[Dict[int, Tuple[float, float, float]]] = None
    
    def get_landmark(self, idx: int) -> Optional[Tuple[float, float, float]]:
        """Get landmark by index."""
        return self.landmarks.get(idx)
    
    def get_pixel_coords(self, idx: int, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Get pixel coordinates for a landmark."""
        lm = self.landmarks.get(idx)
        if lm:
            h, w = frame_shape[:2]
            return (int(lm[0] * w), int(lm[1] * h))
        return None


class PoseEstimator:
    """Pose estimation using MediaPipe Pose (GPU accelerated)."""
    
    def __init__(self, min_confidence: float = 0.6, model_complexity: int = 1):
        """
        Initialize pose estimator.
        
        Args:
            min_confidence: Minimum detection confidence
            model_complexity: 0=Lite, 1=Full, 2=Heavy (accuracy vs speed)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required but not installed. "
                "Please install it: pip install mediapipe==0.10.9"
            )
        
        self.min_confidence = min_confidence
        
        try:
            self.mp_pose = mp.solutions.pose
        except AttributeError as e:
            raise ImportError(
                f"MediaPipe installation is incomplete or corrupted. "
                f"Please reinstall: pip uninstall mediapipe && pip install mediapipe==0.10.9\n"
                f"Error: {e}"
            )
        
        # Initialize pose estimator
        # model_complexity: 0, 1, or 2 (higher = more accurate but slower)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        
    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Estimate poses in frame.
        
        Args:
            frame: BGR image
            
        Returns:
            List of PoseResult objects (MediaPipe only detects one person)
        """
        results = []
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.pose_landmarks:
            landmarks = {}
            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                landmarks[idx] = (lm.x, lm.y, lm.visibility)
            
            world_landmarks = None
            if pose_results.pose_world_landmarks:
                world_landmarks = {}
                for idx, lm in enumerate(pose_results.pose_world_landmarks.landmark):
                    world_landmarks[idx] = (lm.x, lm.y, lm.z)
            
            results.append(PoseResult(
                landmarks=landmarks,
                world_landmarks=world_landmarks
            ))
        
        return results
    
    def get_arm_angles(self, pose: PoseResult) -> Dict[str, float]:
        """
        Calculate arm angles for action detection.
        
        Args:
            pose: PoseResult object
            
        Returns:
            Dictionary with arm angles
        """
        angles = {}
        
        # Left arm angle (shoulder-elbow-wrist)
        left_shoulder = pose.get_landmark(PoseLandmark.LEFT_SHOULDER)
        left_elbow = pose.get_landmark(PoseLandmark.LEFT_ELBOW)
        left_wrist = pose.get_landmark(PoseLandmark.LEFT_WRIST)
        
        if all([left_shoulder, left_elbow, left_wrist]):
            angles['left_elbow'] = self._calculate_angle(
                left_shoulder[:2], left_elbow[:2], left_wrist[:2]
            )
            angles['left_arm_raise'] = self._calculate_arm_raise_angle(
                left_shoulder[:2], left_elbow[:2]
            )
        
        # Right arm angle
        right_shoulder = pose.get_landmark(PoseLandmark.RIGHT_SHOULDER)
        right_elbow = pose.get_landmark(PoseLandmark.RIGHT_ELBOW)
        right_wrist = pose.get_landmark(PoseLandmark.RIGHT_WRIST)
        
        if all([right_shoulder, right_elbow, right_wrist]):
            angles['right_elbow'] = self._calculate_angle(
                right_shoulder[:2], right_elbow[:2], right_wrist[:2]
            )
            angles['right_arm_raise'] = self._calculate_arm_raise_angle(
                right_shoulder[:2], right_elbow[:2]
            )
        
        return angles
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                         p3: Tuple[float, float]) -> float:
        """Calculate angle at p2 formed by p1-p2-p3."""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_arm_raise_angle(self, shoulder: Tuple[float, float], 
                                    elbow: Tuple[float, float]) -> float:
        """Calculate how high the arm is raised (angle from vertical down)."""
        # Vector from shoulder to elbow
        dx = elbow[0] - shoulder[0]
        dy = elbow[1] - shoulder[1]
        
        # Angle from vertical (down is 0 degrees, horizontal is 90)
        angle = np.degrees(np.arctan2(abs(dx), dy))
        return angle
    
    def release(self):
        """Release resources."""
        self.pose.close()
