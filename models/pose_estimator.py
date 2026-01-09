"""Pose estimation using YOLOv8-Pose for multi-person detection."""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import IntEnum

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not available. Please install: pip install ultralytics")


class PoseLandmark(IntEnum):
    """YOLOv8 pose keypoint indices (COCO format - 17 keypoints)."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


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
    """Multi-person pose estimation using YOLOv8-Pose (GPU accelerated)."""
    
    # Model options: yolov8n-pose (fastest), yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose (most accurate)
    MODEL_OPTIONS = {
        0: 'yolov8n-pose.pt',  # Nano - fastest
        1: 'yolov8s-pose.pt',  # Small - balanced
        2: 'yolov8m-pose.pt',  # Medium - more accurate
    }
    
    def __init__(self, min_confidence: float = 0.6, model_complexity: int = 1):
        """
        Initialize pose estimator.
        
        Args:
            min_confidence: Minimum detection confidence
            model_complexity: 0=Nano (fast), 1=Small (balanced), 2=Medium (accurate)
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics is required but not installed. "
                "Please install it: pip install ultralytics"
            )
        
        self.min_confidence = min_confidence
        
        # Select model based on complexity
        model_name = self.MODEL_OPTIONS.get(model_complexity, 'yolov8s-pose.pt')
        
        # Initialize YOLO pose model (downloads automatically on first use)
        print(f"Loading YOLOv8 Pose model: {model_name}")
        self.model = YOLO(model_name)
        
        # Try GPU first, fall back to CPU if CUDA fails
        self.device = self._select_device()
        
    def _select_device(self) -> str:
        """Select best available device, with CUDA fallback to CPU."""
        try:
            import torch
            import torch.nn as nn
            
            if torch.cuda.is_available():
                # Test if CUDA actually works with a convolution operation (like YOLO uses)
                try:
                    # Create a small conv2d operation to test if kernels are available
                    test_conv = nn.Conv2d(3, 8, 3, padding=1).cuda()
                    test_input = torch.randn(1, 3, 32, 32).cuda()
                    with torch.no_grad():
                        _ = test_conv(test_input)
                    
                    del test_conv, test_input
                    torch.cuda.empty_cache()
                    
                    print("✓ YOLOv8 Pose using device: cuda")
                    return 'cuda'
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if 'no kernel image is available' in error_msg:
                        print("\n" + "="*70)
                        print("⚠ GPU NOT COMPATIBLE")
                        print("="*70)
                        print(f"Your RTX 5070 TI is too new for PyTorch {torch.__version__}")
                        print("Falling back to CPU for pose estimation (will be slower)")
                        print("\nTo enable GPU support:")
                        print("1. Uninstall current PyTorch:")
                        print("   pip uninstall torch torchvision -y")
                        print("2. Install PyTorch nightly with CUDA 12.4:")
                        print("   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124")
                        print("="*70 + "\n")
                    else:
                        print(f"CUDA test failed: {e}")
                        print("Falling back to CPU for pose estimation")
                    return 'cpu'
            else:
                print("CUDA not available, using CPU for pose estimation")
                return 'cpu'
        except ImportError:
            print("PyTorch not found, using CPU")
            return 'cpu'
        
    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Estimate poses in frame (supports multiple people).
        
        Args:
            frame: BGR image
            
        Returns:
            List of PoseResult objects (one per detected person)
        """
        results = []
        h, w = frame.shape[:2]
        
        # Run YOLO pose detection
        detections = self.model(
            frame,
            conf=self.min_confidence,
            device=self.device,
            verbose=False
        )
        
        # Process each detection
        for detection in detections:
            if detection.keypoints is None:
                continue
                
            # Get keypoints data: shape (num_persons, 17, 3) where 3 = (x, y, confidence)
            keypoints_data = detection.keypoints.data.cpu().numpy()
            
            for person_kpts in keypoints_data:
                landmarks = {}
                
                for idx, kpt in enumerate(person_kpts):
                    x, y, conf = kpt
                    
                    # Normalize coordinates to 0-1 range (same as MediaPipe format)
                    x_norm = x / w
                    y_norm = y / h
                    
                    # Store as (x_normalized, y_normalized, confidence)
                    landmarks[idx] = (float(x_norm), float(y_norm), float(conf))
                
                # Only add if we have valid keypoints
                if landmarks and any(lm[2] > self.min_confidence for lm in landmarks.values()):
                    results.append(PoseResult(
                        landmarks=landmarks,
                        world_landmarks=None  # YOLO doesn't provide world coordinates
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
        # YOLO models don't require explicit cleanup
        pass
