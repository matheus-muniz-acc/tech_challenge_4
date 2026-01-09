"""Face detection using MediaPipe Face Detection."""

import cv2
import numpy as np
from typing import List, Optional, Tuple, NamedTuple
from dataclasses import dataclass

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Please install: pip install mediapipe==0.10.9")


@dataclass
class FaceDetection:
    """Represents a detected face."""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    landmarks: Optional[List[Tuple[int, int]]] = None
    face_image: Optional[np.ndarray] = None


class FaceDetector:
    """GPU-accelerated face detection using MediaPipe."""
    
    def __init__(self, min_confidence: float = 0.6, model_selection: int = 1):
        """
        Initialize face detector.
        
        Args:
            min_confidence: Minimum detection confidence (0.0-1.0)
            model_selection: 0 for short-range (2m), 1 for full-range (5m)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe is required but not installed. "
                "Please install it: pip install mediapipe==0.10.9"
            )
        
        self.min_confidence = min_confidence
        
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
        except AttributeError as e:
            raise ImportError(
                f"MediaPipe installation is incomplete or corrupted. "
                f"Please reinstall: pip uninstall mediapipe && pip install mediapipe==0.10.9\n"
                f"Error: {e}"
            )
        
        # Initialize face detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=model_selection
        )
        
        # Initialize face mesh for landmarks (optional, more detailed)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect(self, frame: np.ndarray, extract_faces: bool = True) -> List[FaceDetection]:
        """
        Detect faces in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
            extract_faces: Whether to extract face ROI images
            
        Returns:
            List of FaceDetection objects
        """
        detections = []
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                confidence = detection.score[0]
                
                if confidence < self.min_confidence:
                    continue
                
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Clamp to frame boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extract key landmarks (eyes, nose, mouth, ears)
                landmarks = []
                for keypoint in detection.location_data.relative_keypoints:
                    px = int(keypoint.x * w)
                    py = int(keypoint.y * h)
                    landmarks.append((px, py))
                
                # Extract face image if requested
                face_image = None
                if extract_faces and width > 0 and height > 0:
                    # Add padding for better emotion analysis
                    pad = int(0.2 * max(width, height))
                    y1 = max(0, y - pad)
                    y2 = min(h, y + height + pad)
                    x1 = max(0, x - pad)
                    x2 = min(w, x + width + pad)
                    face_image = frame[y1:y2, x1:x2].copy()
                
                detections.append(FaceDetection(
                    bbox=(x, y, width, height),
                    confidence=confidence,
                    landmarks=landmarks,
                    face_image=face_image
                ))
        
        return detections
    
    def release(self):
        """Release resources."""
        self.face_detection.close()
        self.face_mesh.close()
