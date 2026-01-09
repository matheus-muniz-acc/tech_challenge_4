"""Emotion analysis using DeepFace."""

import cv2
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from deepface import DeepFace


@dataclass
class EmotionResult:
    """Represents emotion analysis result."""
    dominant_emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    bbox: Tuple[int, int, int, int]  # Face bounding box


class EmotionAnalyzer:
    """Emotion analysis using DeepFace (GPU accelerated via TensorFlow)."""
    
    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, min_confidence: float = 0.6, use_gpu: bool = True):
        """
        Initialize emotion analyzer.
        
        Args:
            min_confidence: Minimum confidence threshold
            use_gpu: Whether to use GPU acceleration
        """
        self.min_confidence = min_confidence
        self.use_gpu = use_gpu
        
        # Pre-warm the model by running a dummy analysis
        self._warm_up()
        
    def _warm_up(self):
        """Pre-load the emotion model."""
        try:
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            DeepFace.analyze(
                dummy,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
        except Exception:
            pass
        
    def analyze(self, frame: np.ndarray, faces: Optional[List] = None) -> List[EmotionResult]:
        """
        Analyze emotions in frame.
        
        Args:
            frame: BGR image
            faces: Optional list of FaceDetection objects (if None, detects faces internally)
            
        Returns:
            List of EmotionResult objects
        """
        results = []
        
        if faces is not None:
            # Analyze provided face regions
            for face in faces:
                if face.face_image is not None and face.face_image.size > 0:
                    emotions = self._analyze_face_image(face.face_image)
                    if emotions:
                        dominant, confidence = max(emotions.items(), key=lambda x: x[1])
                        # DeepFace returns percentages, normalize to 0-1
                        confidence = confidence / 100.0
                        if confidence >= self.min_confidence:
                            # Normalize all emotions to 0-1
                            normalized = {k: v / 100.0 for k, v in emotions.items()}
                            results.append(EmotionResult(
                                dominant_emotion=dominant,
                                confidence=confidence,
                                all_emotions=normalized,
                                bbox=face.bbox
                            ))
        else:
            # Use DeepFace's built-in detection
            try:
                detections = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Handle single or multiple results
                if isinstance(detections, dict):
                    detections = [detections]
                
                for det in detections:
                    emotions = det.get('emotion', {})
                    region = det.get('region', {})
                    
                    if emotions:
                        dominant = det.get('dominant_emotion', max(emotions, key=emotions.get))
                        confidence = emotions.get(dominant, 0) / 100.0
                        
                        if confidence >= self.min_confidence:
                            bbox = (
                                region.get('x', 0),
                                region.get('y', 0),
                                region.get('w', 0),
                                region.get('h', 0)
                            )
                            normalized = {k: v / 100.0 for k, v in emotions.items()}
                            results.append(EmotionResult(
                                dominant_emotion=dominant,
                                confidence=confidence,
                                all_emotions=normalized,
                                bbox=bbox
                            ))
            except Exception:
                pass
        
        return results
    
    def _analyze_face_image(self, face_image: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Analyze emotions in a cropped face image.
        
        Args:
            face_image: Cropped face BGR image
            
        Returns:
            Dictionary of emotion scores or None if analysis fails
        """
        if face_image is None or face_image.size == 0:
            return None
            
        # Ensure minimum size for analysis
        h, w = face_image.shape[:2]
        if h < 48 or w < 48:
            # Resize if too small
            scale = 48 / min(h, w)
            face_image = cv2.resize(face_image, None, fx=scale, fy=scale)
        
        try:
            result = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            # Handle list or dict response
            if isinstance(result, list):
                result = result[0] if result else {}
            
            return result.get('emotion', None)
        except Exception:
            pass
            
        return None
    
    def analyze_single_face(self, face_image: np.ndarray) -> Optional[EmotionResult]:
        """
        Analyze emotion for a single face image.
        
        Args:
            face_image: Cropped face BGR image
            
        Returns:
            EmotionResult or None
        """
        emotions = self._analyze_face_image(face_image)
        if emotions:
            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant] / 100.0
            normalized = {k: v / 100.0 for k, v in emotions.items()}
            return EmotionResult(
                dominant_emotion=dominant,
                confidence=confidence,
                all_emotions=normalized,
                bbox=(0, 0, face_image.shape[1], face_image.shape[0])
            )
        return None
