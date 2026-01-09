"""Real-time visualization for video analysis."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from models.face_detector import FaceDetection
from models.emotion_analyzer import EmotionResult
from models.pose_estimator import PoseResult, PoseLandmark
from models.action_classifier import ActionResult, ActionType
from models.anomaly_detector import Anomaly, AnomalyType
from config import AnalysisConfig


class Visualizer:
    """Handles real-time visualization of analysis results."""
    
    # Pose connections for drawing skeleton (YOLO COCO 17 keypoints)
    POSE_CONNECTIONS = [
        # Head
        (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE),
        (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE),
        (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EAR),
        (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EAR),
        # Torso
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
        (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
        (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
        # Left arm
        (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
        (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
        # Right arm
        (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
        (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
        # Left leg
        (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
        (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
        # Right leg
        (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
        (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    ]
    
    def __init__(self, config: AnalysisConfig):
        """
        Initialize visualizer.
        
        Args:
            config: Analysis configuration
        """
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def draw_all(self, 
                 frame: np.ndarray,
                 faces: List[FaceDetection],
                 emotions: List[EmotionResult],
                 poses: List[PoseResult],
                 actions: List[ActionResult],
                 anomalies: List[Anomaly],
                 frame_info: dict) -> np.ndarray:
        """
        Draw all visualizations on frame.
        
        Args:
            frame: Input frame (will be modified)
            faces: Face detections
            emotions: Emotion results
            poses: Pose results
            actions: Action results
            anomalies: Detected anomalies
            frame_info: Frame metadata (frame_number, timestamp, fps)
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Draw poses first (background layer)
        if self.config.show_landmarks and poses:
            output = self.draw_poses(output, poses)
        
        # Draw faces and emotions
        if faces:
            output = self.draw_faces(output, faces)
        
        if self.config.show_emotions and emotions:
            output = self.draw_emotions(output, emotions)
        
        # Draw actions
        if self.config.show_actions and actions:
            output = self.draw_actions(output, actions, poses)
        
        # Draw anomalies
        if self.config.show_anomalies and anomalies:
            output = self.draw_anomalies(output, anomalies)
        
        # Draw frame info overlay
        output = self.draw_frame_info(output, frame_info)
        
        return output
    
    def draw_faces(self, frame: np.ndarray, faces: List[FaceDetection]) -> np.ndarray:
        """Draw face bounding boxes."""
        for face in faces:
            x, y, w, h = face.bbox
            color = self.config.color_neutral
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence
            conf_text = f"{face.confidence:.0%}"
            cv2.putText(frame, conf_text, (x, y - 5),
                       self.font, 0.5, color, 1)
            
            # Draw landmarks if available
            if self.config.show_landmarks and face.landmarks:
                for lx, ly in face.landmarks:
                    cv2.circle(frame, (lx, ly), 2, color, -1)
        
        return frame
    
    def draw_emotions(self, frame: np.ndarray, emotions: List[EmotionResult]) -> np.ndarray:
        """Draw emotion labels on faces."""
        for emotion in emotions:
            x, y, w, h = emotion.bbox
            
            # Determine color based on emotion type
            if emotion.dominant_emotion in self.config.positive_emotions:
                color = self.config.color_positive
            elif emotion.dominant_emotion in self.config.negative_emotions:
                color = self.config.color_negative
            else:
                color = self.config.color_neutral
            
            # Update bounding box color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label with background
            label = f"{emotion.dominant_emotion}: {emotion.confidence:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            
            # Background rectangle
            cv2.rectangle(frame, (x, y + h), (x + text_w + 4, y + h + text_h + 8), color, -1)
            
            # Text
            cv2.putText(frame, label, (x + 2, y + h + text_h + 4),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
        
        return frame
    
    def draw_poses(self, frame: np.ndarray, poses: List[PoseResult]) -> np.ndarray:
        """Draw pose skeletons."""
        h, w = frame.shape[:2]
        
        for pose in poses:
            # Draw connections
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                start = pose.get_landmark(start_idx)
                end = pose.get_landmark(end_idx)
                
                if start and end:
                    # Check visibility
                    if start[2] > 0.5 and end[2] > 0.5:
                        start_px = (int(start[0] * w), int(start[1] * h))
                        end_px = (int(end[0] * w), int(end[1] * h))
                        cv2.line(frame, start_px, end_px, (0, 255, 255), 2)
            
            # Draw landmark points
            for idx, (x, y, vis) in pose.landmarks.items():
                if vis > 0.5:
                    px, py = int(x * w), int(y * h)
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
        
        return frame
    
    def draw_actions(self, frame: np.ndarray, actions: List[ActionResult],
                     poses: List[PoseResult]) -> np.ndarray:
        """Draw action labels."""
        h, w = frame.shape[:2]
        
        # Group actions for display
        action_texts = []
        for action in actions:
            if action.action == ActionType.UNKNOWN:
                continue
                
            # Determine color
            action_name = action.action.value
            if action_name in self.config.positive_actions:
                color = self.config.color_positive
            elif action_name in self.config.negative_actions:
                color = self.config.color_negative
            else:
                color = self.config.color_neutral
            
            action_texts.append((action_name, action.confidence, color))
        
        # Draw action labels in top-left area
        y_offset = 80
        for action_name, confidence, color in action_texts:
            label = f"Action: {action_name} ({confidence:.0%})"
            cv2.putText(frame, label, (10, y_offset),
                       self.font, self.font_scale, color, self.thickness)
            y_offset += 25
        
        return frame
    
    def draw_anomalies(self, frame: np.ndarray, anomalies: List[Anomaly]) -> np.ndarray:
        """Draw anomaly warnings."""
        h, w = frame.shape[:2]
        
        if not anomalies:
            return frame
        
        # Draw red border for anomalies
        border_thickness = max(3, int(anomalies[0].severity * 8))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), 
                     self.config.color_anomaly, border_thickness)
        
        # Draw anomaly labels in top-right
        y_offset = 30
        for anomaly in anomalies:
            label = f"! {anomaly.anomaly_type.value}"
            (text_w, text_h), _ = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
            
            x_pos = w - text_w - 15
            
            # Background
            cv2.rectangle(frame, (x_pos - 5, y_offset - text_h - 5),
                         (w - 5, y_offset + 5), self.config.color_anomaly, -1)
            
            # Text
            cv2.putText(frame, label, (x_pos, y_offset),
                       self.font, self.font_scale, (255, 255, 255), self.thickness)
            
            y_offset += 30
        
        return frame
    
    def draw_frame_info(self, frame: np.ndarray, frame_info: dict) -> np.ndarray:
        """Draw frame information overlay."""
        h, w = frame.shape[:2]
        
        frame_num = frame_info.get('frame_number', 0)
        timestamp = frame_info.get('timestamp', 0.0)
        fps = frame_info.get('fps', 0.0)
        processing_fps = frame_info.get('processing_fps', 0.0)
        
        # Top-left info box
        info_lines = [
            f"Frame: {frame_num}",
            f"Time: {timestamp:.2f}s",
            f"FPS: {processing_fps:.1f}"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (10, y_offset),
                       self.font, 0.5, (255, 255, 255), 1)
            y_offset += 18
        
        return frame
    
    def create_summary_panel(self, 
                             stats: dict,
                             panel_width: int = 300,
                             panel_height: int = 400) -> np.ndarray:
        """
        Create a summary statistics panel.
        
        Args:
            stats: Dictionary of statistics
            panel_width: Panel width
            panel_height: Panel height
            
        Returns:
            Summary panel image
        """
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, "Analysis Summary", (10, 30),
                   self.font, 0.7, (255, 255, 255), 2)
        
        # Draw line separator
        cv2.line(panel, (10, 45), (panel_width - 10, 45), (100, 100, 100), 1)
        
        y_offset = 70
        
        # Positive observations
        cv2.putText(panel, f"Positive: {stats.get('positive', 0)}", (10, y_offset),
                   self.font, 0.6, self.config.color_positive, 1)
        y_offset += 25
        
        # Negative observations
        cv2.putText(panel, f"Negative: {stats.get('negative', 0)}", (10, y_offset),
                   self.font, 0.6, self.config.color_negative, 1)
        y_offset += 25
        
        # Neutral observations
        cv2.putText(panel, f"Neutral: {stats.get('neutral', 0)}", (10, y_offset),
                   self.font, 0.6, self.config.color_neutral, 1)
        y_offset += 25
        
        # Anomalies
        cv2.putText(panel, f"Anomalies: {stats.get('anomalies', 0)}", (10, y_offset),
                   self.font, 0.6, self.config.color_anomaly, 1)
        y_offset += 35
        
        # Emotions breakdown
        cv2.line(panel, (10, y_offset), (panel_width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 25
        
        cv2.putText(panel, "Emotions Detected:", (10, y_offset),
                   self.font, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        emotions = stats.get('emotions', {})
        for emotion, count in sorted(emotions.items(), key=lambda x: -x[1])[:5]:
            cv2.putText(panel, f"  {emotion}: {count}", (10, y_offset),
                       self.font, 0.45, (180, 180, 180), 1)
            y_offset += 18
        
        # Actions breakdown
        y_offset += 10
        cv2.putText(panel, "Actions Detected:", (10, y_offset),
                   self.font, 0.5, (200, 200, 200), 1)
        y_offset += 20
        
        actions = stats.get('actions', {})
        for action, count in sorted(actions.items(), key=lambda x: -x[1])[:5]:
            cv2.putText(panel, f"  {action}: {count}", (10, y_offset),
                       self.font, 0.45, (180, 180, 180), 1)
            y_offset += 18
        
        return panel
