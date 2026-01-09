"""Main video processing pipeline."""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
from pathlib import Path

from config import AnalysisConfig, ProcessingConfig
from models import (
    FaceDetector, 
    EmotionAnalyzer, 
    PoseEstimator, 
    ActionClassifier,
    AnomalyDetector
)
from visualization import Visualizer
from report_generator import ReportGenerator


class VideoProcessor:
    """Main video analysis processor with real-time visualization."""
    
    def __init__(self, 
                 analysis_config: Optional[AnalysisConfig] = None,
                 processing_config: Optional[ProcessingConfig] = None):
        """
        Initialize video processor.
        
        Args:
            analysis_config: Analysis configuration
            processing_config: Processing configuration
        """
        self.analysis_config = analysis_config or AnalysisConfig()
        self.processing_config = processing_config or ProcessingConfig()
        
        # Initialize models
        print("Initializing models...")
        self.face_detector = FaceDetector(
            min_confidence=self.analysis_config.face_confidence
        )
        self.emotion_analyzer = EmotionAnalyzer(
            min_confidence=self.analysis_config.emotion_confidence,
            use_gpu=self.processing_config.use_gpu
        )
        self.pose_estimator = PoseEstimator(
            min_confidence=self.analysis_config.pose_confidence,
            model_complexity=1  # Balanced accuracy/speed
        )
        self.action_classifier = ActionClassifier(
            min_confidence=self.analysis_config.action_confidence,
            history_size=self.processing_config.motion_history_size
        )
        self.anomaly_detector = AnomalyDetector(
            motion_threshold=self.analysis_config.anomaly_motion_threshold,
            emotion_change_frames=self.analysis_config.anomaly_emotion_change_frames,
            sensitivity=self.analysis_config.anomaly_sensitivity,
            history_size=self.processing_config.emotion_history_size
        )
        
        # Initialize visualization and reporting
        self.visualizer = Visualizer(self.analysis_config)
        self.report_generator = ReportGenerator(
            self.analysis_config,
            consolidation_window=self.analysis_config.consolidation_window
        )
        
        # Video properties
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30.0
        self.frame_width = 0
        self.frame_height = 0
        self.total_frames = 0
        
        # Processing state
        self.frame_count = 0
        self.processing_times = []
        
        print("Models initialized successfully!")
        
    def load_video(self, video_path: str) -> bool:
        """
        Load a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video loaded successfully
        """
        if not Path(video_path).exists():
            print(f"Error: Video file not found: {video_path}")
            return False
        
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {video_path}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Update dependencies with FPS
        self.anomaly_detector.set_fps(self.fps)
        self.report_generator.set_video_info(
            video_path, self.fps, self.total_frames
        )
        
        print(f"Video loaded: {video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Duration: {self.total_frames / self.fps:.2f}s ({self.total_frames} frames)")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        start_time = time.time()
        
        # Resize for processing if needed
        process_frame = frame
        if self.processing_config.target_width:
            scale = self.processing_config.target_width / frame.shape[1]
            if scale < 1.0:
                process_frame = cv2.resize(
                    frame, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_AREA
                )
        
        # Get frame timestamp
        timestamp = self.frame_count / self.fps
        
        # Face detection
        faces = self.face_detector.detect(process_frame, extract_faces=True)
        
        # Emotion analysis
        emotions = self.emotion_analyzer.analyze(process_frame, faces)
        
        # Pose estimation
        poses = self.pose_estimator.estimate(process_frame)
        
        # Action classification
        actions = self.action_classifier.classify(poses, process_frame.shape[:2])
        
        # Anomaly detection
        anomalies = self.anomaly_detector.detect(
            self.frame_count, faces, emotions, poses
        )
        
        # Scale detections back to original frame size if needed
        if self.processing_config.target_width and frame.shape[1] != process_frame.shape[1]:
            scale = frame.shape[1] / process_frame.shape[1]
            faces = self._scale_faces(faces, scale)
            emotions = self._scale_emotions(emotions, scale)
        
        # Record observations for report
        for emotion in emotions:
            self.report_generator.add_emotion(
                self.frame_count, timestamp,
                emotion.dominant_emotion, emotion.confidence
            )
        
        for action in actions:
            self.report_generator.add_action(
                self.frame_count, timestamp,
                action.action.value, action.confidence
            )
        
        for anomaly in anomalies:
            self.report_generator.add_anomaly(anomaly)
        
        # Calculate processing FPS
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        avg_process_time = sum(self.processing_times) / len(self.processing_times)
        processing_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0
        
        # Frame info for visualization
        frame_info = {
            'frame_number': self.frame_count,
            'timestamp': timestamp,
            'fps': self.fps,
            'processing_fps': processing_fps
        }
        
        # Visualize results
        annotated_frame = self.visualizer.draw_all(
            frame, faces, emotions, poses, actions, anomalies, frame_info
        )
        
        # Results dictionary
        results = {
            'frame_number': self.frame_count,
            'timestamp': timestamp,
            'faces': len(faces),
            'emotions': [e.dominant_emotion for e in emotions],
            'actions': [a.action.value for a in actions if a.action.value != 'unknown'],
            'anomalies': [a.anomaly_type.value for a in anomalies],
            'processing_time': process_time
        }
        
        self.frame_count += 1
        
        return annotated_frame, results
    
    def _scale_faces(self, faces, scale):
        """Scale face detections to original frame size."""
        from models.face_detector import FaceDetection
        scaled = []
        for face in faces:
            x, y, w, h = face.bbox
            scaled.append(FaceDetection(
                bbox=(int(x * scale), int(y * scale), 
                      int(w * scale), int(h * scale)),
                confidence=face.confidence,
                landmarks=[(int(lx * scale), int(ly * scale)) 
                          for lx, ly in (face.landmarks or [])],
                face_image=face.face_image
            ))
        return scaled
    
    def _scale_emotions(self, emotions, scale):
        """Scale emotion bboxes to original frame size."""
        from models.emotion_analyzer import EmotionResult
        scaled = []
        for emotion in emotions:
            x, y, w, h = emotion.bbox
            scaled.append(EmotionResult(
                dominant_emotion=emotion.dominant_emotion,
                confidence=emotion.confidence,
                all_emotions=emotion.all_emotions,
                bbox=(int(x * scale), int(y * scale),
                      int(w * scale), int(h * scale))
            ))
        return scaled
    
    def run(self, show_summary_panel: bool = True) -> str:
        """
        Run the video analysis with real-time visualization.
        
        Args:
            show_summary_panel: Whether to show live summary panel
            
        Returns:
            Path to generated report
        """
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("No video loaded. Call load_video() first.")
        
        # Create window
        window_name = "Video Analysis"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nStarting video analysis...")
        print("Press 'q' to quit, 'p' to pause, 's' to save snapshot")
        print("-" * 50)
        
        paused = False
        frame_skip = self.processing_config.process_every_n_frames
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\nEnd of video reached.")
                    break
                
                # Skip frames for performance
                if self.frame_count % frame_skip != 0:
                    self.frame_count += 1
                    continue
                
                # Process frame
                annotated_frame, results = self.process_frame(frame)
                
                # Add summary panel if requested
                if show_summary_panel:
                    stats = self.report_generator.get_live_stats()
                    summary_panel = self.visualizer.create_summary_panel(
                        stats, panel_height=annotated_frame.shape[0]
                    )
                    annotated_frame = np.hstack([annotated_frame, summary_panel])
                
                # Display
                cv2.imshow(window_name, annotated_frame)
                
                # Progress indicator every 100 frames
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames} frames)")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nAnalysis stopped by user.")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):
                snapshot_path = f"snapshot_{self.frame_count}.png"
                cv2.imwrite(snapshot_path, annotated_frame)
                print(f"Snapshot saved: {snapshot_path}")
        
        # Cleanup
        cv2.destroyAllWindows()
        
        # Generate report
        print("\nGenerating analysis report...")
        report_path = self._generate_report()
        
        return report_path
    
    def _generate_report(self) -> str:
        """Generate and save the analysis report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = f"analysis_report_{timestamp}.json"
        
        saved_path = self.report_generator.save_report(report_path)
        
        # Print summary
        report = self.report_generator.generate_report()
        summary = report['summary']
        
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Observations: {summary['total_observations']}")
        print(f"  - Positive: {summary['positive_count']}")
        print(f"  - Negative: {summary['negative_count']}")
        print(f"  - Neutral: {summary['neutral_count']}")
        print(f"  - Anomalies: {summary['anomaly_count']}")
        print("-" * 50)
        print(f"Emotions Detected: {summary['emotions_detected']}")
        print(f"Actions Detected: {summary['actions_detected']}")
        print("=" * 50)
        print(f"\nReport saved to: {saved_path}")
        
        return saved_path
    
    def release(self):
        """Release all resources."""
        if self.cap:
            self.cap.release()
        self.face_detector.release()
        self.pose_estimator.release()
        cv2.destroyAllWindows()
