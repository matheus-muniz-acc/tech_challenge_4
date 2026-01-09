"""
Video Analysis System - Main Entry Point

A computer vision system for analyzing videos with:
- Face detection and emotion analysis
- Action detection (handshakes, standing, sitting, waving)
- Anomaly detection (jarring movements, rapid emotion changes)
- Real-time visualization
- JSON report generation

Usage:
    python main.py <video_path> [options]
    
Example:
    python main.py video.mp4
    python main.py video.mp4 --no-display --confidence 0.7
"""

import argparse
import sys
from pathlib import Path

from config import AnalysisConfig, ProcessingConfig
from video_processor import VideoProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Analysis System - Face, Emotion, Action & Anomaly Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py video.mp4
    python main.py video.mp4 --confidence 0.7
    python main.py video.mp4 --no-summary --skip-frames 3
    python main.py video.mp4 --sensitivity 0.7 --output report.json
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.6,
        help='Minimum confidence threshold for detections (0.0-1.0, default: 0.6)'
    )
    
    parser.add_argument(
        '--sensitivity', '-s',
        type=float,
        default=0.5,
        help='Anomaly detection sensitivity (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=2,
        help='Process every N frames for performance (default: 2)'
    )
    
    parser.add_argument(
        '--no-summary',
        action='store_true',
        help='Disable live summary panel'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output path for the analysis report (default: auto-generated)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Processing width for performance (default: 640, 0 for original)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate video path
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    if not video_path.is_file():
        print(f"Error: Not a file: {video_path}")
        sys.exit(1)
    
    # Create configurations
    analysis_config = AnalysisConfig(
        face_confidence=args.confidence,
        emotion_confidence=args.confidence,
        pose_confidence=args.confidence,
        action_confidence=max(args.confidence, 0.65),  # Actions need slightly higher
        anomaly_sensitivity=args.sensitivity
    )
    
    processing_config = ProcessingConfig(
        process_every_n_frames=args.skip_frames,
        target_width=args.width if args.width > 0 else None,
        use_gpu=not args.no_gpu
    )
    
    print("=" * 60)
    print("VIDEO ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Anomaly sensitivity: {args.sensitivity}")
    print(f"Frame skip: {args.skip_frames}")
    print(f"GPU enabled: {not args.no_gpu}")
    print("=" * 60)
    
    # Initialize processor
    processor = VideoProcessor(analysis_config, processing_config)
    
    # Load video
    if not processor.load_video(str(video_path)):
        sys.exit(1)
    
    try:
        # Run analysis
        report_path = processor.run(show_summary_panel=not args.no_summary)
        
        # Rename report if custom output specified
        if args.output:
            import shutil
            shutil.move(report_path, args.output)
            report_path = args.output
            print(f"Report saved to: {report_path}")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        raise
    finally:
        # Cleanup
        processor.release()
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
