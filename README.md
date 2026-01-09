# Video Analysis System

A real-time computer vision system for analyzing videos with face detection, emotion analysis, action detection, and anomaly detection. Built with GPU acceleration for optimal performance.

## Features

- **Face Detection**: MediaPipe-based face detection with landmark extraction
- **Emotion Analysis**: Real-time facial emotion recognition (happy, sad, angry, fear, surprise, disgust, neutral)
- **Action Detection**: Detects handshakes, standing, sitting, and hand waving gestures
- **Anomaly Detection**: Identifies jarring movements, rapid emotion changes, and detection discontinuities
- **Real-time Visualization**: Live overlay of detections with summary panel
- **Report Generation**: JSON reports with time-range consolidated observations

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (RTX series recommended)
- Windows/Linux/macOS

## Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA** (for RTX GPUs):
   ```bash
   # For CUDA 12.x (RTX 40/50 series)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python main.py video.mp4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--confidence, -c` | Detection confidence threshold (0.0-1.0) | 0.6 |
| `--sensitivity, -s` | Anomaly detection sensitivity (0.0-1.0) | 0.5 |
| `--skip-frames` | Process every N frames | 2 |
| `--no-summary` | Disable live summary panel | False |
| `--no-gpu` | Disable GPU acceleration | False |
| `--output, -o` | Custom output path for report | Auto |
| `--width` | Processing width (0 for original) | 640 |

### Examples

```bash
# Higher confidence threshold
python main.py video.mp4 --confidence 0.7

# More sensitive anomaly detection
python main.py video.mp4 --sensitivity 0.8

# Faster processing (skip more frames)
python main.py video.mp4 --skip-frames 4

# Custom report output
python main.py video.mp4 --output my_analysis.json
```

### Keyboard Controls (During Analysis)

- **Q**: Quit analysis
- **P**: Pause/Resume
- **S**: Save snapshot

## Output Report

The system generates a JSON report with the following structure:

```json
{
  "metadata": {
    "generated_at": "2024-01-01T12:00:00",
    "video_path": "video.mp4",
    "video_duration_seconds": 120.5,
    "total_frames": 3615,
    "fps": 30.0
  },
  "summary": {
    "total_observations": 45,
    "positive_count": 12,
    "negative_count": 3,
    "neutral_count": 30,
    "anomaly_count": 2,
    "emotions_detected": {"happy": 100, "neutral": 500},
    "actions_detected": {"standing": 200, "wave": 10}
  },
  "positive_observations": [...],
  "negative_observations": [...],
  "neutral_observations": [...],
  "anomalies": [...]
}
```

### Observation Categories

**Positive:**
- Emotions: happy, surprise
- Actions: handshake, wave

**Negative:**
- Emotions: angry, fear, sad, disgust

**Neutral:**
- Emotions: neutral
- Actions: standing, sitting

**Anomalies:**
- Jarring movement
- Rapid emotion change
- Undetectable face
- Pose discontinuity

## Architecture

```
video_processor.py      # Main processing pipeline
├── models/
│   ├── face_detector.py      # MediaPipe face detection
│   ├── emotion_analyzer.py   # FER emotion recognition
│   ├── pose_estimator.py     # MediaPipe pose estimation
│   ├── action_classifier.py  # Action classification from poses
│   └── anomaly_detector.py   # Anomaly detection
├── visualization.py    # Real-time visualization
├── report_generator.py # Report generation with consolidation
├── config.py          # Configuration settings
└── main.py            # CLI entry point
```

## Configuration

Edit `config.py` to customize:

- Confidence thresholds
- Emotion/action categorization
- Anomaly sensitivity
- Visualization colors
- Processing parameters

## Performance Tips

1. **Increase `--skip-frames`** for faster processing (trades accuracy)
2. **Reduce `--width`** to process at lower resolution
3. **Ensure CUDA is properly installed** for GPU acceleration
4. Close other GPU-intensive applications

## Troubleshooting

### "CUDA not available"
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Verify CUDA installation: `nvidia-smi`

### Low FPS
- Increase `--skip-frames` value
- Reduce processing `--width`
- Check GPU utilization with `nvidia-smi`

### Missing detections
- Lower `--confidence` threshold
- Ensure good lighting in video
- Check if faces/bodies are clearly visible

## License

MIT License
