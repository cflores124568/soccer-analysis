# Soccer Analysis System

A comprehensive computer vision system for analyzing soccer matches using YOLO object detection and advanced tracking algorithms. Tracks players, referees, and ball movement while calculating real-time statistics including speed, distance, ball possession, and team control percentages.

## Features

- **YOLO-based Object Detection**: Uses YOLOv8 for accurate player, referee, and ball detection
- **Multi-Object Tracking**: ByteTrack algorithm maintains consistent player identities across frames
- **Camera Movement Compensation**: Adjusts tracking coordinates to account for camera pan/tilt/zoom
- **Team Classification**: Automatic team assignment using K-means clustering on jersey colors
- **Ball Possession Analysis**: Real-time ball possession tracking with team control statistics
- **Speed & Distance Metrics**: Calculate player speeds (km/h) and distances covered using perspective transformation
- **Ball Trajectory Interpolation**: Fills gaps in ball detection using temporal interpolation
- **Performance Caching**: Stub system for caching processing results to speed up re-runs
- **Annotated Video Output**: Professional overlay graphics with all statistics and tracking data

## Installation

### Requirements
- Python 3.8+
- YOLO model weights file

### Dependencies
```bash
pip install ultralytics supervision opencv-python numpy pandas scikit-learn pickle
```

## How It Works

### 1. Object Detection & Tracking
- **YOLO Detection**: Uses YOLOv8 for detecting players, referees, and ball in each frame
- **ByteTrack**: Maintains consistent tracking IDs across frames, handling occlusions and re-entries
- **Batch Processing**: Processes frames in batches of 20 to optimize GPU memory usage
- **Goalkeeper Normalization**: Treats goalkeepers as regular players for unified tracking

### 2. Camera Movement Compensation
- Detects camera pan, tilt, and zoom movements between frames
- Adjusts all tracking coordinates to maintain spatial consistency
- Essential for accurate speed and distance calculations

### 3. Perspective Transformation
- Converts pixel coordinates to real-world field measurements
- Enables accurate speed (km/h) and distance (meters) calculations
- Accounts for camera angle and field perspective

### 4. Team Assignment
- Extracts jersey colors from player bounding boxes
- Uses K-means clustering to separate teams based on color
- Assigns team colors and maintains consistency across frames
- Handles goalkeeper classification

### 5. Ball Possession Analysis
- Tracks ball position with temporal interpolation for missing detections
- Assigns possession to nearest player within threshold distance
- Calculates real-time team ball control percentages
- Maintains possession state when assignment is ambiguous

### 6. Performance Optimization
- **Stub System**: Caches processing results to disk for faster re-runs
- **Batch Processing**: Optimizes memory usage for large video files
- **Interpolation**: Fills detection gaps without re-processing entire video

## Configuration

### Key Parameters
- **Batch Size**: `batch_size = 20` in `detect_frames()` - adjust based on GPU memory
- **Detection Confidence**: `conf = 0.1` - lower values detect more objects but may include false positives
- **Ball Assignment**: Modify distance threshold in `PlayerBallAssigner`
- **Speed Calculation**: Adjust frame window in `SpeedAndDistance_Estimator`

## Output Features

The system generates annotated videos showing:
- **Player Tracking**: Colored ellipses with track IDs for each player
- **Team Identification**: Team-colored annotations and player numbers
- **Ball Tracking**: Green triangle indicator following ball movement
- **Possession Indicators**: Red triangles above players with ball possession
- **Referee Tracking**: Yellow ellipses for referee positions
- **Real-time Statistics**: 
  - Team ball control percentages
  - Player speeds (km/h)
  - Distance covered (meters)
  - Camera movement indicators

## Model Requirements

- **YOLO Model**: Trained to detect classes: `player`, `goalkeeper`, `referee`, `ball`
- **Format**: PyTorch (.pt) format compatible with Ultralytics
- **Performance**: Model should achieve good balance between speed and accuracy for real-time analysis

## Dependencies

```txt
ultralytics>=8.0.0
supervision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
```
