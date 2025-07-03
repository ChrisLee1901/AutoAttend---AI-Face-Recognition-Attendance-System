# AutoAttend - AI Face Recognition Attendance System

An intelligent attendance management system using advanced face recognition technology with real-time detection and tracking capabilities.

## 🌟 Features

- **Real-time Face Recognition**: Live face detection and recognition using MTCNN and DeepFace
- **Multiple Model Support**: Supports Facenet512, VGG-Face, ArcFace, OpenFace, DeepFace, and SFace models
- **GPU Acceleration**: Automatic GPU detection with CUDA support and CPU fallback
- **Advanced Face Tracking**: Multi-face tracking with smoothing algorithms and persistence
- **Modern GUI Interface**: Dark-themed desktop application with real-time camera preview
- **Command Line Interface**: Full CLI support for headless operations
- **Automatic Attendance Logging**: Real-time attendance recording with CSV export
- **Statistics Dashboard**: Live attendance statistics and analytics
- **Batch Processing**: Efficient face alignment and embedding generation

## 📋 System Requirements

### Hardware Requirements
- **Camera**: USB webcam or built-in camera
- **GPU** (Optional): NVIDIA GPU with CUDA support for enhanced performance
- **RAM**: Minimum 4GB (8GB recommended for optimal performance)
- **Storage**: 2GB free space for face data and models

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **CUDA** (Optional): For GPU acceleration

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# Download the project files to your local machine
# Ensure ver.py is in your working directory
```

### 2. Install Required Dependencies
```bash
pip install opencv-python
pip install mtcnn
pip install deepface
pip install tensorflow
pip install scikit-learn
pip install scipy
pip install pillow
pip install numpy
pip install pandas
```

### 3. GPU Setup (Optional but Recommended)
For NVIDIA GPU acceleration:
```bash
pip install tensorflow-gpu
# Ensure CUDA and cuDNN are properly installed
```

## 📁 Project Structure

The system automatically creates the following directory structure:

```
AutoAttend/
├── ver.py                          # Main application file
├── README.md                       # This documentation
├── camera_capture_frames/          # Raw captured face images
│   ├── Name1_ID1/
│   │   ├── frame_0001.jpg
│   │   ├── frame_0002.jpg
│   │   └── ...
│   └── Name2_ID2/
│       └── ...
├── Align/                          # Aligned face images (112x112)
│   ├── Name1_ID1/
│   ├── Name2_ID2/
│   └── ...
├── Embedded/                       # Face embeddings database
│   ├── face_database_Name1_ID1.npz
│   ├── face_database_Name2_ID2.npz
│   └── ...
└── logs/
    └── attendance.csv              # Attendance records
```

## 🚀 Quick Start

### Running the Application

#### GUI Mode (Default)
```bash
python ver.py
```
The modern GUI interface will launch automatically with:
- Dark theme design
- Real-time camera preview
- Control panel for all operations
- Activity logs and statistics

#### CLI Mode
To use command line interface, modify the last line in `ver.py`:
```python
if __name__ == "__main__":
    main_cli()  # Change from main_ui() to main_cli()
```

Then run:
```bash
python ver.py
```

## 📖 Complete Usage Guide

### Step 1: Register New Users

#### Via GUI:
1. Launch the application: `python ver.py`
2. In the **"📝 User Information"** section:
   - Enter **Full Name** (e.g., "John Doe")
   - Enter **ID/Student ID** (e.g., "SE123456")
3. Click **"📸 Capture Registration Photos"**
4. Position your face in the camera view
5. Press **'Q'** when sufficient photos are captured (system saves ~5 frames per second)
6. Photos are automatically saved to `camera_capture_frames/Name_ID/`

#### Via CLI:
1. Select option `1` - Capture Frames
2. Enter name and ID when prompted
3. Look at the camera and press 'Q' to stop
4. System automatically saves captured frames

### Step 2: Process Face Data

#### 2.1 Face Alignment
**Purpose**: Standardizes face images to 112x112 pixels for consistent processing

**Via GUI**: Click **"🔄 Align Faces"**
**Via CLI**: Select option `2`

**Process**:
- Uses MTCNN detector for face detection
- Extracts facial landmarks (eyes, nose, mouth)
- Applies affine transformation for alignment
- Saves aligned faces to `Align/` directory

#### 2.2 Generate Face Embeddings
**Purpose**: Creates mathematical representations (embeddings) for face recognition

**Via GUI**: Click **"🧠 Create Embeddings"**
**Via CLI**: Select option `3`

**Process**:
- Uses DeepFace with selected model (default: Facenet512)
- Processes aligned faces in batches of 5
- Normalizes embeddings for consistent comparison
- Saves embeddings as `.npz` files in `Embedded/` directory

### Step 3: Real-time Face Recognition

#### Via GUI:
1. Click **"🔍 Start Recognition"**
2. The camera view shows:
   - **Green boxes**: Recognized faces with names
   - **Orange boxes**: Unknown faces
   - **Yellow boxes**: Faces being processed
   - **FPS counter**: Real-time performance
   - **Face count**: Number of detected faces
3. Click **"⏹️ Stop Recognition"** to end

#### Via CLI:
1. Select option `4`
2. Press 'Q' to quit recognition mode

**Recognition Features**:
- **Face Tracking**: Maintains identity across frames
- **Smoothing**: Reduces bounding box jitter
- **Multi-face Support**: Can track multiple faces (optimized for largest face)
- **Automatic Logging**: Records attendance when recognition confidence is high
- **Real-time Performance**: Optimized for live video processing

## ⚙️ Configuration

### Model Selection
Edit the configuration in `ver.py`:

```python
# Available models
MODEL_NAME = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, ArcFace, SFace

# Recognition sensitivity (lower = stricter)
THRESHOLD = 0.25  # Range: 0.1 (very strict) to 0.6 (very lenient)

# Camera settings
CAMERA_INDEX = 0  # Change to 1, 2, etc. for external cameras
```

### Directory Paths
Modify paths if needed:
```python
FRAME_SAVE_ROOT = Path("camera_capture_frames")
ALIGN_SAVE_ROOT = Path("Align")
EMBEDDED_DIR = Path("Embedded")
ATTENDANCE_LOG_PATH = Path("logs/attendance.csv")
```

### Performance Tuning
```python
# Face detection frequency (higher = faster but less accurate)
detection_interval = 5  # Process every 5th frame

# Tracking parameters
max_distance_threshold = 100  # Face matching distance
max_frames_missing = 10      # Frames before losing track
smoothing_factor = 0.3       # Smoothing strength (0-1)
```

## 🎨 GUI Features

### Modern Interface Components
- **Header**: App title with GPU/CPU status indicator
- **Control Panel**: User input and function buttons
- **Camera View**: Live preview with recognition overlays
- **Statistics Panel**: Real-time attendance data
- **Activity Log**: Color-coded operation logs
- **Status Bar**: Current operation and timestamp

### Control Functions
- **📸 Capture Registration Photos**: Record new user faces
- **🔄 Align Faces**: Process and standardize face images
- **🧠 Create Embeddings**: Generate face recognition data
- **🔍 Start/Stop Recognition**: Toggle real-time attendance
- **📊 Export Report**: Save attendance data
- **🗑️ Clear Data**: Remove all stored data

## 📊 Attendance Data

### CSV Format
Attendance records are saved in the following format:
```csv
Name,Timestamp,Status
John_Doe_SE123456,2024-01-15 09:30:45,Present
Jane_Smith_SE789012,2024-01-15 09:31:20,Present
```

### Automatic Logging Rules
- Recognition must occur for 3+ consecutive frames
- Each person logged only once per session
- Timestamp format: YYYY-MM-DD HH:MM:SS
- Status: Always "Present" for successful recognition

## 🔧 Technical Details

### Face Detection Pipeline
1. **MTCNN Detection**: Detects faces and facial landmarks
2. **Largest Face Selection**: Focuses on primary subject
3. **Face Tracking**: Maintains identity across frames
4. **Landmark Extraction**: Gets eye, nose, mouth positions
5. **Affine Transformation**: Aligns face to standard template
6. **Embedding Generation**: Creates 512-dimensional vector
7. **Similarity Comparison**: Matches against stored embeddings

### Performance Optimization
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Batch Processing**: Efficient embedding generation
- **Frame Skipping**: Processes every 5th frame for real-time performance
- **Memory Management**: Careful resource allocation and cleanup
- **Smart Tracking**: Reduces unnecessary face re-recognition

### Model Information
- **Default Model**: Facenet512 (512-dimensional embeddings)
- **Input Size**: 112x112 RGB images
- **Detection Backend**: MTCNN for face detection
- **Similarity Metric**: Cosine distance
- **Normalization**: L2 normalization for embeddings

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Camera Not Working
```bash
# Check available cameras
# Try different CAMERA_INDEX values (0, 1, 2, ...)
```
**Solutions**:
- Ensure camera permissions are granted
- Close other applications using the camera
- Try different camera indices
- Check camera drivers

#### GPU Not Detected
```bash
# Verify CUDA installation
nvidia-smi

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
**Solutions**:
- Install CUDA and cuDNN
- Reinstall tensorflow-gpu
- System will automatically fallback to CPU

#### Poor Recognition Accuracy
**Causes and Solutions**:
- **Lighting**: Ensure good, even lighting
- **Angles**: Capture faces from multiple angles
- **Quality**: Use higher resolution camera
- **Threshold**: Adjust `THRESHOLD` value (lower = stricter)
- **Model**: Try different models (ArcFace, SFace)

#### Memory Issues
**Solutions**:
- Reduce batch size in embedding generation
- Close other applications
- Use CPU mode: Set `CAMERA_INDEX = -1` temporarily
- Restart the application

#### Model Loading Errors
**Solutions**:
- Ensure internet connection for initial model download
- Check disk space for model storage
- Restart application to retry initialization

## 🚀 Advanced Usage

### Custom Integration

#### Database Integration
Replace CSV logging with database:
```python
# Modify the attendance logging section in run_real_time_recognition()
# Replace CSV writer with database insert operations
```

#### API Integration
Add web service capabilities:
- Create REST endpoints for recognition
- Add webhook notifications
- Implement remote monitoring

#### Multi-Camera Setup
For multiple camera locations:
```python
# Modify CAMERA_INDEX for each instance
# Run multiple instances with different configurations
```

### Performance Monitoring

#### System Diagnostics
```python
# Check system status
python -c "
import cv2, tensorflow as tf, numpy as np
print(f'OpenCV: {cv2.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
"
```

#### Performance Metrics
- **FPS**: Displayed in real-time during recognition
- **Memory Usage**: Monitor via system tools
- **GPU Utilization**: Check with `nvidia-smi`
- **Recognition Accuracy**: Adjust threshold based on results

## 📞 Support and Troubleshooting

### Error Messages and Solutions

#### "Cannot open camera"
- Check camera connections
- Verify camera permissions
- Try different camera indices

#### "No embedding files found"
- Run Step 2 (face alignment and embedding generation)
- Check `Embedded/` directory for `.npz` files

#### "GPU memory error"
- Close other GPU applications
- Reduce batch size
- Use CPU mode

#### "Model initialization failed"
- Check internet connection
- Ensure sufficient disk space
- Restart application

### Performance Optimization Tips

1. **Hardware**: Use SSD for faster file operations
2. **Camera**: Higher resolution cameras improve accuracy
3. **Lighting**: Consistent lighting improves recognition
4. **Background**: Simple backgrounds reduce false detections
5. **Distance**: Maintain 1-3 feet from camera for optimal results

## 📄 License and Credits

This project uses the following open-source libraries:
- **OpenCV**: Computer vision operations
- **MTCNN**: Face detection and landmark extraction
- **DeepFace**: Face recognition models
- **TensorFlow**: Machine learning backend
- **scikit-learn**: Data preprocessing
- **NumPy**: Numerical operations

## 🤝 Contributing

To modify or extend the system:

1. **Adding New Models**: Modify the `MODEL_NAME` configuration
2. **Custom UI**: Edit the `AutoAttendUI` class
3. **New Features**: Add functions to the core module
4. **Database Support**: Replace CSV operations with database calls

## 📧 Technical Support

For technical issues:
1. Check this documentation first
2. Verify system requirements
3. Run diagnostic commands
4. Check error logs in the application

---

**AutoAttend** - Intelligent attendance management with advanced face recognition technology! 🎯
