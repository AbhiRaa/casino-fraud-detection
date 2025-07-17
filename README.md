# Casino Fraud Detection Prototype

A real-time face detection and recognition system for casino fraud detection using YOLOv8-face and ArcFace embeddings.

## Features

- **Real-time face detection** using YOLOv8-face (nano model)
- **Face recognition** with ArcFace 512-d embeddings (buffalo_l model)
- **Watchlist matching** with cosine similarity
- **Visual alerts** with colored bounding boxes (green for unknown, red for matches)
- **Audio alerts** for detected matches
- **Performance monitoring** with latency/FPS display
- **CSV logging** of all detection events
- **Configurable thresholds** and camera settings

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download models** (automatically handled on first run):
   - YOLOv8n-face model will be downloaded by Ultralytics
   - ArcFace buffalo_l model will be downloaded by InsightFace

## Setup

### 1. Create Sample Folder Structure

```bash
python generate_watchlist.py --create_sample --img_root ./sample_faces
```

This creates a folder structure like:
```
sample_faces/
├── john_doe/
│   └── README.txt
├── jane_smith/
│   └── README.txt
└── bob_johnson/
    └── README.txt
```

### 2. Add Face Images

For each person, add 3-5 clear face images to their folder:
```
sample_faces/
├── john_doe/
│   ├── img1.jpg
│   ├── img2.png
│   └── photo1.jpeg
├── jane_smith/
│   ├── face1.jpg
│   ├── face2.jpg
│   └── face3.png
└── bob_johnson/
    ├── headshot1.jpg
    └── headshot2.jpg
```

**Image requirements:**
- Clear, well-lit face photos
- Frontal or near-frontal poses work best
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- Multiple images per person improve accuracy

### 3. Generate Watchlist

```bash
python generate_watchlist.py --img_root ./sample_faces --out watchlist.json
```

This creates a `watchlist.json` file with averaged embeddings for each person.

### 4. Add Alert Sound (Optional)

Place an audio file in the project directory for audio alerts. The system will check for files in this priority order:
1. `siren-alert-96052.mp3` (high-priority siren alert)
2. `alert.mp3` (standard alert)
3. `alert.wav` (WAV format)

If no audio file is present, the system will use a console beep.

## Usage

### Basic Usage

```bash
python casino_proto.py
```

### Advanced Usage

```bash
python casino_proto.py --watchlist watchlist.json --threshold 0.6 --show_fps
```

### Command Line Options

- `--watchlist PATH`: Path to watchlist JSON file (default: `watchlist.json`)
- `--threshold FLOAT`: Similarity threshold for matching (default: `0.55`)
- `--cam_id INT`: Camera device ID (default: `0`)
- `--show_fps`: Show FPS instead of latency

### Controls

- **ESC**: Exit the application
- Camera window displays real-time detection results

## Output

### Visual Feedback
- **Green boxes**: Unknown faces
- **Red boxes**: Matched faces (score > threshold)
- **Labels**: Show name and similarity score
- **Performance metrics**: Latency (ms) or FPS display

### Audio Alerts
- Plays asynchronously when matches are detected
- Uses `alert.mp3` if available, otherwise console beep

### CSV Logging
All detection events are logged to `detections.csv` with columns:
- `timestamp`: Detection time
- `name`: Matched person name
- `score`: Similarity score
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`: Bounding box coordinates

## Performance

### Target Performance
- **< 150ms** per frame on laptop GPU (RTX or Apple M-series)
- **< 250ms** per frame on CPU

### Optimization Tips
1. **GPU acceleration**: Ensure CUDA is available for faster inference
2. **Camera resolution**: Default 640x480 for good balance of quality/speed
3. **Model selection**: YOLOv8n-face provides good speed/accuracy tradeoff
4. **Batch processing**: Single frame processing for real-time requirements

## Troubleshooting

### Common Issues

1. **Camera not found**:
   - Try different `--cam_id` values (0, 1, 2, etc.)
   - Check camera permissions
   - Ensure camera isn't used by other applications

2. **Slow performance**:
   - Check if GPU acceleration is working
   - Reduce camera resolution
   - Close other resource-intensive applications

3. **No faces detected**:
   - Ensure good lighting
   - Face should be clearly visible and unobstructed
   - Try adjusting camera angle

4. **Audio alerts not working**:
   - Check if `alert.mp3` exists
   - Verify audio system is working
   - Check volume settings

5. **Model download failures**:
   - Ensure internet connection
   - Check firewall/proxy settings
   - Try running with `--verbose` flag

### Dependencies Issues

If you encounter package conflicts:
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

For InsightFace installation issues on some systems:
```bash
pip install insightface --no-deps
pip install onnxruntime-gpu  # or onnxruntime for CPU
```

## File Structure

```
casino_proto/
├── casino_proto.py          # Main detection system
├── generate_watchlist.py    # Watchlist generation utility
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── watchlist.json          # Generated watchlist (after setup)
├── detections.csv          # Detection log (created during runtime)
├── alert.mp3               # Optional audio alert file
└── sample_faces/           # Sample image folder structure
    ├── person1/
    ├── person2/
    └── person3/
```

## Known Limitations

1. **Single camera input**: Currently supports one camera at a time
2. **Lighting dependency**: Performance degrades in poor lighting conditions
3. **Angle sensitivity**: Works best with frontal/near-frontal faces
4. **Processing delay**: 150-250ms latency may miss very fast movements
5. **Memory usage**: Keeps models loaded in memory for performance
6. **Network dependency**: Initial model downloads require internet

## Security Considerations

- This is a **prototype** for demonstration purposes
- In production, consider:
  - Encrypted storage of biometric data
  - Secure transmission of alerts
  - Regular model updates and retraining
  - Compliance with privacy regulations
  - Audit trails and access controls

## Future Enhancements

- Multi-camera support
- Database integration
- Web dashboard for monitoring
- Real-time streaming to security center
- Advanced analytics and reporting
- Mobile app integration
- Cloud deployment options
