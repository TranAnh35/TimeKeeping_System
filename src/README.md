# Source Code Structure

```
src/
├── main.py                      # Application entry point
├── __init__.py                  # Package exports
│
├── config/                      # Configuration
│   ├── config.json              # Runtime settings (editable)
│   └── README.md                # Settings documentation
│
├── core/                        # Core infrastructure
│   ├── settings.py              # Configuration loader (singleton)
│   ├── camera.py                # Camera management
│   ├── tflite_helper.py         # TFLite interpreter helper
│   └── model_factory.py         # Factory for detector/recognizer
│
├── data/                        # Data layer
│   └── database.py              # SQLite attendance database
│
├── web/                         # Web server
│   ├── server.py                # Flask dashboard
│   └── management.py            # Employee management API
│
├── detect/                      # Face detection (INT8)
│   └── detect.py                # UltraLight detector
│
├── recognition/                 # Face recognition (INT8)
│   └── recognition.py           # MobileFaceNet recognizer
│
└── processing/                  # Frame processing
    ├── attendance.py            # Attendance tracking logic
    ├── display.py               # UI overlay rendering
    └── frame_skip.py            # Adaptive frame skipping
```

## Quick Start

```python
# Run application
python -m src.main

# Import modules
from src import create_detector, create_recognizer, settings

detector = create_detector()
recognizer = create_recognizer()
```

## Configuration

Edit `src/config/config.json` to change settings. See `src/config/README.md` for details.

## Web Interface

- Dashboard: `http://<IP>:5000`
- Management: `http://<IP>:5000/manage`

## Models (INT8 optimized)

- Detection: `models/detection/version-RFB-320_int8_without_postprocessing.tflite`
- Recognition: `models/recognition/MobileFaceNet_int8.tflite` (128-dim embeddings)

## Database

- `face_db.pkl`: Face embeddings
- `attendance.db`: Attendance records
