# src package
"""
TimeKeeping_System - Face Detection & Recognition (INT8 Optimized)

Structure:
    src/
    ├── core/                     # Core infrastructure
    │   ├── settings.py           # Configuration
    │   ├── camera.py             # Camera management
    │   ├── tflite_helper.py      # TFLite interpreter helper
    │   └── model_factory.py      # Factory for detector/recognizer
    ├── data/                     # Data layer
    │   └── database.py           # SQLite database
    ├── web/                      # Web server
    │   ├── server.py             # Flask web server
    │   └── management.py         # Employee management API
    ├── processing/               # Processing modules
    │   ├── frame_skip.py         # Frame skip handler
    │   ├── display.py            # UI/Overlay handler
    │   └── attendance.py         # Attendance tracking logic
    ├── detect/                   # Face detection (INT8)
    │   └── detect.py             # INT8 quantized detection
    ├── recognition/              # Face recognition (INT8)
    │   └── recognition.py        # INT8 quantized recognition
    └── main.py                   # Main application

Usage:
    from src import create_detector, create_recognizer
    
    detector = create_detector()
    recognizer = create_recognizer()
"""

from .core.model_factory import create_detector, create_recognizer
from .core.settings import settings
from .detect import UltraLightFaceDetector
from .recognition import FaceRecognizer

__all__ = [
    'settings',
    'create_detector',
    'create_recognizer',
    'UltraLightFaceDetector',
    'FaceRecognizer',
]
