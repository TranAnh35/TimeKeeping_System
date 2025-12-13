# src/recognition/__init__.py
"""
Face Recognition module - INT8 optimized.

- Model: MobileFaceNet_int8.tflite
- Embedding: 128-dim
- Database: face_db.pkl
"""

from .recognition import FaceRecognizer, get_recognizer

__all__ = [
    'FaceRecognizer',
    'get_recognizer',
]
