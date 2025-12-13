# src/detect/__init__.py
"""
Face Detection module - INT8 optimized.

Exports:
- UltraLightFaceDetector: INT8 quantized face detector
- detect_faces: Quick function để detect faces
"""

from .detect import UltraLightFaceDetector, detect_faces

__all__ = [
    'UltraLightFaceDetector',
    'detect_faces',
]
