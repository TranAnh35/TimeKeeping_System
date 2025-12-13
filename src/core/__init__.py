# src/core/__init__.py
"""
Core modules - Infrastructure & Configuration.

- settings: Unified configuration
- camera: Camera management
- tflite_helper: TFLite interpreter helper
- model_factory: Factory for detector/recognizer
"""

from .settings import settings, Settings
from .camera import CameraManager, CameraConfig, create_camera
from .tflite_helper import get_interpreter
from .model_factory import create_detector, create_recognizer

__all__ = [
    'settings',
    'Settings',
    'CameraManager',
    'CameraConfig', 
    'create_camera',
    'get_interpreter',
    'create_detector',
    'create_recognizer',
]
