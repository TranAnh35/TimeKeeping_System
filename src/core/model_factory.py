# src/model_factory.py
"""
Factory module đơn giản để tạo detector và recognizer.
Chỉ sử dụng INT8 models tối ưu cho Raspberry Pi.

Usage:
    from src.model_factory import create_detector, create_recognizer
    
    detector = create_detector()
    recognizer = create_recognizer()
"""
import os
import sys

# Model paths
DEFAULT_DETECTION_MODEL = "models/detection/version-RFB-320_int8_without_postprocessing.tflite"
DEFAULT_RECOGNITION_MODEL = "models/recognition/MobileFaceNet_int8.tflite"
DEFAULT_DB_PATH = "face_db.pkl"


def create_detector(model_path=None, conf_threshold=0.6):
    """
    Tạo Face Detector INT8.
    
    Args:
        model_path: Đường dẫn model (None = default)
        conf_threshold: Ngưỡng confidence
        
    Returns:
        UltraLightFaceDetector instance
    """
    try:
        from .detect.detect import UltraLightFaceDetector
    except ImportError:
        from detect.detect import UltraLightFaceDetector
    
    if model_path is None:
        model_path = DEFAULT_DETECTION_MODEL
    
    print(f"[Detector] Model: {model_path}")
    return UltraLightFaceDetector(model_path=model_path, conf_threshold=conf_threshold)


def create_recognizer(model_path=None, db_path=None):
    """
    Tạo Face Recognizer INT8.
    
    Args:
        model_path: Đường dẫn model (None = default)
        db_path: Đường dẫn database (None = default)
        
    Returns:
        FaceRecognizer instance
    """
    try:
        from .recognition.recognition import FaceRecognizer
    except ImportError:
        from recognition.recognition import FaceRecognizer
    
    if model_path is None:
        model_path = DEFAULT_RECOGNITION_MODEL
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    print(f"[Recognizer] Model: {model_path}")
    print(f"[Recognizer] Database: {db_path}")
    return FaceRecognizer(model_path=model_path, db_path=db_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Model Factory - INT8 Only")
    print("=" * 50)
    
    detector = create_detector()
    recognizer = create_recognizer()
    
    print(f"\nRecognizer embedding dim: {recognizer.embedding_dim}")
    print("✅ Factory OK")
