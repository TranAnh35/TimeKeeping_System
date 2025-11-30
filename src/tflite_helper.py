# src/tflite_helper.py
"""
Helper module để load TFLite interpreter.
Tự động chọn giữa tensorflow.lite và tflite_runtime.
Giúp code chạy được trên cả PC (TensorFlow) và Pi (tflite-runtime).
"""
import os
import platform

# Detect platform để tối ưu số threads
IS_PI = platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")

try:
    from .config import CONFIG as _CONFIG
except ImportError:
    try:
        from config import CONFIG as _CONFIG
    except ImportError:
        _CONFIG = None

def _resolve_thread_count():
    default_threads = 2 if IS_PI else 4
    if _CONFIG is None:
        return default_threads

    configured = _CONFIG.get('TFLITE_NUM_THREADS')
    if configured is None:
        return default_threads

    try:
        value = int(configured)
        return max(1, value)
    except (TypeError, ValueError):
        return default_threads

NUM_THREADS = _resolve_thread_count()

# Cache để tránh print nhiều lần
_logged_runtime = False

def get_interpreter(model_path, num_threads=None):
    """
    Tạo TFLite Interpreter từ model path.
    Thử tflite_runtime trước (nhẹ hơn), fallback sang tensorflow.
    
    Args:
        model_path: Đường dẫn đến file .tflite
        num_threads: Số threads cho inference (mặc định: 4 trên Pi, 2 trên PC)
    """
    global _logged_runtime
    
    if num_threads is None:
        num_threads = NUM_THREADS
    
    try:
        # Thử dùng tflite_runtime (nhẹ, phù hợp Pi)
        from tflite_runtime.interpreter import Interpreter
        if not _logged_runtime:
            print(f"[TFLite] Sử dụng tflite_runtime (threads={num_threads})")
            _logged_runtime = True
        return Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass
    
    try:
        # Fallback sang tensorflow đầy đủ
        import tensorflow as tf
        if not _logged_runtime:
            print(f"[TFLite] Sử dụng tensorflow.lite (threads={num_threads})")
            _logged_runtime = True
        return tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
    except ImportError:
        pass
    
    raise ImportError(
        "Không tìm thấy TFLite interpreter!\n"
        "Cài đặt một trong hai:\n"
        "  - pip install tflite-runtime  (nhẹ, cho Pi)\n"
        "  - pip install tensorflow       (đầy đủ, cho PC)"
    )
