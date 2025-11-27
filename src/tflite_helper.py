# src/tflite_helper.py
"""
Helper module để load TFLite interpreter.
Tự động chọn giữa tensorflow.lite và tflite_runtime.
Giúp code chạy được trên cả PC (TensorFlow) và Pi (tflite-runtime).
"""

def get_interpreter(model_path):
    """
    Tạo TFLite Interpreter từ model path.
    Thử tflite_runtime trước (nhẹ hơn), fallback sang tensorflow.
    """
    try:
        # Thử dùng tflite_runtime (nhẹ, phù hợp Pi)
        from tflite_runtime.interpreter import Interpreter
        print(f"[TFLite] Sử dụng tflite_runtime")
        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    
    try:
        # Fallback sang tensorflow đầy đủ
        import tensorflow as tf
        print(f"[TFLite] Sử dụng tensorflow.lite")
        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        pass
    
    raise ImportError(
        "Không tìm thấy TFLite interpreter!\n"
        "Cài đặt một trong hai:\n"
        "  - pip install tflite-runtime  (nhẹ, cho Pi)\n"
        "  - pip install tensorflow       (đầy đủ, cho PC)"
    )
