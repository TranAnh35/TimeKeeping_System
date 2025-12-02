"""
Configuration loader for TimeKeeping_System.

Loads config from `config/config.json` if present and applies defaults.
This keeps runtime options centralized and editable without changing source code.
"""
import os
import json
import platform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.json')

IS_PI = platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")

DEFAULTS = {
    "COOLDOWN_SECONDS": 300,
    "HOLD_TIME_SECONDS": 1.5,
    "ENABLE_WEB_SERVER": True,
    "WEB_PORT": 5000,
    "ENABLE_ANTISPOOF": False,
    "RECOGNITION_THRESHOLD": 0.55,
    "FORCE_GUI_MODE": False,
    "ENABLE_MIDNIGHT_CHECKOUT": True,
    # If None, fallback to auto-detect: True on Pi, False otherwise
    "LOW_MEMORY_MODE": None,
    "CAMERA_WIDTH": None,
    "CAMERA_HEIGHT": None,
    "GC_INTERVAL": 900,
    "ENABLE_ADAPTIVE_SKIP": None,
    "TARGET_PROCESS_TIME": 0.15,
    "MIN_FRAME_SKIP": 1,
    "MAX_FRAME_SKIP": 5,
    "DEFAULT_FRAME_SKIP": None,
    "TFLITE_NUM_THREADS": None,
    # Center ROI: Chỉ nhận diện khi mặt nằm trong vùng trung tâm
    "ENABLE_CENTER_ROI": True,
    "CENTER_ROI_RATIO": 0.6,  # Tỉ lệ vùng trung tâm (0.6 = 60% chiều rộng màn hình)
    # Model paths - sử dụng INT8 mặc định trên Pi để tăng tốc
    "USE_INT8_MODELS": None,  # None = auto (True trên Pi, False trên Windows)
    "DETECTION_MODEL": None,   # None = tự chọn dựa trên USE_INT8_MODELS
    "RECOGNITION_MODEL": None  # None = tự chọn dựa trên USE_INT8_MODELS
}

def _load_config(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        # If the JSON is invalid, ignore and use defaults
        return {}

# Load user config and merge with defaults
_user_cfg = _load_config(DEFAULT_CONFIG_PATH)
CONFIG = DEFAULTS.copy()
CONFIG.update(_user_cfg)

# Post-process some computed defaults
if CONFIG['LOW_MEMORY_MODE'] is None:
    CONFIG['LOW_MEMORY_MODE'] = IS_PI

if CONFIG['ENABLE_ADAPTIVE_SKIP'] is None:
    CONFIG['ENABLE_ADAPTIVE_SKIP'] = IS_PI

if CONFIG['DEFAULT_FRAME_SKIP'] is None:
    CONFIG['DEFAULT_FRAME_SKIP'] = 2 if IS_PI else 1

# If specific camera sizes are not provided, set based on LOW_MEMORY_MODE
if CONFIG['CAMERA_WIDTH'] is None or CONFIG['CAMERA_HEIGHT'] is None:
    if CONFIG['LOW_MEMORY_MODE']:
        CONFIG['CAMERA_WIDTH'] = 320
        CONFIG['CAMERA_HEIGHT'] = 240
    else:
        CONFIG['CAMERA_WIDTH'] = 640
        CONFIG['CAMERA_HEIGHT'] = 480

if CONFIG['TFLITE_NUM_THREADS'] is None:
    CONFIG['TFLITE_NUM_THREADS'] = 2 if IS_PI else 4

# INT8 models: auto-enable on Pi for better performance
if CONFIG['USE_INT8_MODELS'] is None:
    CONFIG['USE_INT8_MODELS'] = IS_PI

# Set model paths based on INT8 preference
if CONFIG['DETECTION_MODEL'] is None:
    if CONFIG['USE_INT8_MODELS']:
        CONFIG['DETECTION_MODEL'] = "models/detection/version-RFB-320_int8.tflite"
    else:
        CONFIG['DETECTION_MODEL'] = "models/detection/version-RFB-320_without_postprocessing.tflite"

if CONFIG['RECOGNITION_MODEL'] is None:
    if CONFIG['USE_INT8_MODELS']:
        CONFIG['RECOGNITION_MODEL'] = "models/recognition/MobileFaceNet_int8.tflite"
    else:
        CONFIG['RECOGNITION_MODEL'] = "models/recognition/MobileFaceNet.tflite"

def get(key, default=None):
    return CONFIG.get(key, default)

__all__ = ['CONFIG', 'get']
