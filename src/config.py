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
    "GC_INTERVAL": 30,
    "ENABLE_ADAPTIVE_SKIP": None,
    "TARGET_PROCESS_TIME": 0.15,
    "MIN_FRAME_SKIP": 1,
    "MAX_FRAME_SKIP": 5,
    "DEFAULT_FRAME_SKIP": None
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

def get(key, default=None):
    return CONFIG.get(key, default)

__all__ = ['CONFIG', 'get']
