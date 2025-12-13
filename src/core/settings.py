# src/core/settings.py
"""
Configuration đơn giản cho TimeKeeping System.
Chỉ giữ những settings thực sự cần thiết.
"""
import os
import json
import platform
from dataclasses import dataclass, field


# === PLATFORM DETECTION ===
IS_WINDOWS = platform.system() == "Windows"
IS_PI = platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")
HAS_DISPLAY = IS_WINDOWS or os.environ.get("DISPLAY", "") != ""

# === PATHS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.json')


def _load_json_config(path: str) -> dict:
    """Load config từ JSON file, trả về {} nếu lỗi."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class Settings:
    """Configuration đơn giản - chỉ giữ settings cần thiết."""
    
    # === PLATFORM (read-only) ===
    IS_WINDOWS: bool = field(default_factory=lambda: IS_WINDOWS)
    IS_PI: bool = field(default_factory=lambda: IS_PI)
    HAS_DISPLAY: bool = field(default_factory=lambda: HAS_DISPLAY)
    BASE_DIR: str = field(default_factory=lambda: BASE_DIR)
    
    # === ATTENDANCE ===
    COOLDOWN_SECONDS: int = 300          # 5 phút giữa 2 lần check-in/out
    HOLD_TIME_SECONDS: float = 1.5       # Giữ mặt 1.5s để confirm
    RECOGNITION_THRESHOLD: float = 0.55  # Ngưỡng nhận diện
    
    # === WEB SERVER ===
    ENABLE_WEB_SERVER: bool = True
    WEB_PORT: int = 5000
    
    # === DISPLAY ===
    FORCE_GUI_MODE: bool = False
    HEADLESS_MODE: bool = False
    OVERLAY_ENABLED: bool = True
    
    # === CAMERA ===
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    
    # === PERFORMANCE ===
    GC_INTERVAL: int = 900               # GC mỗi 900 frames
    TFLITE_NUM_THREADS: int = 4
    DEFAULT_FRAME_SKIP: int = 1
    ENABLE_ADAPTIVE_SKIP: bool = False
    TARGET_PROCESS_TIME: float = 0.15
    MIN_FRAME_SKIP: int = 1
    MAX_FRAME_SKIP: int = 5
    
    # === FEATURES ===
    ENABLE_CENTER_ROI: bool = True
    CENTER_ROI_RATIO: float = 0.6
    ENABLE_MIDNIGHT_CHECKOUT: bool = True
    
    def __post_init__(self):
        """Tính toán giá trị phụ thuộc platform."""
        self._load_from_json()
        self._compute_defaults()
    
    def _load_from_json(self):
        """Load settings từ config.json nếu có."""
        config = _load_json_config(CONFIG_PATH)
        for key, value in config.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def _compute_defaults(self):
        """Tính giá trị mặc định theo platform."""
        # Camera nhỏ hơn trên Pi
        if self.IS_PI:
            if self.CAMERA_WIDTH > 320:
                self.CAMERA_WIDTH = 320
            if self.CAMERA_HEIGHT > 240:
                self.CAMERA_HEIGHT = 240
            self.TFLITE_NUM_THREADS = 2
            self.DEFAULT_FRAME_SKIP = 2
            self.ENABLE_ADAPTIVE_SKIP = True
        
        # Headless mode
        self.HEADLESS_MODE = not self.IS_WINDOWS and not self.FORCE_GUI_MODE and not self.HAS_DISPLAY
        self.OVERLAY_ENABLED = not self.HEADLESS_MODE
    
    # === PROPERTY ALIASES ===
    @property
    def tflite_num_threads(self) -> int:
        return self.TFLITE_NUM_THREADS
    
    @property
    def recognition_threshold(self) -> float:
        return self.RECOGNITION_THRESHOLD
    
    @property
    def headless_mode(self) -> bool:
        return self.HEADLESS_MODE
    
    @property
    def cooldown_seconds(self) -> int:
        return self.COOLDOWN_SECONDS
    
    @property
    def hold_time_seconds(self) -> float:
        return self.HOLD_TIME_SECONDS
    
    @property
    def camera_width(self) -> int:
        return self.CAMERA_WIDTH
    
    @property
    def camera_height(self) -> int:
        return self.CAMERA_HEIGHT
    
    @property
    def gc_interval(self) -> int:
        return self.GC_INTERVAL
    
    @property
    def enable_web_server(self) -> bool:
        return self.ENABLE_WEB_SERVER
    
    @property
    def web_port(self) -> int:
        return self.WEB_PORT


# === SINGLETON ===
settings = Settings()
