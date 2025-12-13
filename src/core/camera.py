# src/camera.py
"""
Camera Manager module.

Xử lý khởi tạo và quản lý camera với retry logic.
Tối ưu cho cả Windows và Raspberry Pi.

Usage:
    from src.camera import CameraManager
    
    camera = CameraManager()
    if camera.open():
        while True:
            frame = camera.read()
            if frame is None:
                break
            # process frame...
        camera.release()
"""
import cv2
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Cấu hình camera."""
    width: int = 640
    height: int = 480
    fps: int = 15
    buffer_size: int = 1
    warmup_frames: int = 5
    max_retries: int = 3
    retry_delay: float = 2.0
    use_mjpg: bool = False  # Dùng MJPG codec (tốt cho Pi)


class CameraManager:
    """
    Quản lý camera với retry logic và tối ưu cho các platform.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        config: Optional[CameraConfig] = None,
        is_pi: bool = False
    ):
        """
        Khởi tạo CameraManager.
        
        Args:
            device_id: Camera ID (mặc định 0)
            config: CameraConfig object
            is_pi: True nếu chạy trên Raspberry Pi
        """
        self.device_id = device_id
        self.config = config or CameraConfig()
        self.is_pi = is_pi
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
        
        # Tối ưu cho Pi
        if is_pi:
            self.config.use_mjpg = True
            self.config.fps = 15
    
    def open(self) -> bool:
        """
        Mở camera với retry logic.
        
        Returns:
            True nếu thành công
        """
        for attempt in range(self.config.max_retries):
            try:
                self._cap = cv2.VideoCapture(self.device_id)
                
                if self._cap.isOpened():
                    self._configure_camera()
                    self._warmup()
                    self._is_open = True
                    
                    actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"📹 Camera opened: {actual_w}x{actual_h}")
                    
                    return True
                
            except Exception as e:
                logger.warning(f"Camera error: {e}")
            
            if attempt < self.config.max_retries - 1:
                logger.warning(
                    f"⚠️ Camera không sẵn sàng, thử lại "
                    f"({attempt + 1}/{self.config.max_retries})..."
                )
                time.sleep(self.config.retry_delay)
        
        logger.error("❌ Không thể kết nối camera!")
        return False
    
    def _configure_camera(self):
        """Cấu hình camera settings."""
        if self._cap is None:
            return
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        
        if self.is_pi:
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            if self.config.use_mjpg:
                self._cap.set(
                    cv2.CAP_PROP_FOURCC, 
                    cv2.VideoWriter_fourcc(*'MJPG')
                )
    
    def _warmup(self):
        """Đọc vài frame đầu để camera ổn định."""
        if self._cap is None:
            return
        
        for _ in range(self.config.warmup_frames):
            self._cap.grab()
    
    def read(self) -> Optional['cv2.Mat']:
        """
        Đọc một frame từ camera.
        
        Returns:
            Frame (numpy array) hoặc None nếu lỗi
        """
        if not self._is_open or self._cap is None:
            return None
        
        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Không đọc được frame!")
            return None
        
        return frame
    
    def grab(self) -> bool:
        """
        Grab frame mà không decode (nhanh hơn read).
        Dùng khi muốn skip frame.
        """
        if not self._is_open or self._cap is None:
            return False
        return self._cap.grab()
    
    def release(self):
        """Giải phóng camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("📹 Camera released")
    
    def is_opened(self) -> bool:
        """Kiểm tra camera đang mở."""
        return self._is_open and self._cap is not None and self._cap.isOpened()
    
    def get_resolution(self) -> Tuple[int, int]:
        """Lấy resolution thực tế."""
        if self._cap is None:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    def get_fps(self) -> float:
        """Lấy FPS thực tế."""
        if self._cap is None:
            return 0
        return self._cap.get(cv2.CAP_PROP_FPS)
    
    def __enter__(self):
        """Context manager support."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.release()
        return False


def create_camera(
    width: int = 640,
    height: int = 480,
    is_pi: bool = False,
    low_memory: bool = False
) -> CameraManager:
    """
    Factory function để tạo camera với config phù hợp.
    
    Args:
        width: Chiều rộng mong muốn
        height: Chiều cao mong muốn
        is_pi: True nếu Raspberry Pi
        low_memory: True để dùng resolution thấp
        
    Returns:
        CameraManager instance
    """
    if low_memory:
        width = min(width, 320)
        height = min(height, 240)
    
    config = CameraConfig(
        width=width,
        height=height,
        fps=15 if is_pi else 30,
        use_mjpg=is_pi
    )
    
    return CameraManager(config=config, is_pi=is_pi)


if __name__ == "__main__":
    import platform
    
    print("=== Testing CameraManager ===\n")
    
    is_pi = platform.system() == "Linux"
    camera = create_camera(width=640, height=480, is_pi=is_pi)
    
    if camera.open():
        print(f"Resolution: {camera.get_resolution()}")
        print(f"FPS: {camera.get_fps()}")
        
        # Capture 10 frames
        start = time.time()
        for i in range(10):
            frame = camera.read()
            if frame is not None:
                print(f"  Frame {i+1}: {frame.shape}")
        
        elapsed = time.time() - start
        print(f"\nCaptured 10 frames in {elapsed:.2f}s ({10/elapsed:.1f} fps)")
        
        camera.release()
    else:
        print("❌ Failed to open camera")
    
    print("\n✅ Test complete!")
