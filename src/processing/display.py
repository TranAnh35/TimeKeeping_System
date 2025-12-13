# src/display.py
"""
Display/UI Handler module.

Xử lý hiển thị overlay, vẽ bounding box, progress bar, và thông báo.
Tách biệt logic hiển thị khỏi logic xử lý chính.

Usage:
    from src.display import DisplayHandler
    
    display = DisplayHandler(overlay_enabled=True)
    
    # Vẽ ROI
    display.draw_center_roi(frame, 0.6)
    
    # Vẽ face detection
    display.draw_face(frame, box, status='recognized', name='John', distance=0.45)
    
    # Vẽ hold progress
    display.draw_hold_progress(frame, box, progress=0.7, remaining=0.5)
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class FaceStatus(Enum):
    """Trạng thái khuôn mặt."""
    UNKNOWN = "unknown"           # Người lạ
    RECOGNIZED = "recognized"     # Đã nhận diện
    FAKE = "fake"                 # Giả mạo
    OUTSIDE_ROI = "outside_roi"   # Ngoài vùng trung tâm
    HOLDING = "holding"           # Đang giữ mặt để chấm công


@dataclass
class ColorScheme:
    """Bảng màu cho các trạng thái (BGR format)."""
    UNKNOWN: Tuple[int, int, int] = (0, 255, 255)      # Vàng
    RECOGNIZED: Tuple[int, int, int] = (0, 255, 0)     # Xanh lá
    FAKE: Tuple[int, int, int] = (0, 0, 255)           # Đỏ
    OUTSIDE_ROI: Tuple[int, int, int] = (128, 128, 128)  # Xám
    HOLDING: Tuple[int, int, int] = (0, 255, 0)        # Xanh lá
    ROI_BORDER: Tuple[int, int, int] = (200, 200, 200) # Xám nhạt
    PROGRESS_BG: Tuple[int, int, int] = (100, 100, 100)  # Xám đậm
    PROGRESS_FG: Tuple[int, int, int] = (0, 255, 0)    # Xanh lá
    CHECK_IN: Tuple[int, int, int] = (0, 255, 0)       # Xanh lá
    CHECK_OUT: Tuple[int, int, int] = (0, 165, 255)    # Cam


class DisplayHandler:
    """
    Xử lý tất cả hiển thị UI/overlay trên frame.
    """
    
    def __init__(
        self,
        overlay_enabled: bool = True,
        colors: Optional[ColorScheme] = None,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        thickness: int = 2
    ):
        """
        Khởi tạo DisplayHandler.
        
        Args:
            overlay_enabled: True để vẽ overlay
            colors: ColorScheme tùy chỉnh
            font: OpenCV font
            font_scale: Kích thước chữ
            thickness: Độ dày nét vẽ
        """
        self.enabled = overlay_enabled
        self.colors = colors or ColorScheme()
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
    
    def draw_center_roi(
        self,
        frame: np.ndarray,
        roi_ratio: float = 0.6
    ) -> Tuple[int, int, int, int]:
        """
        Vẽ khung ROI trung tâm và trả về tọa độ.
        
        Args:
            frame: Frame để vẽ
            roi_ratio: Tỉ lệ vùng trung tâm (0-1)
            
        Returns:
            (x_min, y_min, x_max, y_max) của ROI
        """
        h, w = frame.shape[:2]
        margin = (1 - roi_ratio) / 2
        
        x_min = int(w * margin)
        x_max = int(w * (1 - margin))
        y_min = int(h * margin)
        y_max = int(h * (1 - margin))
        
        if self.enabled:
            cv2.rectangle(
                frame, 
                (x_min, y_min), (x_max, y_max),
                self.colors.ROI_BORDER, 
                1
            )
        
        return (x_min, y_min, x_max, y_max)
    
    def is_in_roi(
        self,
        face_box: Tuple[int, int, int, int],
        roi_bounds: Tuple[int, int, int, int]
    ) -> bool:
        """
        Kiểm tra tâm khuôn mặt có nằm trong ROI không.
        
        Args:
            face_box: (x, y, w, h) của face
            roi_bounds: (x_min, y_min, x_max, y_max) của ROI
            
        Returns:
            True nếu tâm mặt nằm trong ROI
        """
        x, y, w, h = face_box
        cx, cy = x + w // 2, y + h // 2
        
        rx_min, ry_min, rx_max, ry_max = roi_bounds
        return rx_min <= cx <= rx_max and ry_min <= cy <= ry_max
    
    def draw_face(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        status: FaceStatus = FaceStatus.UNKNOWN,
        name: Optional[str] = None,
        distance: Optional[float] = None,
        message: Optional[str] = None
    ):
        """
        Vẽ bounding box và label cho khuôn mặt.
        
        Args:
            frame: Frame để vẽ
            box: (x, y, w, h)
            status: Trạng thái khuôn mặt
            name: Tên (nếu đã nhận diện)
            distance: Khoảng cách embedding
            message: Thông báo tùy chỉnh
        """
        if not self.enabled:
            return
        
        x, y, w, h = box
        
        # Chọn màu theo status
        color = getattr(self.colors, status.name, self.colors.UNKNOWN)
        
        # Vẽ bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.thickness)
        
        # Xây dựng label text
        if message:
            label = message
        elif status == FaceStatus.FAKE:
            label = f"FAKE - {name}" if name else "FAKE"
        elif status == FaceStatus.OUTSIDE_ROI:
            label = "Di chuyen vao giua"
        elif status == FaceStatus.UNKNOWN:
            label = "Unknown"
            if distance is not None and distance < float('inf'):
                label += f" d={distance:.2f}"
        elif name:
            label = name
            if distance is not None and distance < float('inf'):
                label += f" d={distance:.2f}"
        else:
            label = ""
        
        # Vẽ label
        if label:
            cv2.putText(
                frame, label,
                (x, y - 10),
                self.font, self.font_scale, color, self.thickness
            )
    
    def draw_hold_progress(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        progress: float,
        remaining: float,
        name: str
    ):
        """
        Vẽ progress bar giữ mặt để chấm công.
        
        Args:
            frame: Frame để vẽ
            box: (x, y, w, h)
            progress: Tiến độ (0-1)
            remaining: Thời gian còn lại (giây)
            name: Tên người
        """
        if not self.enabled:
            return
        
        x, y, w, h = box
        
        # Vẽ bounding box
        cv2.rectangle(
            frame, 
            (x, y), (x + w, y + h), 
            self.colors.HOLDING, 
            self.thickness
        )
        
        # Vẽ progress bar
        bar_height = 8
        bar_y = y + h + 5
        
        # Background
        cv2.rectangle(
            frame,
            (x, bar_y), (x + w, bar_y + bar_height),
            self.colors.PROGRESS_BG,
            -1
        )
        
        # Progress
        progress_width = int(w * min(progress, 1.0))
        cv2.rectangle(
            frame,
            (x, bar_y), (x + progress_width, bar_y + bar_height),
            self.colors.PROGRESS_FG,
            -1
        )
        
        # Label
        if remaining > 0:
            label = f"{name} - Giu {remaining:.1f}s"
        else:
            label = name
        
        cv2.putText(
            frame, label,
            (x, y - 10),
            self.font, self.font_scale, self.colors.HOLDING, self.thickness
        )
    
    def draw_check_status(
        self,
        frame: np.ndarray,
        action: str,
        position: Tuple[int, int] = (10, 50)
    ):
        """
        Vẽ thông báo CHECK-IN/CHECK-OUT.
        
        Args:
            frame: Frame để vẽ
            action: 'check_in' hoặc 'check_out'
            position: Vị trí vẽ (x, y)
        """
        if not self.enabled:
            return
        
        if action == 'check_in':
            text = "CHECK-IN OK"
            color = self.colors.CHECK_IN
        else:
            text = "CHECK-OUT OK"
            color = self.colors.CHECK_OUT
        
        cv2.putText(
            frame, text,
            position,
            self.font, 1.0, color, 3
        )
    
    def draw_stats(
        self,
        frame: np.ndarray,
        stats: Dict[str, Any],
        position: Optional[Tuple[int, int]] = None
    ):
        """
        Vẽ thông tin stats (skip, fps, etc) góc dưới màn hình.
        
        Args:
            frame: Frame để vẽ
            stats: Dict chứa thông tin
            position: Vị trí (mặc định góc dưới trái)
        """
        if not self.enabled:
            return
        
        if position is None:
            h = frame.shape[0]
            position = (10, h - 10)
        
        # Build info text
        parts = []
        if 'current_skip' in stats:
            parts.append(f"Skip:{stats['current_skip']}")
        if 'avg_process_ms' in stats:
            parts.append(f"{stats['avg_process_ms']:.0f}ms")
        if 'effective_fps' in stats:
            parts.append(f"~{stats['effective_fps']:.1f}fps")
        
        info_text = " | ".join(parts)
        
        cv2.putText(
            frame, info_text,
            position,
            self.font, 0.4, (200, 200, 200), 1
        )
    
    def show(self, window_name: str, frame: np.ndarray) -> int:
        """
        Hiển thị frame và trả về phím nhấn.
        
        Args:
            window_name: Tên cửa sổ
            frame: Frame để hiển thị
            
        Returns:
            Mã phím nhấn hoặc -1 nếu không có
        """
        if not self.enabled:
            return -1
        
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
    
    def destroy_windows(self):
        """Đóng tất cả cửa sổ."""
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import numpy as np
    
    print("=== Testing DisplayHandler ===\n")
    
    # Create test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray background
    
    display = DisplayHandler(overlay_enabled=True)
    
    # Draw ROI
    roi = display.draw_center_roi(frame, 0.6)
    print(f"ROI bounds: {roi}")
    
    # Draw faces with different statuses
    display.draw_face(frame, (100, 100, 80, 80), FaceStatus.UNKNOWN)
    display.draw_face(frame, (250, 100, 80, 80), FaceStatus.RECOGNIZED, name="John", distance=0.42)
    display.draw_face(frame, (400, 100, 80, 80), FaceStatus.FAKE, name="Fake User")
    display.draw_face(frame, (100, 250, 80, 80), FaceStatus.OUTSIDE_ROI)
    
    # Draw hold progress
    display.draw_hold_progress(frame, (250, 250, 100, 100), progress=0.6, remaining=0.8, name="Jane")
    
    # Draw stats
    display.draw_stats(frame, {
        'current_skip': 2,
        'avg_process_ms': 85,
        'effective_fps': 12.5
    })
    
    # Show
    print("Displaying test frame (press any key to close)...")
    cv2.imshow("Display Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n✅ Test complete!")
