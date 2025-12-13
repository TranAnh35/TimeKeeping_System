# src/frame_skip.py
"""
Adaptive Frame Skip module.

Tự động điều chỉnh số frame bỏ qua dựa trên tải CPU:
- CPU nhàn rỗi → giảm skip (xử lý nhiều frame hơn, mượt hơn)
- CPU quá tải → tăng skip (giảm tải, tránh lag)

Usage:
    from src.frame_skip import AdaptiveFrameSkip
    
    skip = AdaptiveFrameSkip()
    
    for frame_count in range(1000):
        if skip.should_process(frame_count):
            # Process frame
            process_time = do_processing()
            skip.update(process_time)
"""
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


@dataclass
class FrameSkipStats:
    """Thống kê hiệu suất frame skip."""
    current_skip: int
    avg_process_ms: float
    skip_rate_percent: float
    effective_fps: float


class AdaptiveFrameSkip:
    """
    Tự động điều chỉnh frame skip dựa trên thời gian xử lý thực tế.
    """
    
    def __init__(
        self,
        target_time: float = 0.15,      # Target 150ms per frame
        min_skip: int = 1,
        max_skip: int = 5,
        initial_skip: int = 2,
        history_size: int = 5,          # Số samples để tính moving average
        adjust_interval: float = 2.0    # Điều chỉnh mỗi 2 giây
    ):
        self.target_time = target_time
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.current_skip = initial_skip
        
        # Moving average với deque (O(1) append và auto-remove)
        self._time_history: Deque[float] = deque(maxlen=history_size)
        self._history_size = history_size
        
        # Điều chỉnh theo interval
        self._last_adjust_time = time.time()
        self._adjust_interval = adjust_interval
        
        # Stats
        self._total_frames = 0
        self._processed_frames = 0
    
    def update(self, process_time: float) -> int:
        """
        Cập nhật thời gian xử lý và điều chỉnh skip rate.
        
        Args:
            process_time: Thời gian xử lý frame vừa rồi (giây)
            
        Returns:
            current_skip hiện tại
        """
        # Thêm vào history (deque tự xóa phần tử cũ khi đầy)
        self._time_history.append(process_time)
        self._processed_frames += 1
        
        # Chỉ điều chỉnh mỗi adjust_interval giây
        current_time = time.time()
        if current_time - self._last_adjust_time < self._adjust_interval:
            return self.current_skip
        
        self._last_adjust_time = current_time
        
        # Tính trung bình
        if not self._time_history:
            return self.current_skip
        
        avg_time = sum(self._time_history) / len(self._time_history)
        old_skip = self.current_skip
        
        # Điều chỉnh dựa trên tỉ lệ với target
        if avg_time < self.target_time * 0.5:
            # CPU rất nhàn rỗi (<50% target) → giảm skip
            self.current_skip = max(self.min_skip, self.current_skip - 1)
        elif avg_time < self.target_time * 0.8:
            # CPU nhàn rỗi (<80% target) → giảm skip nhẹ
            if self.current_skip > self.min_skip:
                self.current_skip -= 1
        elif avg_time > self.target_time * 1.5:
            # CPU quá tải (>150% target) → tăng skip nhiều
            self.current_skip = min(self.max_skip, self.current_skip + 2)
        elif avg_time > self.target_time * 1.2:
            # CPU hơi cao (>120% target) → tăng skip nhẹ
            self.current_skip = min(self.max_skip, self.current_skip + 1)
        
        return self.current_skip
    
    def should_process(self, frame_count: int) -> bool:
        """
        Kiểm tra frame này có nên xử lý không.
        
        Args:
            frame_count: Số thứ tự frame hiện tại
            
        Returns:
            True nếu nên xử lý frame này
        """
        self._total_frames += 1
        return frame_count % self.current_skip == 0
    
    def get_stats(self) -> FrameSkipStats:
        """Lấy thống kê hiệu suất."""
        avg_time = (
            sum(self._time_history) / len(self._time_history) 
            if self._time_history else 0
        )
        
        skip_rate = (
            (self._total_frames - self._processed_frames) / self._total_frames * 100
            if self._total_frames > 0 else 0
        )
        
        # Giả sử camera 15fps
        effective_fps = (
            self._processed_frames / max(1, self._total_frames) * 15
        )
        
        return FrameSkipStats(
            current_skip=self.current_skip,
            avg_process_ms=avg_time * 1000,
            skip_rate_percent=skip_rate,
            effective_fps=effective_fps
        )
    
    def reset_stats(self):
        """Reset thống kê (hữu ích khi muốn đo lại từ đầu)."""
        self._total_frames = 0
        self._processed_frames = 0
        self._time_history.clear()


class FixedFrameSkip:
    """
    Simple fixed frame skip (không adaptive).
    Dùng khi không cần tự động điều chỉnh.
    """
    
    def __init__(self, skip: int = 1):
        self.current_skip = skip
        self._total_frames = 0
        self._processed_frames = 0
    
    def update(self, process_time: float) -> int:
        """Không làm gì, chỉ đếm."""
        self._processed_frames += 1
        return self.current_skip
    
    def should_process(self, frame_count: int) -> bool:
        self._total_frames += 1
        return frame_count % self.current_skip == 0
    
    def get_stats(self) -> FrameSkipStats:
        return FrameSkipStats(
            current_skip=self.current_skip,
            avg_process_ms=0,
            skip_rate_percent=(
                (self._total_frames - self._processed_frames) / max(1, self._total_frames) * 100
            ),
            effective_fps=self._processed_frames / max(1, self._total_frames) * 15
        )


def create_frame_skip(
    enabled: bool = True,
    target_time: float = 0.15,
    min_skip: int = 1,
    max_skip: int = 5,
    default_skip: int = 2
):
    """
    Factory function để tạo frame skip handler phù hợp.
    
    Args:
        enabled: True = Adaptive, False = Fixed
        target_time: Target processing time (cho Adaptive)
        min_skip, max_skip: Range cho Adaptive
        default_skip: Giá trị mặc định
        
    Returns:
        AdaptiveFrameSkip hoặc FixedFrameSkip instance
    """
    if enabled:
        return AdaptiveFrameSkip(
            target_time=target_time,
            min_skip=min_skip,
            max_skip=max_skip,
            initial_skip=default_skip
        )
    else:
        return FixedFrameSkip(skip=default_skip)


if __name__ == "__main__":
    import random
    
    print("=== Testing AdaptiveFrameSkip ===\n")
    
    skip = AdaptiveFrameSkip(
        target_time=0.1,
        min_skip=1,
        max_skip=5,
        initial_skip=2
    )
    
    # Simulate varying processing times
    for i in range(100):
        if skip.should_process(i):
            # Simulate processing with varying load
            if i < 30:
                process_time = random.uniform(0.03, 0.05)  # Light load
            elif i < 60:
                process_time = random.uniform(0.15, 0.20)  # Heavy load
            else:
                process_time = random.uniform(0.08, 0.12)  # Normal load
            
            skip.update(process_time)
        
        if i % 20 == 19:
            stats = skip.get_stats()
            print(f"Frame {i+1}: skip={stats.current_skip}, "
                  f"avg={stats.avg_process_ms:.0f}ms, "
                  f"effective_fps={stats.effective_fps:.1f}")
    
    print("\n✅ Test passed!")
