# src/attendance.py
"""
Attendance Logic module.

Xử lý logic chấm công: giữ mặt, cooldown, check-in/check-out.

Usage:
    from src.attendance import AttendanceTracker
    
    tracker = AttendanceTracker(
        hold_time=1.5,
        cooldown=300
    )
    
    # Trong loop xử lý
    result = tracker.process_face(name="John", current_time=time.time())
    if result.should_log:
        log_attendance(result.name)
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from enum import Enum


class AttendanceAction(Enum):
    """Loại hành động chấm công."""
    NONE = "none"           # Không làm gì
    HOLDING = "holding"     # Đang giữ mặt
    CHECK_IN = "check_in"
    CHECK_OUT = "check_out"


@dataclass
class ProcessResult:
    """Kết quả xử lý một khuôn mặt."""
    name: str
    action: AttendanceAction = AttendanceAction.NONE
    hold_progress: float = 0.0        # 0-1
    hold_remaining: float = 0.0       # Giây còn lại
    should_log: bool = False          # True nếu cần ghi attendance
    
    @property
    def is_holding(self) -> bool:
        return self.action == AttendanceAction.HOLDING
    
    @property
    def is_complete(self) -> bool:
        return self.action in (AttendanceAction.CHECK_IN, AttendanceAction.CHECK_OUT)


class FaceHoldTracker:
    """
    Theo dõi thời gian giữ mặt trước camera.
    Người dùng cần giữ mặt một thời gian trước khi được chấm công.
    """
    
    def __init__(self, hold_time: float = 1.5):
        """
        Args:
            hold_time: Thời gian cần giữ mặt (giây)
        """
        self.hold_time = hold_time
        self._start_times: Dict[str, float] = {}
    
    def start_tracking(self, name: str, current_time: float):
        """Bắt đầu theo dõi một người."""
        if name not in self._start_times:
            self._start_times[name] = current_time
    
    def get_hold_duration(self, name: str, current_time: float) -> float:
        """Lấy thời gian đã giữ mặt."""
        if name not in self._start_times:
            return 0.0
        return current_time - self._start_times[name]
    
    def get_progress(self, name: str, current_time: float) -> float:
        """Lấy tiến độ giữ mặt (0-1)."""
        duration = self.get_hold_duration(name, current_time)
        return min(duration / self.hold_time, 1.0)
    
    def get_remaining(self, name: str, current_time: float) -> float:
        """Lấy thời gian còn lại (giây)."""
        duration = self.get_hold_duration(name, current_time)
        return max(0.0, self.hold_time - duration)
    
    def is_complete(self, name: str, current_time: float) -> bool:
        """Kiểm tra đã giữ đủ thời gian chưa."""
        return self.get_hold_duration(name, current_time) >= self.hold_time
    
    def stop_tracking(self, name: str):
        """Dừng theo dõi một người."""
        if name in self._start_times:
            del self._start_times[name]
    
    def cleanup_missing(self, present_names: Set[str], cooldown_names: Set[str]):
        """
        Xóa tracking của những người không còn trong frame.
        Giữ lại những người đang trong cooldown.
        """
        to_remove = [
            name for name in self._start_times 
            if name not in present_names and name not in cooldown_names
        ]
        for name in to_remove:
            del self._start_times[name]
    
    def reset(self):
        """Reset tất cả tracking."""
        self._start_times.clear()


class CooldownTracker:
    """
    Theo dõi cooldown giữa các lần chấm công.
    Tránh chấm công liên tục trong thời gian ngắn.
    """
    
    def __init__(self, cooldown_seconds: float = 300):
        """
        Args:
            cooldown_seconds: Thời gian cooldown (giây)
        """
        self.cooldown_seconds = cooldown_seconds
        self._last_checkin: Dict[str, float] = {}
    
    def is_in_cooldown(self, name: str, current_time: float) -> bool:
        """Kiểm tra người này có đang trong cooldown không."""
        if name not in self._last_checkin:
            return False
        elapsed = current_time - self._last_checkin[name]
        return elapsed < self.cooldown_seconds
    
    def record_checkin(self, name: str, current_time: float):
        """Ghi nhận thời điểm chấm công."""
        self._last_checkin[name] = current_time
    
    def get_remaining_cooldown(self, name: str, current_time: float) -> float:
        """Lấy thời gian cooldown còn lại (giây)."""
        if name not in self._last_checkin:
            return 0.0
        elapsed = current_time - self._last_checkin[name]
        return max(0.0, self.cooldown_seconds - elapsed)
    
    def get_cooldown_names(self) -> Set[str]:
        """Lấy tất cả tên đang trong cooldown."""
        return set(self._last_checkin.keys())
    
    def reset(self):
        """Reset tất cả cooldown."""
        self._last_checkin.clear()


class AttendanceTracker:
    """
    Kết hợp FaceHoldTracker và CooldownTracker để xử lý chấm công hoàn chỉnh.
    """
    
    def __init__(
        self,
        hold_time: float = 1.5,
        cooldown_seconds: float = 300
    ):
        """
        Args:
            hold_time: Thời gian giữ mặt (giây)
            cooldown_seconds: Thời gian cooldown (giây)
        """
        self.hold_tracker = FaceHoldTracker(hold_time)
        self.cooldown_tracker = CooldownTracker(cooldown_seconds)
        
        # Callback khi chấm công thành công
        self._on_attendance_callback = None
    
    def set_attendance_callback(self, callback):
        """
        Đặt callback được gọi khi chấm công thành công.
        
        Args:
            callback: function(name: str) -> str  # Returns 'check_in' or 'check_out'
        """
        self._on_attendance_callback = callback
    
    def process_face(
        self,
        name: str,
        current_time: Optional[float] = None
    ) -> ProcessResult:
        """
        Xử lý một khuôn mặt đã được nhận diện.
        
        Args:
            name: Tên người đã nhận diện
            current_time: Thời điểm hiện tại (mặc định time.time())
            
        Returns:
            ProcessResult với thông tin xử lý
        """
        if current_time is None:
            current_time = time.time()
        
        # Kiểm tra cooldown
        if self.cooldown_tracker.is_in_cooldown(name, current_time):
            # Đang trong cooldown, không cần giữ mặt nữa
            return ProcessResult(
                name=name,
                action=AttendanceAction.NONE,
                hold_progress=1.0,  # Đã hoàn thành trước đó
                should_log=False
            )
        
        # Bắt đầu/tiếp tục tracking
        self.hold_tracker.start_tracking(name, current_time)
        
        progress = self.hold_tracker.get_progress(name, current_time)
        remaining = self.hold_tracker.get_remaining(name, current_time)
        
        # Kiểm tra đã giữ đủ thời gian chưa
        if self.hold_tracker.is_complete(name, current_time):
            # Giữ đủ thời gian -> Chấm công
            self.cooldown_tracker.record_checkin(name, current_time)
            self.hold_tracker.stop_tracking(name)
            
            # Gọi callback để log attendance
            action = AttendanceAction.CHECK_IN
            if self._on_attendance_callback:
                result = self._on_attendance_callback(name)
                if result == 'check_out':
                    action = AttendanceAction.CHECK_OUT
            
            return ProcessResult(
                name=name,
                action=action,
                hold_progress=1.0,
                hold_remaining=0.0,
                should_log=True
            )
        
        # Đang trong quá trình giữ mặt
        return ProcessResult(
            name=name,
            action=AttendanceAction.HOLDING,
            hold_progress=progress,
            hold_remaining=remaining,
            should_log=False
        )
    
    def cleanup_frame(self, recognized_names: Set[str]):
        """
        Cleanup sau mỗi frame.
        Xóa tracking của những người không còn trong frame (trừ cooldown).
        
        Args:
            recognized_names: Set các tên được nhận diện trong frame này
        """
        cooldown_names = self.cooldown_tracker.get_cooldown_names()
        self.hold_tracker.cleanup_missing(recognized_names, cooldown_names)
    
    def reset(self):
        """Reset tất cả tracking."""
        self.hold_tracker.reset()
        self.cooldown_tracker.reset()


if __name__ == "__main__":
    print("=== Testing AttendanceTracker ===\n")
    
    # Tạo tracker với hold_time ngắn để test
    tracker = AttendanceTracker(hold_time=1.0, cooldown_seconds=5.0)
    
    # Mock callback
    call_count = {"check_in": 0, "check_out": 0}
    def mock_log(name):
        if call_count["check_in"] % 2 == 0:
            call_count["check_in"] += 1
            return "check_in"
        else:
            call_count["check_out"] += 1
            return "check_out"
    
    tracker.set_attendance_callback(mock_log)
    
    # Test scenario
    start_time = time.time()
    
    print("1. First detection (should start holding):")
    result = tracker.process_face("John", start_time)
    print(f"   Action: {result.action.value}")
    print(f"   Progress: {result.hold_progress:.1%}")
    print(f"   Remaining: {result.hold_remaining:.2f}s")
    
    print("\n2. After 0.5s (still holding):")
    result = tracker.process_face("John", start_time + 0.5)
    print(f"   Action: {result.action.value}")
    print(f"   Progress: {result.hold_progress:.1%}")
    
    print("\n3. After 1.0s (should complete):")
    result = tracker.process_face("John", start_time + 1.0)
    print(f"   Action: {result.action.value}")
    print(f"   Should log: {result.should_log}")
    
    print("\n4. Immediately after (in cooldown):")
    result = tracker.process_face("John", start_time + 1.1)
    print(f"   Action: {result.action.value}")
    print(f"   In cooldown: {tracker.cooldown_tracker.is_in_cooldown('John', start_time + 1.1)}")
    
    print("\n5. After cooldown expires (6s later):")
    result = tracker.process_face("John", start_time + 7.0)
    print(f"   Action: {result.action.value}")
    print(f"   Progress: {result.hold_progress:.1%}")
    
    print("\n✅ Test complete!")
