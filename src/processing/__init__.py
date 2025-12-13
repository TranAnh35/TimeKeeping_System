# src/processing/__init__.py
"""
Processing modules - Frame & Face Processing.

- frame_skip: Adaptive frame skip
- display: UI/Overlay handler
- attendance: Attendance logic (hold, cooldown)
"""

from .frame_skip import AdaptiveFrameSkip, FixedFrameSkip, create_frame_skip
from .display import DisplayHandler, FaceStatus
from .attendance import AttendanceTracker, AttendanceAction, ProcessResult

__all__ = [
    'AdaptiveFrameSkip',
    'FixedFrameSkip',
    'create_frame_skip',
    'DisplayHandler',
    'FaceStatus',
    'AttendanceTracker',
    'AttendanceAction',
    'ProcessResult',
]
