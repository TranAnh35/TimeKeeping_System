# src/main.py
"""
TimeKeeping System - Main Entry Point.

Đây là file điều phối chính của hệ thống chấm công.
Logic đã được tách ra các module riêng biệt:
- core/: Infrastructure (settings, camera, tflite)
- processing/: Frame processing (display, attendance, frame_skip)
- detect/: Face detection
- recognition/: Face recognition

File này chỉ đảm nhiệm việc kết nối các module lại với nhau.

Usage:
    python -m src.main                    # Auto-detect model type
    python -m src.main --int8             # Force INT8 models
    python -m src.main --float32          # Force Float32 models
    python -m src.main --threshold 0.6    # Custom recognition threshold
    python -m src.main --no-web           # Disable web server
"""
import os
import sys
import gc
import time
import datetime
import threading
import logging
import argparse

# === SETUP DISPLAY TRƯỚC KHI IMPORT CV2 ===
if os.environ.get("DISPLAY", "") == "":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2

# === IMPORTS ===
try:
    from .core import settings, CameraManager, CameraConfig
    from .core.model_factory import create_detector, create_recognizer
    from .processing import create_frame_skip, DisplayHandler, FaceStatus
    from .processing import AttendanceTracker, AttendanceAction
    from .data.database import (
        log_attendance, add_employee, remove_employee,
        sync_employees_with_face_db, init_db,
        midnight_checkout_all_sessions
    )
except ImportError:
    from core import settings, CameraManager, CameraConfig
    from core.model_factory import create_detector, create_recognizer
    from processing import create_frame_skip, DisplayHandler, FaceStatus
    from processing import AttendanceTracker, AttendanceAction
    from data.database import (
        log_attendance, add_employee, remove_employee,
        sync_employees_with_face_db, init_db,
        midnight_checkout_all_sessions
    )

# === LOGGING SETUP ===
if settings.IS_PI:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('attendance.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# === OPENCV THREADING ===
if settings.IS_PI:
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='TimeKeeping System - Face Recognition Attendance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                    # Run with defaults
  python -m src.main --threshold 0.6    # Custom threshold
  python -m src.main --no-web --headless
        """
    )
    
    # Recognition settings
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        metavar='VALUE',
        help=f'Recognition threshold (default: {settings.RECOGNITION_THRESHOLD})'
    )
    
    # Web server
    parser.add_argument(
        '--no-web',
        action='store_true',
        help='Disable web server'
    )
    parser.add_argument(
        '--port', '-p',
        type=int,
        metavar='PORT',
        help=f'Web server port (default: {settings.WEB_PORT})'
    )
    
    # Display mode
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no GUI)'
    )
    parser.add_argument(
        '--gui',
        action='store_true', 
        help='Force GUI mode (even on Pi)'
    )
    
    # Camera settings
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        metavar='ID',
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--resolution', '-r',
        type=str,
        metavar='WxH',
        help=f'Camera resolution (default: {settings.CAMERA_WIDTH}x{settings.CAMERA_HEIGHT})'
    )
    
    # Timing
    parser.add_argument(
        '--cooldown',
        type=int,
        metavar='SECONDS',
        help=f'Cooldown between check-ins (default: {settings.COOLDOWN_SECONDS}s)'
    )
    parser.add_argument(
        '--hold-time',
        type=float,
        metavar='SECONDS',
        help=f'Face hold time to confirm (default: {settings.HOLD_TIME_SECONDS}s)'
    )
    
    # Debug
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def apply_arguments(args):
    """Apply command line arguments to settings."""
    changes = []
    
    # Recognition threshold
    if args.threshold is not None:
        settings.RECOGNITION_THRESHOLD = args.threshold
        changes.append(f"Threshold: {args.threshold}")
    
    # Web server
    if args.no_web:
        settings.ENABLE_WEB_SERVER = False
        changes.append("Web: disabled")
    if args.port:
        settings.WEB_PORT = args.port
        changes.append(f"Port: {args.port}")
    
    # Display mode
    if args.headless:
        settings.HEADLESS_MODE = True
        settings.OVERLAY_ENABLED = False
        changes.append("Mode: headless")
    if args.gui:
        settings.FORCE_GUI_MODE = True
        settings.HEADLESS_MODE = False
        settings.OVERLAY_ENABLED = True
        changes.append("Mode: GUI (forced)")
    
    # Camera resolution
    if args.resolution:
        try:
            w, h = map(int, args.resolution.lower().split('x'))
            settings.CAMERA_WIDTH = w
            settings.CAMERA_HEIGHT = h
            changes.append(f"Resolution: {w}x{h}")
        except ValueError:
            print(f"⚠️ Invalid resolution format: {args.resolution} (use WxH, e.g., 640x480)")
    
    # Timing
    if args.cooldown:
        settings.COOLDOWN_SECONDS = args.cooldown
        changes.append(f"Cooldown: {args.cooldown}s")
    if args.hold_time:
        settings.HOLD_TIME_SECONDS = args.hold_time
        changes.append(f"Hold time: {args.hold_time}s")
    
    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        changes.append("Verbose: ON")
    
    return changes


def start_web_server(detector=None, recognizer=None):
    """Chạy web server trong thread riêng."""
    try:
        try:
            from .web.server import run_server, setup_management
        except ImportError:
            from web.server import run_server, setup_management
        
        # Setup management module với detector và recognizer
        if detector is not None and recognizer is not None:
            setup_management(detector, recognizer)
            logger.info("Web Management đã được kích hoạt: /manage")
        
        run_server(host='0.0.0.0', port=settings.WEB_PORT)
    except Exception as e:
        logger.error(f"Web server error: {e}")
        import traceback
        traceback.print_exc()


def print_startup_info(recognizer, sync_result: dict):
    """In thông tin khởi động."""
    print("\n" + "=" * 50)
    print("🕐 HỆ THỐNG CHẤM CÔNG")
    print("=" * 50)
    
    # Platform mode
    if settings.IS_WINDOWS:
        mode = "Windows (GUI)"
    elif settings.FORCE_GUI_MODE:
        mode = "Pi (GUI - debug mode)"
    else:
        mode = "Pi (Headless)"
    
    n_people, n_emb = recognizer.get_db_info()
    
    print(f"📍 Mode: {mode}")
    print(f"👥 Database: {n_people} người ({n_emb} ảnh)")
    print(f"⏱️ Cooldown: {settings.COOLDOWN_SECONDS}s ({settings.COOLDOWN_SECONDS // 60}m)")
    
    if sync_result['added'] or sync_result['removed']:
        print(f"🔄 Sync: +{sync_result['added']} -{sync_result['removed']}")
    
    print(f"🧠 Models: INT8 (optimized)")
    
    if settings.ENABLE_ADAPTIVE_SKIP:
        print(f"⚡ Adaptive Skip: ON (target={int(settings.TARGET_PROCESS_TIME * 1000)}ms)")
    else:
        print(f"⚡ Frame Skip: {settings.DEFAULT_FRAME_SKIP} (fixed)")
    
    if settings.ENABLE_CENTER_ROI:
        print(f"🎯 Center ROI: ON ({int(settings.CENTER_ROI_RATIO * 100)}% màn hình)")
    
    if settings.ENABLE_MIDNIGHT_CHECKOUT:
        print(f"⏰ Auto Checkout: ON (00:00 mỗi ngày)")
    
    if settings.ENABLE_WEB_SERVER:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "localhost"
        print(f"🌐 Web: http://{local_ip}:{settings.WEB_PORT}")
    
    print("-" * 50)
    if not settings.HEADLESS_MODE:
        print("⌨️  r=đăng ký | d=xóa | l=list | q=thoát")
    else:
        print("⌨️  Ctrl+C để thoát")
    print("=" * 50 + "\n")


def handle_keyboard(key: int, frame, detections, recognizer) -> bool:
    """
    Xử lý phím nhấn.
    
    Returns:
        True nếu nên thoát chương trình
    """
    if key == ord('q'):
        return True
    
    elif key == ord('l'):
        # List registered faces
        print("\n📋 Database:")
        names = recognizer.get_registered_names()
        if names:
            for i, name in enumerate(names, 1):
                emb_data = recognizer.db.get(name)
                if isinstance(emb_data, list):
                    count = len(emb_data)
                elif hasattr(emb_data, "shape"):
                    count = emb_data.shape[0]
                else:
                    count = 1
                print(f"   {i}. {name} ({count})")
        else:
            print("   (trống)")
        print()
    
    elif key == ord('d'):
        # Delete face
        cv2.destroyAllWindows()
        print("\n🗑️ Xóa người:")
        names = recognizer.get_registered_names()
        if not names:
            print("   Database trống!")
        else:
            for i, name in enumerate(names, 1):
                print(f"   {i}. {name}")
            choice = input("Nhập tên (Enter=hủy): ").strip()
            if choice:
                if recognizer.remove_face(choice):
                    recognizer.save_db()
                    remove_employee(choice)
                    print(f"   ✅ Đã xóa: {choice}")
                else:
                    print(f"   ❌ Không tìm thấy: {choice}")
        print()
        cv2.namedWindow("May Cham Cong")
    
    elif key == ord('r'):
        # Register new face
        if len(detections) > 0:
            detections.sort(key=lambda d: d['box'][2] * d['box'][3], reverse=True)
            det = detections[0]
            x, y, w, h = det['box']
            face_reg = frame[y:y+h, x:x+w]
            
            cv2.destroyAllWindows()
            name = input("Tên nhân viên: ").strip()
            if name:
                recognizer.add_face(name, face_reg)
                recognizer.save_db()
                add_employee(name)
                print(f"   ✅ Đã đăng ký: {name}\n")
            
            cv2.namedWindow("May Cham Cong")
    
    return False


def main():
    """Main entry point."""
    
    # === 0. PARSE ARGUMENTS ===
    args = parse_arguments()
    arg_changes = apply_arguments(args)
    
    if arg_changes:
        print("🔧 Command-line overrides:")
        for change in arg_changes:
            print(f"   • {change}")
        print()
    
    # === 1. KHỞI TẠO DATABASE ===
    init_db()
    
    # === 2. CAMERA ===
    camera_config = CameraConfig(
        width=settings.CAMERA_WIDTH,
        height=settings.CAMERA_HEIGHT,
        fps=15 if settings.IS_PI else 30
    )
    camera = CameraManager(device_id=args.camera, config=camera_config, is_pi=settings.IS_PI)
    
    if not camera.open():
        print("❌ Không thể kết nối camera!")
        return
    
    # === 3. MODELS ===
    try:
        # Face detector
        detector = create_detector()
        
        # Face recognizer
        recognizer = create_recognizer()
        
        # Sync database
        sync_result = sync_employees_with_face_db(recognizer.get_registered_names())
        
        gc.collect()
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo: {e}")
        camera.release()
        return
    
    # === 5. COMPONENTS ===
    # Frame skip
    frame_skip = create_frame_skip(
        enabled=settings.ENABLE_ADAPTIVE_SKIP,
        target_time=settings.TARGET_PROCESS_TIME,
        min_skip=settings.MIN_FRAME_SKIP,
        max_skip=settings.MAX_FRAME_SKIP,
        default_skip=settings.DEFAULT_FRAME_SKIP
    )
    
    # Display handler
    display = DisplayHandler(overlay_enabled=settings.OVERLAY_ENABLED)
    
    # Attendance tracker
    attendance = AttendanceTracker(
        hold_time=settings.HOLD_TIME_SECONDS,
        cooldown_seconds=settings.COOLDOWN_SECONDS
    )
    attendance.set_attendance_callback(log_attendance)
    
    # === 6. STARTUP INFO ===
    print_startup_info(recognizer, sync_result)
    logger.info(f"CONFIG: COOLDOWN={settings.COOLDOWN_SECONDS}s, "
                f"HOLD={settings.HOLD_TIME_SECONDS}s, "
                f"WEB={settings.ENABLE_WEB_SERVER}:{settings.WEB_PORT}")
    
    # === 7. WEB SERVER (background) - start sau khi có detector/recognizer ===
    if settings.ENABLE_WEB_SERVER:
        web_thread = threading.Thread(
            target=start_web_server, 
            args=(detector, recognizer),
            daemon=True
        )
        web_thread.start()
    
    # === 8. MIDNIGHT CHECKOUT (on startup) ===
    if settings.ENABLE_MIDNIGHT_CHECKOUT:
        auto_results = midnight_checkout_all_sessions()
        for r in auto_results:
            logger.warning(f"⚠️ Midnight checkout: {r['name']} ({r['duration_str']})")
    
    # === 8. MAIN LOOP ===
    frame_count = 0
    last_midnight_check = datetime.datetime.now().date()
    last_status_time = 0
    last_db_reload_check = 0  # Hot-reload database check
    DB_RELOAD_INTERVAL = 5.0  # Check mỗi 5 giây
    
    try:
        while True:
            frame_count += 1
            current_time = time.time()
            
            # --- Hot-reload database check (định kỳ mỗi 5s) ---
            if current_time - last_db_reload_check >= DB_RELOAD_INTERVAL:
                recognizer.reload_db_if_changed()
                last_db_reload_check = current_time
            
            # --- Frame skip: dùng grab() để bỏ qua frame mà không decode ---
            if not frame_skip.should_process(frame_count):
                camera.grab()  # Chỉ advance buffer, không decode (nhanh hơn read)
                continue
            
            # --- Đọc frame (chỉ khi cần xử lý) ---
            frame = camera.read()
            if frame is None:
                logger.warning("Không đọc được camera!")
                break
            
            # --- Midnight checkout check (chỉ kiểm tra 1 lần/ngày) ---
            if settings.ENABLE_MIDNIGHT_CHECKOUT:
                today = datetime.datetime.now().date()
                if today > last_midnight_check:
                    for r in midnight_checkout_all_sessions():
                        logger.warning(f"⚠️ Midnight checkout: {r['name']} ({r['duration_str']})")
                    last_midnight_check = today
            
            process_start = time.time()
            
            # --- GC ---
            if settings.GC_INTERVAL and frame_count % settings.GC_INTERVAL == 0:
                gc.collect()
            
            # --- Detection ---
            detections = detector.detect_faces(frame)
            
            # --- Center ROI ---
            roi_bounds = None
            if settings.ENABLE_CENTER_ROI:
                roi_bounds = display.draw_center_roi(frame, settings.CENTER_ROI_RATIO)
            
            # --- Process faces ---
            recognized_this_frame = set()
            
            for det in detections:
                x, y, w, h = det['box']
                
                # Skip too small faces
                if w < 60 or h < 60:
                    continue
                
                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue
                
                # Center ROI check
                if roi_bounds and not display.is_in_roi((x, y, w, h), roi_bounds):
                    display.draw_face(frame, (x, y, w, h), FaceStatus.OUTSIDE_ROI)
                    continue
                
                # Recognition
                label, distance = recognizer.recognize(
                    face, 
                    threshold=settings.RECOGNITION_THRESHOLD
                )
                
                if label is None:
                    display.draw_face(
                        frame, (x, y, w, h), 
                        FaceStatus.UNKNOWN, 
                        distance=distance
                    )
                    
                else:
                    # Recognized -> process attendance
                    recognized_this_frame.add(label)
                    
                    result = attendance.process_face(label, current_time)
                    
                    if result.is_holding:
                        display.draw_hold_progress(
                            frame, (x, y, w, h),
                            result.hold_progress,
                            result.hold_remaining,
                            label
                        )
                    else:
                        display.draw_face(
                            frame, (x, y, w, h),
                            FaceStatus.RECOGNIZED,
                            name=label,
                            distance=distance
                        )
                    
                    if result.should_log:
                        symbol = "🟢" if result.action == AttendanceAction.CHECK_IN else "🔴"
                        logger.info(f"{symbol} {label} - {result.action.value.upper().replace('_', '-')}")
                        
                        if not settings.HEADLESS_MODE:
                            display.draw_check_status(frame, result.action.value)
            
            # Cleanup trackers
            attendance.cleanup_frame(recognized_this_frame)
            
            # Update frame skip
            process_time = time.time() - process_start
            frame_skip.update(process_time)
            
            # --- DISPLAY ---
            if settings.HEADLESS_MODE:
                # Headless: periodic status log
                if current_time - last_status_time > 300:
                    stats = frame_skip.get_stats()
                    logger.info(f"♻️ Running... Faces: {len(detections)}, "
                               f"Skip: {stats.current_skip}, Avg: {stats.avg_process_ms:.0f}ms")
                    last_status_time = current_time
            else:
                # GUI mode
                stats = frame_skip.get_stats()
                display.draw_stats(frame, {
                    'current_skip': stats.current_skip,
                    'avg_process_ms': stats.avg_process_ms,
                    'effective_fps': stats.effective_fps
                })
                
                cv2.imshow("May Cham Cong", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if handle_keyboard(key, frame, detections, recognizer):
                    break
    
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng (Ctrl+C)")
    
    finally:
        camera.release()
        if not settings.HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("👋 Bye!")


if __name__ == "__main__":
    main()
