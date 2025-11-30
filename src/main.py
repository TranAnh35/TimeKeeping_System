# src/main.py
import cv2
import time
import datetime
import os
import gc
import sys
import platform
import threading
import logging

# --- CẤU HÌNH LOGGING ---
# Log ra file trên Pi để debug từ xa
IS_WINDOWS_EARLY = platform.system() == "Windows"
if not IS_WINDOWS_EARLY:
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

# Support both direct script execution and module import
try:
    from .detect import detect_faces
    from .recognition import FaceRecognizer
    from .database import (log_attendance, add_employee, remove_employee, 
                           sync_employees_with_face_db, init_db,
                           midnight_checkout_all_sessions)
    from .config import CONFIG
except ImportError:
    from detect import detect_faces
    from recognition import FaceRecognizer
    from database import (log_attendance, add_employee, remove_employee, 
                          sync_employees_with_face_db, init_db,
                          midnight_checkout_all_sessions)
    from config import CONFIG

# --- TỰ ĐỘNG PHÁT HIỆN PLATFORM ---
IS_WINDOWS = platform.system() == "Windows"
IS_PI = platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")

# --- CẤU HÌNH ---
# Các giá trị mặc định được lấy từ 'config/config.json' qua module `src/config.py` (CONFIG dict)
COOLDOWN_SECONDS = int(CONFIG.get('COOLDOWN_SECONDS', 300))  # seconds
HOLD_TIME_SECONDS = float(CONFIG.get('HOLD_TIME_SECONDS', 1.5))  # seconds
ENABLE_WEB_SERVER = bool(CONFIG.get('ENABLE_WEB_SERVER', True))
WEB_PORT = int(CONFIG.get('WEB_PORT', 5000))
ENABLE_ANTISPOOF = bool(CONFIG.get('ENABLE_ANTISPOOF', False))

# Threshold cho recognition:
RECOGNITION_THRESHOLD = float(CONFIG.get('RECOGNITION_THRESHOLD', 0.55))

# --- CHẾ ĐỘ HIỂN THỊ (GUI) ---
# FORCE_GUI_MODE: Bật này để hiển thị cửa sổ camera trên Pi (kết nối màn hình HDMI)
FORCE_GUI_MODE = bool(CONFIG.get('FORCE_GUI_MODE', False))  # Đặt True khi muốn debug trên Pi với màn hình

# Chế độ hoạt động: Windows luôn có GUI, Pi mặc định headless (trừ khi FORCE_GUI)
HEADLESS_MODE = not IS_WINDOWS and not FORCE_GUI_MODE

# --- CẤU HÌNH AUTO CHECK-OUT LÚC NỬA ĐÊM ---
# Tự động check-out tất cả sessions đang mở vào lúc 00:00 mỗi ngày
ENABLE_MIDNIGHT_CHECKOUT = bool(CONFIG.get('ENABLE_MIDNIGHT_CHECKOUT', True))

# --- CẤU HÌNH TỐI ƯU RAM (cho Pi 3) ---
LOW_MEMORY_MODE = bool(CONFIG.get('LOW_MEMORY_MODE', IS_PI))  # Tự động bật trên Pi (có thể override từ file config)
CAMERA_WIDTH = int(CONFIG.get('CAMERA_WIDTH', 640 if not LOW_MEMORY_MODE else 320))
CAMERA_HEIGHT = int(CONFIG.get('CAMERA_HEIGHT', 480 if not LOW_MEMORY_MODE else 240))
GC_INTERVAL = int(CONFIG.get('GC_INTERVAL', 30))

# --- ADAPTIVE FRAME SKIP ---
# Tự động điều chỉnh số frame bỏ qua dựa trên tải CPU
ENABLE_ADAPTIVE_SKIP = bool(CONFIG.get('ENABLE_ADAPTIVE_SKIP', IS_PI))  # Chỉ bật trên Pi
TARGET_PROCESS_TIME = float(CONFIG.get('TARGET_PROCESS_TIME', 0.15))  # Mục tiêu: xử lý mỗi frame trong 150ms
MIN_FRAME_SKIP = int(CONFIG.get('MIN_FRAME_SKIP', 1))
MAX_FRAME_SKIP = int(CONFIG.get('MAX_FRAME_SKIP', 5))
DEFAULT_FRAME_SKIP = int(CONFIG.get('DEFAULT_FRAME_SKIP', 2 if IS_PI else 1))


class AdaptiveFrameSkip:
    """
    Tự động điều chỉnh frame skip dựa trên thời gian xử lý thực tế.
    - CPU nhàn rỗi → giảm skip (xử lý nhiều frame hơn, mượt hơn)
    - CPU quá tải → tăng skip (giảm tải, tránh lag)
    """
    def __init__(self, target_time=TARGET_PROCESS_TIME, 
                 min_skip=MIN_FRAME_SKIP, max_skip=MAX_FRAME_SKIP,
                 initial_skip=DEFAULT_FRAME_SKIP):
        self.target_time = target_time
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.current_skip = initial_skip
        
        # Smoothing: dùng moving average để tránh dao động
        self.time_history = []
        self.history_size = 5
        
        # Stats
        self.total_frames = 0
        self.processed_frames = 0
        self.last_adjust_time = time.time()
        self.adjust_interval = 2.0  # Điều chỉnh mỗi 2 giây
    
    def update(self, process_time):
        """
        Cập nhật thời gian xử lý và điều chỉnh skip rate.
        
        Args:
            process_time: Thời gian xử lý frame vừa rồi (giây)
        """
        self.time_history.append(process_time)
        if len(self.time_history) > self.history_size:
            self.time_history.pop(0)
        
        self.processed_frames += 1
        
        # Chỉ điều chỉnh mỗi adjust_interval giây
        current_time = time.time()
        if current_time - self.last_adjust_time < self.adjust_interval:
            return self.current_skip
        
        self.last_adjust_time = current_time
        
        # Tính trung bình thời gian xử lý
        avg_time = sum(self.time_history) / len(self.time_history)
        
        old_skip = self.current_skip
        
        # Điều chỉnh skip dựa trên tỉ lệ với target
        if avg_time < self.target_time * 0.5:
            # CPU rất nhàn rỗi (<75ms) → giảm skip nhiều
            self.current_skip = max(self.min_skip, self.current_skip - 1)
        elif avg_time < self.target_time * 0.8:
            # CPU nhàn rỗi (<120ms) → giảm skip nhẹ
            if self.current_skip > self.min_skip:
                self.current_skip -= 1
        elif avg_time > self.target_time * 1.5:
            # CPU quá tải (>225ms) → tăng skip nhiều
            self.current_skip = min(self.max_skip, self.current_skip + 2)
        elif avg_time > self.target_time * 1.2:
            # CPU hơi cao (>180ms) → tăng skip nhẹ
            self.current_skip = min(self.max_skip, self.current_skip + 1)
        
        # Log khi thay đổi
        if old_skip != self.current_skip:
            logger.debug(f"Adaptive skip: {old_skip} → {self.current_skip} (avg={avg_time*1000:.0f}ms)")
        
        return self.current_skip
    
    def should_process(self, frame_count):
        """
        Kiểm tra frame này có nên xử lý không.
        
        Returns:
            True nếu nên xử lý frame này
        """
        self.total_frames += 1
        return frame_count % self.current_skip == 0
    
    def get_stats(self):
        """Lấy thống kê hiệu suất"""
        if self.total_frames == 0:
            return "No stats yet"
        
        avg_time = sum(self.time_history) / len(self.time_history) if self.time_history else 0
        skip_rate = (self.total_frames - self.processed_frames) / self.total_frames * 100
        
        return {
            'current_skip': self.current_skip,
            'avg_process_ms': avg_time * 1000,
            'skip_rate_percent': skip_rate,
            'effective_fps': self.processed_frames / max(1, self.total_frames) * 15  # Giả sử camera 15fps
        }

def start_web_server():
    """Chạy web server trong thread riêng"""
    try:
        from web_server import run_server
        run_server(host='0.0.0.0', port=WEB_PORT)
    except Exception:
        pass  # Lỗi web server không ảnh hưởng chấm công chính

def init_camera(max_retries=3, retry_delay=2):
    """Khởi tạo camera với retry logic"""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Thêm cấu hình cho Pi camera
            if IS_PI:
                cap.set(cv2.CAP_PROP_FPS, 15)  # Giảm FPS để ổn định
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # Warm-up camera - đọc vài frame đầu để ổn định
            for _ in range(5):
                cap.grab()
            return cap
        
        print(f"⚠️ Camera không sẵn sàng, thử lại ({attempt + 1}/{max_retries})...")
        time.sleep(retry_delay)
    
    return None

def main():
    # 0. Khởi tạo Database
    init_db()
    
    # 0.1 Khởi động Web Server (chạy nền)
    if ENABLE_WEB_SERVER:
        web_thread = threading.Thread(target=start_web_server, daemon=True)
        web_thread.start()
    
    # 1. Khởi tạo Camera với retry
    cap = init_camera()
    if cap is None:
        print("❌ Không thể kết nối camera sau nhiều lần thử!")
        return

    try:
        # Lazy loading: Chỉ load AntiSpoof nếu cần
        anti = None
        if ENABLE_ANTISPOOF:
            try:
                from .antispoof import AntiSpoof
            except ImportError:
                from antispoof import AntiSpoof
            anti = AntiSpoof()
        
        recognizer = FaceRecognizer()
        
        # Đồng bộ SQLite employees với face_db.pkl
        sync_result = sync_employees_with_face_db(recognizer.get_registered_names())
        
        # Garbage collect sau khi load xong models
        gc.collect()
        
    except Exception as e:
        logger.error(f"Lỗi khởi tạo: {e}")
        return

    # Dictionary lưu thời gian chấm công gần nhất
    last_checkin = {} 
    face_hold_tracker = {}

    # --- LOG KHỞI ĐỘNG GỌN GÀNG ---
    print("\n" + "="*50)
    print("🕐 HỆ THỐNG CHẤM CÔNG")
    print("="*50)
    
    # Hiển thị mode chi tiết hơn
    if IS_WINDOWS:
        mode = "Windows (GUI)"
    elif FORCE_GUI_MODE:
        mode = "Pi (GUI - debug mode)"
    else:
        mode = "Pi (Headless)"
    
    n_people, n_emb = recognizer.get_db_info()
    print(f"📍 Mode: {mode}")
    print(f"👥 Database: {n_people} người ({n_emb} ảnh)")
    print(f"⏱️ Cooldown: {COOLDOWN_SECONDS}s ({COOLDOWN_SECONDS//60}m)")
    
    if sync_result['added'] or sync_result['removed']:
        print(f"🔄 Sync: +{sync_result['added']} -{sync_result['removed']}")
    
    if LOW_MEMORY_MODE:
        print(f"💾 Low-RAM: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    
    if ENABLE_ADAPTIVE_SKIP:
        print(f"⚡ Adaptive Skip: ON (target={int(TARGET_PROCESS_TIME*1000)}ms, range={MIN_FRAME_SKIP}-{MAX_FRAME_SKIP})")
    else:
        print(f"⚡ Frame Skip: {DEFAULT_FRAME_SKIP} (fixed)")
    
    if ENABLE_MIDNIGHT_CHECKOUT:
        print(f"⏰ Auto Checkout: ON (00:00 mỗi ngày)")

    # Log cơ bản từ config để xác nhận
    logger.info(f"CONFIG: COOLDOWN={COOLDOWN_SECONDS}s, HOLD_TIME={HOLD_TIME_SECONDS}s, WEB={ENABLE_WEB_SERVER}:{WEB_PORT}, ANTISPOOF={ENABLE_ANTISPOOF}")
    
    if ENABLE_WEB_SERVER:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "localhost"
        print(f"🌐 Web: http://{local_ip}:{WEB_PORT}")
    
    print("-"*50)
    if not HEADLESS_MODE:
        print("⌨️  r=đăng ký | d=xóa | l=list | q=thoát")
    else:
        print("⌨️  Ctrl+C để thoát")
    print("="*50 + "\n")

    frame_count = 0
    last_status_time = 0
    last_midnight_check = datetime.datetime.now().date()  # Ngày cuối cùng đã kiểm tra midnight
    
    # Khởi tạo Adaptive Frame Skip
    adaptive_skip = AdaptiveFrameSkip() if ENABLE_ADAPTIVE_SKIP else None
    
    # Kiểm tra midnight checkout ngay khi khởi động (cho sessions từ hôm qua)
    if ENABLE_MIDNIGHT_CHECKOUT:
        auto_results = midnight_checkout_all_sessions()
        if auto_results:
            for r in auto_results:
                logger.warning(f"⚠️ Midnight checkout: {r['name']} ({r['duration_str']})")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Không đọc được camera!")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Kiểm tra midnight checkout khi sang ngày mới
            if ENABLE_MIDNIGHT_CHECKOUT:
                today = datetime.datetime.now().date()
                if today > last_midnight_check:
                    auto_results = midnight_checkout_all_sessions()
                    if auto_results:
                        for r in auto_results:
                            logger.warning(f"⚠️ Midnight checkout: {r['name']} ({r['duration_str']})")
                    last_midnight_check = today
            
            # Skip frames để tiết kiệm CPU/RAM
            if ENABLE_ADAPTIVE_SKIP and adaptive_skip:
                should_process = adaptive_skip.should_process(frame_count)
            else:
                should_process = (frame_count % DEFAULT_FRAME_SKIP == 0)
            
            if not should_process:
                if not HEADLESS_MODE:
                    cv2.imshow("May Cham Cong", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Bắt đầu đo thời gian xử lý
            process_start_time = time.time()
            
            # Garbage collection định kỳ
            if frame_count % GC_INTERVAL == 0:
                gc.collect()

            # detection module nhận BGR (chuẩn OpenCV)
            detections = detect_faces(frame)
            
            # Danh sách người được nhận diện trong frame này
            recognized_this_frame = set()

            for det in detections:
                x, y, w, h = det['box']
                
                # Validate kích thước: Mặt quá nhỏ (<60px) thì bỏ qua để đỡ tốn CPU detect anti-spoof
                if w < 60 or h < 60:
                    continue

                face = frame[y:y+h, x:x+w]
                if face.size == 0: continue

                # --- BƯỚC 1: Anti-Spoofing (có thể tắt để test) ---
                if ENABLE_ANTISPOOF and anti is not None:
                    is_real = anti.is_live(face)
                else:
                    is_real = True  # Bỏ qua anti-spoof
                
                # --- BƯỚC 2: Recognition ---
                label, distance = recognizer.recognize(face, threshold=RECOGNITION_THRESHOLD)
                
                # Hiển thị distance để debug
                dist_text = f"d={distance:.2f}" if distance < float('inf') else ""
                
                if not is_real:
                    # FAKE: Màu đỏ
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    name_text = label if label else "Unknown"
                    cv2.putText(frame, f"FAKE - {name_text}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # REAL: Xử lý theo có nhận diện được hay không
                    if label is None:
                        # Người lạ (Vàng) - chưa đăng ký hoặc distance quá xa
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, f"Unknown {dist_text}", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        # Người quen (Xanh lá) - đã đăng ký
                        recognized_this_frame.add(label)
                        current_time = time.time()
                        
                        # --- BƯỚC 3: Logic giữ mặt ---
                        # Nếu chưa theo dõi người này, bắt đầu theo dõi
                        if label not in face_hold_tracker:
                            face_hold_tracker[label] = current_time
                        
                        # Tính thời gian đã giữ mặt
                        hold_duration = current_time - face_hold_tracker[label]
                        remaining = max(0, HOLD_TIME_SECONDS - hold_duration)
                        
                        # Hiển thị progress bar giữ mặt
                        progress = min(hold_duration / HOLD_TIME_SECONDS, 1.0)
                        bar_width = w
                        bar_height = 8
                        bar_y = y + h + 5
                        
                        # Vẽ khung và progress
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                        cv2.rectangle(frame, (x, bar_y), (x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
                        
                        if remaining > 0:
                            # Đang đếm ngược
                            cv2.putText(frame, f"{label} - Giu {remaining:.1f}s", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            # Đủ thời gian giữ mặt -> Chấm công
                            cv2.putText(frame, f"{label} {dist_text}", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # --- BƯỚC 4: Logic Chấm Công (Debounce) ---
                            if label not in last_checkin or (current_time - last_checkin[label] > COOLDOWN_SECONDS):
                                action = log_attendance(label)  # Returns 'check_in' or 'check_out'
                                last_checkin[label] = current_time
                                # Xóa khỏi tracker để tránh chấm công lại ngay
                                # (sẽ được thêm lại nếu người đó vẫn trong frame sau cooldown)
                                if label in face_hold_tracker:
                                    del face_hold_tracker[label]
                                
                                # Log ngắn gọn
                                symbol = "🟢" if action == 'check_in' else "🔴"
                                logger.info(f"{symbol} {label} - {action.upper().replace('_', '-')}")
                                
                                if not HEADLESS_MODE:
                                    if action == 'check_in':
                                        cv2.putText(frame, "CHECK-IN OK", (10, 50), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                    else:
                                        cv2.putText(frame, "CHECK-OUT OK", (10, 50), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            
            # Xóa tracker của những người không còn trong frame
            faces_to_remove = [name for name in face_hold_tracker if name not in recognized_this_frame]
            for name in faces_to_remove:
                # Chỉ xóa nếu không trong cooldown
                if name not in last_checkin or (time.time() - last_checkin.get(name, 0) > COOLDOWN_SECONDS):
                    del face_hold_tracker[name]

            # --- CẬP NHẬT ADAPTIVE FRAME SKIP ---
            if ENABLE_ADAPTIVE_SKIP and adaptive_skip:
                process_time = time.time() - process_start_time
                adaptive_skip.update(process_time)

            # --- PHẦN HIỂN THỊ VÀ ĐIỀU KHIỂN ---
            if HEADLESS_MODE:
                # HEADLESS MODE (Pi): Log định kỳ mỗi 5 phút
                current_time = time.time()
                if current_time - last_status_time > 300:  # 5 phút
                    # Thêm thông tin adaptive skip vào log
                    if ENABLE_ADAPTIVE_SKIP and adaptive_skip:
                        stats = adaptive_skip.get_stats()
                        logger.info(f"♻️ Running... Faces: {len(detections)}, Skip: {stats['current_skip']}, Avg: {stats['avg_process_ms']:.0f}ms")
                    else:
                        logger.info(f"♻️ Running... Faces detected: {len(detections)}")
                    last_status_time = current_time
            else:
                # GUI MODE (Windows): Hiển thị cửa sổ camera và xử lý phím
                
                # Hiển thị thông tin Adaptive Skip trên GUI
                if ENABLE_ADAPTIVE_SKIP and adaptive_skip:
                    stats = adaptive_skip.get_stats()
                    info_text = f"Skip:{stats['current_skip']} | {stats['avg_process_ms']:.0f}ms | ~{stats['effective_fps']:.1f}fps"
                    cv2.putText(frame, info_text, (10, frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow("May Cham Cong", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # Hiển thị danh sách đã đăng ký
                    print("\n📋 Database:")
                    names = recognizer.get_registered_names()
                    if names:
                        for i, name in enumerate(names, 1):
                            emb_count = len(recognizer.db[name]) if isinstance(recognizer.db[name], list) else 1
                            print(f"   {i}. {name} ({emb_count})")
                    else:
                        print("   (trống)")
                    print()
                elif key == ord('d'):
                    # Xóa người khỏi database
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
                    # Đăng ký khuôn mặt mới
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
                    
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng (Ctrl+C)")
    finally:
        cap.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("👋 Bye!")

if __name__ == "__main__":
    main()