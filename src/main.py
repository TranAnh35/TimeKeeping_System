# src/main.py
import cv2
import time
import datetime
import os
import gc
import sys
import platform
import threading
from detect import detect_faces
from recognition import FaceRecognizer
from database import log_attendance, add_employee, remove_employee, sync_employees_with_face_db, init_db, get_current_status

# --- TỰ ĐỘNG PHÁT HIỆN PLATFORM ---
IS_WINDOWS = platform.system() == "Windows"
IS_PI = platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")

# --- CẤU HÌNH ---
COOLDOWN_SECONDS = 60  # Thời gian chờ giữa 2 lần chấm công cho cùng 1 người
HOLD_TIME_SECONDS = 2.0  # Thời gian giữ mặt trong camera trước khi chấm công (giây)
ENABLE_WEB_SERVER = True  # Bật/tắt web dashboard
WEB_PORT = 5000
ENABLE_ANTISPOOF = False  # Tạm tắt anti-spoof để tiết kiệm RAM
RECOGNITION_THRESHOLD = 0.4  # Threshold cho recognition (càng nhỏ càng chặt)

# --- CHẾ ĐỘ HOẠT ĐỘNG ---
# Windows: GUI mode (có cửa sổ camera, có thể thêm/xóa thành viên)
# Pi: Headless mode (không GUI, chỉ chấm công, quản lý qua web)
HEADLESS_MODE = not IS_WINDOWS  # Tự động: Pi = headless, Windows = GUI

# --- CẤU HÌNH TỐI ƯU RAM (cho Pi 3) ---
LOW_MEMORY_MODE = IS_PI  # Tự động bật trên Pi
CAMERA_WIDTH = 640 if not LOW_MEMORY_MODE else 320
CAMERA_HEIGHT = 480 if not LOW_MEMORY_MODE else 240
FRAME_SKIP = 1 if not LOW_MEMORY_MODE else 2
GC_INTERVAL = 30

def start_web_server():
    """Chạy web server trong thread riêng"""
    try:
        from web_server import run_server
        run_server(host='0.0.0.0', port=WEB_PORT)
    except ImportError:
        print("[WARNING] Không tìm thấy web_server.py, bỏ qua web dashboard")
    except Exception as e:
        print(f"[WARNING] Không thể khởi động web server: {e}")

def main():
    # 0. Khởi tạo Database
    init_db()
    
    # 0.1 Khởi động Web Server (chạy nền)
    if ENABLE_WEB_SERVER:
        web_thread = threading.Thread(target=start_web_server, daemon=True)
        web_thread.start()
    
    # 1. Khởi tạo Camera và Models
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Giảm buffer size để tiết kiệm RAM
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        # Lazy loading: Chỉ load AntiSpoof nếu cần
        anti = None
        if ENABLE_ANTISPOOF:
            from antispoof import AntiSpoof
            anti = AntiSpoof()
        
        recognizer = FaceRecognizer()
        
        # Đồng bộ SQLite employees với face_db.pkl
        sync_employees_with_face_db(recognizer.get_registered_names())
        
        # Garbage collect sau khi load xong models
        gc.collect()
        
    except Exception as e:
        print(f"Lỗi khởi tạo Model: {e}")
        return

    # Dictionary lưu thời gian chấm công gần nhất: { 'Ten_NV': time_float }
    last_checkin = {} 
    
    # Dictionary theo dõi thời gian giữ mặt: { 'Ten_NV': first_seen_time }
    face_hold_tracker = {}

    print("Hệ thống chấm công sẵn sàng!")
    print(f"📍 Platform: {'Windows (GUI Mode)' if IS_WINDOWS else 'Raspberry Pi (Headless Mode)'}")
    print(f"⏱️  Giữ mặt trong camera {HOLD_TIME_SECONDS}s để chấm công.")
    if LOW_MEMORY_MODE:
        print(f"💾 Chế độ tiết kiệm RAM: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, skip {FRAME_SKIP} frame(s)")
    
    if not HEADLESS_MODE:
        print("\n🖥️  CHẾ ĐỘ DEBUG (Windows):")
        print("   Nhấn 'r' để đăng ký khuôn mặt mới.")
        print("   Nhấn 'd' để xóa người khỏi database.")
        print("   Nhấn 'l' để xem danh sách đã đăng ký.")
        print("   Nhấn 'q' để thoát.")
    else:
        print("\n🤖 CHẾ ĐỘ HEADLESS (Pi):")
        print("   Quản lý thành viên qua Web Dashboard")
        print("   Nhấn Ctrl+C để thoát.")
    
    if ENABLE_WEB_SERVER:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "localhost"
        print(f"\n🌐 Web Dashboard:")
        print(f"   📱 Từ điện thoại: http://{local_ip}:{WEB_PORT}")
        print(f"   💻 Local: http://localhost:{WEB_PORT}\n")

    frame_count = 0
    last_status_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được camera!")
                break
            
            frame_count += 1
            
            # Skip frames để tiết kiệm CPU/RAM
            if frame_count % FRAME_SKIP != 0:
                if not HEADLESS_MODE:
                    cv2.imshow("May Cham Cong", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
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
                                # Reset tracker sau khi chấm công
                                face_hold_tracker[label] = current_time + COOLDOWN_SECONDS
                                
                                # Log ra console (cho cả headless mode)
                                action_text = "CHECK-IN" if action == 'check_in' else "CHECK-OUT"
                                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ✅ {action_text}: {label}")
                                
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

            # --- PHẦN HIỂN THỊ VÀ ĐIỀU KHIỂN ---
            if HEADLESS_MODE:
                # HEADLESS MODE (Pi): Không hiển thị GUI, chỉ log ra console định kỳ
                current_time = time.time()
                if current_time - last_status_time > 60:  # Log mỗi 60 giây
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Đang chạy... Frame: {frame_count}")
                    last_status_time = current_time
            else:
                # GUI MODE (Windows): Hiển thị cửa sổ camera và xử lý phím
                cv2.imshow("May Cham Cong", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l'):
                    # Hiển thị danh sách đã đăng ký
                    print("\n" + "="*40)
                    print("📋 DANH SÁCH ĐÃ ĐĂNG KÝ:")
                    print("="*40)
                    names = recognizer.get_registered_names()
                    if names:
                        for i, name in enumerate(names, 1):
                            emb_count = len(recognizer.db[name]) if isinstance(recognizer.db[name], list) else 1
                            print(f"  {i}. {name} ({emb_count} ảnh)")
                    else:
                        print("  (Chưa có ai đăng ký)")
                    print("="*40 + "\n")
                elif key == ord('d'):
                    # Xóa người khỏi database
                    cv2.destroyAllWindows()
                    print("\n" + "="*40)
                    print("🗑️  XÓA NGƯỜI KHỎI DATABASE")
                    print("="*40)
                    names = recognizer.get_registered_names()
                    if not names:
                        print("Database trống, không có gì để xóa!")
                    else:
                        print("Danh sách hiện có:")
                        for i, name in enumerate(names, 1):
                            emb_count = len(recognizer.db[name]) if isinstance(recognizer.db[name], list) else 1
                            print(f"  {i}. {name} ({emb_count} ảnh)")
                        print("-"*40)
                        choice = input("Nhập tên cần xóa (hoặc Enter để hủy): ").strip()
                        if choice:
                            if recognizer.remove_face(choice):
                                recognizer.save_db()
                                remove_employee(choice)  # Xóa khỏi SQLite database
                                print(f"✅ Đã xóa '{choice}' khỏi database!")
                            else:
                                print(f"❌ Không tìm thấy '{choice}' trong database!")
                        else:
                            print("Đã hủy.")
                    print("="*40 + "\n")
                    cv2.namedWindow("May Cham Cong")
                elif key == ord('r'):
                    # Logic đăng ký đơn giản (lấy khuôn mặt to nhất trong khung hình)
                    if len(detections) > 0:
                        # Sắp xếp lấy box to nhất (diện tích w*h)
                        detections.sort(key=lambda d: d['box'][2] * d['box'][3], reverse=True)
                        
                        det = detections[0]
                        x, y, w, h = det['box']
                        face_reg = frame[y:y+h, x:x+w]
                        
                        cv2.destroyAllWindows()
                        name = input("Nhap ten nhan vien moi: ")
                        if name:
                            recognizer.add_face(name, face_reg)
                            recognizer.save_db()
                            add_employee(name)
                            print(f"Da dang ky thanh cong: {name}")
                        
                        cv2.namedWindow("May Cham Cong")
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Đã dừng bởi người dùng (Ctrl+C)")
    finally:
        cap.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("👋 Tạm biệt!")

if __name__ == "__main__":
    main()