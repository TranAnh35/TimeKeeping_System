# src/database.py
"""
Module quản lý database SQLite cho hệ thống chấm công THEO GIỜ.
Hỗ trợ check-in/check-out và tính tổng giờ làm việc.
Xử lý ca làm việc xuyên đêm (overnight).
Thread-safe cho multi-threaded access (main + web server).
"""
import sqlite3
import os
import threading
from datetime import datetime, timedelta

DB_PATH = "attendance.db"

# Thread-local storage cho connections
_local = threading.local()

def get_connection():
    """
    Tạo kết nối đến database (thread-safe).
    Mỗi thread sẽ có connection riêng.
    """
    if not hasattr(_local, 'connection') or _local.connection is None:
        _local.connection = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.connection.row_factory = sqlite3.Row  # Trả về dict thay vì tuple
    return _local.connection

def init_db():
    """Khởi tạo database và bảng nếu chưa có"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Bảng chấm công (log từng sự kiện)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            date DATE,
            time TIME,
            status TEXT DEFAULT 'check_in'
        )
    ''')
    
    # Bảng nhân viên
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Bảng sessions - lưu các phiên làm việc (check_in -> check_out)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            check_in_time DATETIME NOT NULL,
            check_out_time DATETIME,
            duration_minutes INTEGER DEFAULT 0,
            is_overnight INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def get_current_status(name):
    """
    Lấy trạng thái hiện tại của người dùng.
    Returns: ('checked_in', session_id, check_in_time) hoặc ('checked_out', None, None)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Tìm session chưa check-out
    cursor.execute('''
        SELECT id, check_in_time FROM work_sessions 
        WHERE name = ? AND check_out_time IS NULL
        ORDER BY check_in_time DESC LIMIT 1
    ''', (name,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return ('checked_in', row['id'], row['check_in_time'])
    return ('checked_out', None, None)

def _parse_datetime(dt_str):
    """Parse datetime string với hoặc không có microseconds"""
    if '.' in dt_str:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

def check_in(name):
    """
    Check-in: Bắt đầu phiên làm việc mới.
    Nếu đang có session mở, tự động check-out session cũ trước.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now()
    
    # Kiểm tra có session đang mở không
    status, session_id, _ = get_current_status(name)
    if status == 'checked_in':
        _do_checkout(cursor, session_id, now)
    
    cursor.execute('''
        INSERT INTO work_sessions (name, check_in_time)
        VALUES (?, ?)
    ''', (name, now))
    
    cursor.execute('''
        INSERT INTO attendance (name, timestamp, date, time, status)
        VALUES (?, ?, ?, ?, 'check_in')
    ''', (name, now, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    
    conn.commit()
    conn.close()
    return 'check_in'

def check_out(name):
    """
    Check-out: Kết thúc phiên làm việc.
    Tính duration và đánh dấu overnight nếu qua ngày.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now()
    
    # Tìm session đang mở
    cursor.execute('''
        SELECT id, check_in_time FROM work_sessions 
        WHERE name = ? AND check_out_time IS NULL
        ORDER BY check_in_time DESC LIMIT 1
    ''', (name,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        print(f"[Database] {name} chưa check-in!")
        return None
    
    _do_checkout(cursor, row['id'], now)
    
    cursor.execute('''
        INSERT INTO attendance (name, timestamp, date, time, status)
        VALUES (?, ?, ?, ?, 'check_out')
    ''', (name, now, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")))
    
    conn.commit()
    conn.close()
    return 'check_out'

def _do_checkout(cursor, session_id, checkout_time):
    """Helper: Thực hiện checkout cho một session"""
    cursor.execute('SELECT check_in_time FROM work_sessions WHERE id = ?', (session_id,))
    row = cursor.fetchone()
    if row:
        checkin_time = _parse_datetime(row['check_in_time'])
        duration = int((checkout_time - checkin_time).total_seconds() / 60)
        is_overnight = 1 if checkin_time.date() != checkout_time.date() else 0
        
        cursor.execute('''
            UPDATE work_sessions 
            SET check_out_time = ?, duration_minutes = ?, is_overnight = ?
            WHERE id = ?
        ''', (checkout_time, duration, is_overnight, session_id))

def smart_attendance(name):
    """
    Chấm công thông minh: Tự động check-in hoặc check-out dựa trên trạng thái.
    Returns: 'check_in' hoặc 'check_out'
    """
    status, _, _ = get_current_status(name)
    if status == 'checked_out':
        return check_in(name)
    else:
        return check_out(name)

def log_attendance(name, status="check_in"):
    """Backward compatible: Gọi smart_attendance"""
    return smart_attendance(name)

def add_employee(name):
    """Thêm nhân viên mới vào danh sách (nếu chưa có)"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT OR IGNORE INTO employees (name) VALUES (?)', (name,))
        conn.commit()
    finally:
        conn.close()

def remove_employee(name):
    """Xóa nhân viên khỏi danh sách"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM employees WHERE name = ?', (name,))
        conn.commit()
        deleted = cursor.rowcount > 0
        return deleted
    finally:
        conn.close()

def get_all_employees():
    """Lấy danh sách tất cả nhân viên"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, created_at FROM employees ORDER BY name')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def sync_employees_with_face_db(valid_names):
    """
    Đồng bộ bảng employees với danh sách tên từ face_db.pkl
    - Xóa những tên không còn trong face_db
    - Thêm những tên mới từ face_db
    - Reset auto-increment sequence
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Lấy employees hiện tại
    cursor.execute('SELECT name FROM employees')
    current_employees = set(row[0] for row in cursor.fetchall())
    valid_names_set = set(valid_names)
    
    # Xóa những tên không hợp lệ
    to_remove = current_employees - valid_names_set
    for name in to_remove:
        cursor.execute('DELETE FROM employees WHERE name = ?', (name,))
    
    # Thêm những tên mới
    to_add = valid_names_set - current_employees
    for name in to_add:
        cursor.execute('INSERT OR IGNORE INTO employees (name) VALUES (?)', (name,))
    
    # Reset sqlite_sequence cho employees
    cursor.execute('SELECT COUNT(*) FROM employees')
    count = cursor.fetchone()[0]
    cursor.execute('UPDATE sqlite_sequence SET seq = ? WHERE name = ?', (count, 'employees'))
    
    conn.commit()
    conn.close()
    
    # Return số lượng thay đổi để main.py quyết định log
    return {'synced': len(valid_names_set), 'added': len(to_add), 'removed': len(to_remove)}

def get_today_attendance():
    """Lấy danh sách chấm công hôm nay"""
    conn = get_connection()
    cursor = conn.cursor()
    
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        SELECT name, time, status FROM attendance 
        WHERE date = ? 
        ORDER BY timestamp DESC
    ''', (today,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_today_sessions():
    """Lấy các phiên làm việc hôm nay (hoặc phiên đang mở từ hôm qua)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    cursor.execute('''
        SELECT name, check_in_time, check_out_time, duration_minutes, is_overnight
        FROM work_sessions 
        WHERE date(check_in_time) = ? OR check_out_time IS NULL
        ORDER BY check_in_time DESC
    ''', (today,))
    
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        d = dict(row)
        if d['check_out_time'] is None:
            checkin = _parse_datetime(d['check_in_time'])
            d['duration_minutes'] = int((datetime.now() - checkin).total_seconds() / 60)
            d['status'] = 'working'
        else:
            d['status'] = 'completed'
        sessions.append(d)
    
    return sessions

def get_active_members():
    """Lấy danh sách người đang làm việc (chưa check-out)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, check_in_time 
        FROM work_sessions 
        WHERE check_out_time IS NULL
        ORDER BY check_in_time DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    active = []
    for row in rows:
        checkin = _parse_datetime(row['check_in_time'])
        duration = int((datetime.now() - checkin).total_seconds() / 60)
        active.append({
            'name': row['name'],
            'check_in_time': row['check_in_time'],
            'duration_minutes': duration
        })
    return active

def get_attendance_by_date(date_str):
    """Lấy chấm công theo ngày cụ thể (format: YYYY-MM-DD)"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, time, status FROM attendance 
        WHERE date = ? 
        ORDER BY timestamp DESC
    ''', (date_str,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_attendance_range(start_date, end_date):
    """Lấy chấm công trong khoảng thời gian"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, date, time, status FROM attendance 
        WHERE date BETWEEN ? AND ?
        ORDER BY timestamp DESC
    ''', (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_summary_by_person(start_date=None, end_date=None):
    """
    Thống kê TỔNG GIỜ LÀM của từng người trong khoảng thời gian.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if not start_date:
        now = datetime.now()
        start_date = now.replace(day=1).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
    
    cursor.execute('''
        SELECT 
            name, 
            COUNT(*) as total_sessions,
            SUM(duration_minutes) as total_minutes,
            AVG(duration_minutes) as avg_minutes,
            SUM(is_overnight) as overnight_sessions
        FROM work_sessions 
        WHERE date(check_in_time) BETWEEN ? AND ?
          AND check_out_time IS NOT NULL
        GROUP BY name
        ORDER BY total_minutes DESC
    ''', (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        total_mins = row['total_minutes'] or 0
        hours = total_mins // 60
        mins = total_mins % 60
        avg_mins = row['avg_minutes'] or 0
        
        result.append({
            'name': row['name'],
            'total_sessions': row['total_sessions'],
            'total_hours': f"{hours}h {mins}m",
            'total_minutes': total_mins,
            'avg_per_session': f"{int(avg_mins // 60)}h {int(avg_mins % 60)}m",
            'overnight_sessions': row['overnight_sessions'] or 0
        })
    
    return result

def get_daily_detail(name, date_str):
    """Chi tiết các lần chấm công trong ngày của 1 người"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT time, status FROM attendance 
        WHERE name = ? AND date = ?
        ORDER BY time ASC
    ''', (name, date_str))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def export_to_csv(output_path="export_attendance.csv", start_date=None, end_date=None):
    """Export dữ liệu phiên làm việc ra file CSV"""
    import csv
    
    if not start_date:
        now = datetime.now()
        start_date = now.replace(day=1).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Export work_sessions thay vì attendance raw
    cursor.execute('''
        SELECT name, check_in_time, check_out_time, duration_minutes, is_overnight
        FROM work_sessions 
        WHERE date(check_in_time) BETWEEN ? AND ?
        ORDER BY check_in_time DESC
    ''', (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, output_path)
    
    with open(full_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['Tên', 'Check-in', 'Check-out', 'Thời gian', 'Qua đêm'])
        for row in rows:
            duration = row['duration_minutes'] or 0
            hours = duration // 60
            mins = duration % 60
            writer.writerow([
                row['name'],
                row['check_in_time'],
                row['check_out_time'] or 'Đang làm',
                f"{hours}h {mins}m",
                'Có' if row['is_overnight'] else 'Không'
            ])
    
    return full_path