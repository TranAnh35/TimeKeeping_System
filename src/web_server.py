# src/web_server.py
"""
Web server đơn giản để xem chấm công từ xa qua WiFi.
Truy cập: http://<IP_Pi>:5000
"""
from flask import Flask, render_template_string, jsonify, request, send_file
from database import (
    get_today_attendance, 
    get_attendance_by_date,
    get_summary_by_person,
    get_all_employees,
    get_today_sessions,
    get_active_members,
    export_to_csv
)
from datetime import datetime
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Chấm Công Lab</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="refresh" content="30">
    <style>
        * { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #f5f7fa; color: #333; padding: 15px; min-height: 100vh; }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { color: #2e7d32; margin-bottom: 20px; text-align: center; font-size: 1.5em; }
        
        /* Stats Cards */
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }
        .stat-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .stat-card.active { border-color: #4CAF50; box-shadow: 0 0 10px rgba(76, 175, 80, 0.2); }
        .stat-num { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .stat-label { font-size: 0.8em; color: #666; margin-top: 5px; }
        
        /* Sections */
        .section { background: #fff; border-radius: 12px; padding: 15px; margin-bottom: 15px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .section-title { color: #2e7d32; font-size: 1em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        
        /* Members inline */
        .members-list { display: flex; flex-wrap: wrap; gap: 8px; }
        .member-tag { background: #e8f5e9; color: #2e7d32; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; }
        
        /* Active members - highlight */
        .active-list { display: flex; flex-wrap: wrap; gap: 10px; }
        .active-member { background: linear-gradient(135deg, #4CAF50, #43a047); color: #fff; padding: 10px 15px; border-radius: 10px; }
        .active-member .name { font-weight: bold; }
        .active-member .time { font-size: 0.8em; color: #c8e6c9; }
        
        /* Summary Table */
        .summary-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        .summary-table th { background: #4CAF50; color: #fff; padding: 10px; text-align: left; }
        .summary-table td { padding: 10px; border-bottom: 1px solid #e0e0e0; }
        .summary-table tr:hover { background: #f5f5f5; }
        .hours { color: #2e7d32; font-weight: bold; }
        
        /* Sessions - compact */
        .sessions-grid { display: grid; gap: 8px; }
        .session-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: #f5f5f5; border-radius: 8px; font-size: 0.85em; }
        .session-row.working { border-left: 3px solid #4CAF50; background: linear-gradient(90deg, rgba(76,175,80,0.1), #fff); }
        .session-row .name { font-weight: bold; min-width: 100px; color: #333; }
        .session-row .times { color: #666; flex: 1; }
        .session-row .duration { color: #2e7d32; font-weight: bold; }
        .badge { font-size: 0.7em; padding: 2px 6px; border-radius: 10px; }
        .badge-working { background: #4CAF50; color: #fff; }
        .badge-overnight { background: #ff9800; color: #fff; }
        
        /* Export */
        .export-form { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        input[type="date"] { background: #fff; border: 1px solid #ccc; color: #333; padding: 8px; border-radius: 6px; }
        .btn { background: #4CAF50; color: #fff; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #43a047; }
        
        .no-data { text-align: center; color: #999; padding: 20px; }
        .refresh-note { text-align: center; color: #999; font-size: 0.75em; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕐 Chấm Công Lab</h1>
        
        <!-- Stats -->
        <div class="stats">
            <div class="stat-card {{ 'active' if active_count > 0 else '' }}">
                <div class="stat-num">{{ active_count }}</div>
                <div class="stat-label">Đang làm việc</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{{ total_members }}</div>
                <div class="stat-label">Thành viên</div>
            </div>
        </div>
        
        <!-- Đang làm việc -->
        {% if active_members %}
        <div class="section">
            <div class="section-title">🟢 Đang làm việc</div>
            <div class="active-list">
                {% for m in active_members %}
                <div class="active-member">
                    <div class="name">{{ m.name }}</div>
                    <div class="time">{{ m.duration_hours }}h {{ m.duration_mins }}m</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
        
        <!-- Thành viên -->
        <div class="section">
            <div class="section-title">👥 Thành viên ({{ total_members }})</div>
            <div class="members-list">
                {% for name in member_names %}
                <span class="member-tag">{{ name }}</span>
                {% endfor %}
            </div>
        </div>
        
        <!-- Tổng hợp tháng -->
        <div class="section">
            <div class="section-title">📊 Tổng hợp tháng {{ current_month }}</div>
            {% if summary %}
            <table class="summary-table">
                <tr>
                    <th>Tên</th>
                    <th>Tổng giờ</th>
                    <th>Số phiên</th>
                    <th>TB/phiên</th>
                </tr>
                {% for row in summary %}
                <tr>
                    <td><strong>{{ row.name }}</strong></td>
                    <td class="hours">{{ row.total_hours }}</td>
                    <td>{{ row.total_sessions }}</td>
                    <td>{{ row.avg_per_session }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p class="no-data">Chưa có dữ liệu</p>
            {% endif %}
        </div>
        
        <!-- Phiên làm việc hôm nay -->
        <div class="section">
            <div class="section-title">📅 Hôm nay ({{ today }})</div>
            {% if sessions %}
            <div class="sessions-grid">
                {% for s in sessions %}
                <div class="session-row {{ 'working' if s.status == 'working' else '' }}">
                    <span class="name">{{ s.name }}</span>
                    <span class="times">{{ s.check_in_short }} → {{ s.check_out_short }}</span>
                    <span class="duration">{{ s.duration_str }}</span>
                    {% if s.status == 'working' %}<span class="badge badge-working">Đang làm</span>{% endif %}
                    {% if s.is_overnight %}<span class="badge badge-overnight">Qua đêm</span>{% endif %}
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p class="no-data">Chưa có phiên làm việc</p>
            {% endif %}
            <p class="refresh-note">Tự động cập nhật mỗi 30s</p>
        </div>
        
        <!-- Export -->
        <div class="section">
            <div class="section-title">📥 Export CSV</div>
            <form action="/export" method="get" class="export-form">
                <input type="date" name="start" value="{{ start_of_month }}">
                <span>→</span>
                <input type="date" name="end" value="{{ today }}">
                <button type="submit" class="btn">📥 Tải Excel</button>
            </form>
        </div>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    """Trang chủ - Dashboard"""
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_month = now.strftime("%m/%Y")
    start_of_month = now.replace(day=1).strftime("%Y-%m-%d")
    
    sessions = get_today_sessions()
    summary = get_summary_by_person()
    employees = get_all_employees()
    active_members_raw = get_active_members()
    
    active_members = []
    for m in active_members_raw:
        mins = m['duration_minutes']
        active_members.append({
            'name': m['name'],
            'duration_hours': mins // 60,
            'duration_mins': mins % 60
        })
    
    formatted_sessions = []
    for s in sessions:
        ci = s['check_in_time']
        ci_short = ci[11:16] if len(ci) > 16 else ci  # HH:MM
        
        co = s['check_out_time']
        if co:
            co_short = co[11:16] if len(co) > 16 else co
        else:
            co_short = "..."
        
        mins = s['duration_minutes'] or 0
        duration_str = f"{mins // 60}h {mins % 60}m"
        
        formatted_sessions.append({
            'name': s['name'],
            'check_in_short': ci_short,
            'check_out_short': co_short,
            'duration_str': duration_str,
            'status': s['status'],
            'is_overnight': s.get('is_overnight', 0)
        })
    
    member_names = [e['name'] for e in employees]
    
    return render_template_string(
        HTML_TEMPLATE,
        sessions=formatted_sessions,
        summary=summary,
        member_names=member_names,
        active_members=active_members,
        today=today,
        current_month=current_month,
        start_of_month=start_of_month,
        total_members=len(member_names),
        active_count=len(active_members),
        today_sessions=len(sessions)
    )

@app.route('/api/today')
def api_today():
    """API lấy sessions hôm nay (JSON)"""
    return jsonify(get_today_sessions())

@app.route('/api/active')
def api_active():
    """API lấy danh sách người đang làm việc"""
    return jsonify(get_active_members())

@app.route('/api/summary')
def api_summary():
    """API thống kê theo người"""
    return jsonify(get_summary_by_person())

@app.route('/api/members')
def api_members():
    """API lấy danh sách thành viên đã đăng ký"""
    employees = get_all_employees()
    return jsonify([{'name': e['name']} for e in employees])

@app.route('/export')
def export():
    """Export CSV và tải về"""
    start = request.args.get('start')
    end = request.args.get('end')
    
    filename = f"attendance_{start}_to_{end}.csv"
    filepath = export_to_csv(filename, start, end)
    
    return send_file(
        filepath,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

def get_local_ip():
    """Lấy địa chỉ IP local của máy"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def run_server(host='0.0.0.0', port=5000):
    """Chạy web server"""
    local_ip = get_local_ip()
    print(f"\n{'='*50}")
    print(f"🌐 Web Dashboard đang chạy!")
    print(f"📱 Truy cập từ điện thoại/máy khác:")
    print(f"   http://{local_ip}:{port}")
    print(f"💻 Truy cập local: http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    run_server()
