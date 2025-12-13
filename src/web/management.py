# src/web_management.py
"""
Web Management API - Quản lý đăng ký/xóa khuôn mặt từ xa.

Thêm các endpoint:
- GET  /manage           - Trang quản lý (Web UI)
- GET  /api/employees    - Danh sách nhân viên với số embeddings
- POST /api/register     - Đăng ký khuôn mặt mới (upload ảnh)
- DELETE /api/employees/<name> - Xóa nhân viên

Workflow:
1. Truy cập http://<IP_Pi>:5000/manage từ laptop/điện thoại
2. Upload ảnh người mới → tự động detect face + extract embedding
3. Click nút xóa để remove người cũ
"""
import os
import io
import base64
import numpy as np
import cv2
from flask import Blueprint, request, jsonify, render_template_string

# Blueprint để dễ dàng tích hợp vào web_server.py
management_bp = Blueprint('management', __name__)

# Global references - sẽ được set từ main.py
_detector = None
_recognizer = None


def init_management(detector, recognizer):
    """
    Khởi tạo module management với detector và recognizer instances.
    Gọi từ main.py sau khi đã tạo các models.
    """
    global _detector, _recognizer
    _detector = detector
    _recognizer = recognizer
    print("[Web Management] Initialized with detector and recognizer")


# ============================================================================
# HTML TEMPLATE - Trang quản lý
# ============================================================================
MANAGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Quản Lý Thành Viên</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #f5f7fa; color: #333; padding: 20px; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #1976d2; margin-bottom: 25px; text-align: center; }
        
        /* Sections */
        .section { background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 20px; 
                   border: 1px solid #e0e0e0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .section-title { color: #1976d2; font-size: 1.2em; margin-bottom: 15px; 
                        display: flex; align-items: center; gap: 10px; }
        
        /* Form đăng ký */
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 600; color: #555; }
        .form-group input[type="text"] { width: 100%; padding: 12px; border: 1px solid #ccc; 
                                         border-radius: 8px; font-size: 1em; }
        .form-group input[type="file"] { width: 100%; padding: 10px; border: 2px dashed #ccc; 
                                         border-radius: 8px; cursor: pointer; background: #fafafa; }
        .form-group input[type="file"]:hover { border-color: #1976d2; background: #e3f2fd; }
        
        /* Buttons */
        .btn { padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; 
               font-weight: bold; font-size: 1em; transition: all 0.2s; }
        .btn-primary { background: #1976d2; color: #fff; }
        .btn-primary:hover { background: #1565c0; }
        .btn-primary:disabled { background: #90caf9; cursor: not-allowed; }
        .btn-danger { background: #d32f2f; color: #fff; padding: 8px 16px; font-size: 0.9em; }
        .btn-danger:hover { background: #c62828; }
        .btn-secondary { background: #757575; color: #fff; }
        .btn-secondary:hover { background: #616161; }
        
        /* Preview images */
        .preview-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
        .preview-item { position: relative; }
        .preview-item img { width: 100px; height: 100px; object-fit: cover; border-radius: 8px; border: 2px solid #ddd; }
        .preview-item .remove { position: absolute; top: -8px; right: -8px; background: #d32f2f; 
                                color: #fff; border: none; border-radius: 50%; width: 24px; height: 24px; 
                                cursor: pointer; font-size: 14px; line-height: 22px; }
        
        /* Employee list */
        .employee-list { display: grid; gap: 10px; }
        .employee-card { display: flex; justify-content: space-between; align-items: center; 
                        padding: 15px; background: #f5f5f5; border-radius: 8px; }
        .employee-info { display: flex; align-items: center; gap: 15px; }
        .employee-name { font-weight: bold; font-size: 1.1em; }
        .employee-count { color: #666; font-size: 0.9em; }
        
        /* Status messages */
        .status { padding: 15px; border-radius: 8px; margin-bottom: 15px; display: none; }
        .status.success { background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; display: block; }
        .status.error { background: #ffebee; color: #c62828; border: 1px solid #ef9a9a; display: block; }
        .status.loading { background: #e3f2fd; color: #1976d2; border: 1px solid #90caf9; display: block; }
        
        /* Progress */
        .progress-bar { width: 100%; height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; margin-top: 10px; display: none; }
        .progress-bar.active { display: block; }
        .progress-bar .fill { height: 100%; background: #1976d2; transition: width 0.3s; }
        
        /* Nav */
        .nav { margin-bottom: 20px; text-align: center; }
        .nav a { color: #1976d2; text-decoration: none; margin: 0 15px; }
        .nav a:hover { text-decoration: underline; }
        
        /* Responsive */
        @media (max-width: 600px) {
            .employee-card { flex-direction: column; align-items: flex-start; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👤 Quản Lý Thành Viên</h1>
        
        <div class="nav">
            <a href="/">← Trang chấm công</a>
            <a href="/manage">🔄 Làm mới</a>
        </div>
        
        <!-- Status message -->
        <div id="status" class="status"></div>
        
        <!-- Section: Đăng ký mới -->
        <div class="section">
            <div class="section-title">➕ Đăng Ký Thành Viên Mới</div>
            
            <form id="registerForm">
                <div class="form-group">
                    <label for="name">Tên thành viên:</label>
                    <input type="text" id="name" name="name" placeholder="Nhập tên (VD: Nguyen Van A)" required>
                </div>
                
                <div class="form-group">
                    <label for="images">Chọn ảnh khuôn mặt (1-10 ảnh):</label>
                    <input type="file" id="images" name="images" accept="image/*" multiple required>
                    <small style="color: #666;">Nên chọn 3-5 ảnh với các góc độ khác nhau để tăng độ chính xác</small>
                </div>
                
                <div class="preview-container" id="preview"></div>
                
                <div class="progress-bar" id="progressBar">
                    <div class="fill" id="progressFill" style="width: 0%"></div>
                </div>
                
                <button type="submit" class="btn btn-primary" id="submitBtn">📸 Đăng Ký</button>
            </form>
        </div>
        
        <!-- Section: Danh sách thành viên -->
        <div class="section">
            <div class="section-title">👥 Danh Sách Thành Viên ({{ total }})</div>
            
            {% if employees %}
            <div class="employee-list">
                {% for emp in employees %}
                <div class="employee-card" id="card-{{ emp.name | replace(' ', '_') }}">
                    <div class="employee-info">
                        <span class="employee-name">{{ emp.name }}</span>
                        <span class="employee-count">{{ emp.embedding_count }} ảnh đã đăng ký</span>
                    </div>
                    <button class="btn btn-danger" onclick="deleteEmployee('{{ emp.name }}')">🗑️ Xóa</button>
                </div>
                {% endfor %}
            </div>
            {% else %}
            <p style="text-align: center; color: #999; padding: 20px;">Chưa có thành viên nào được đăng ký</p>
            {% endif %}
        </div>
    </div>
    
    <script>
        const statusDiv = document.getElementById('status');
        const registerForm = document.getElementById('registerForm');
        const imageInput = document.getElementById('images');
        const previewDiv = document.getElementById('preview');
        const progressBar = document.getElementById('progressBar');
        const progressFill = document.getElementById('progressFill');
        const submitBtn = document.getElementById('submitBtn');
        
        // Preview images
        let selectedFiles = [];
        
        imageInput.addEventListener('change', function(e) {
            selectedFiles = Array.from(e.target.files);
            previewDiv.innerHTML = '';
            
            selectedFiles.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const item = document.createElement('div');
                    item.className = 'preview-item';
                    item.innerHTML = `
                        <img src="${e.target.result}" alt="Preview">
                        <button class="remove" onclick="removeImage(${index})">&times;</button>
                    `;
                    previewDiv.appendChild(item);
                };
                reader.readAsDataURL(file);
            });
        });
        
        function removeImage(index) {
            selectedFiles.splice(index, 1);
            // Re-render preview
            previewDiv.innerHTML = '';
            selectedFiles.forEach((file, i) => {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const item = document.createElement('div');
                    item.className = 'preview-item';
                    item.innerHTML = `
                        <img src="${e.target.result}" alt="Preview">
                        <button class="remove" onclick="removeImage(${i})">&times;</button>
                    `;
                    previewDiv.appendChild(item);
                };
                reader.readAsDataURL(file);
            });
        }
        
        function showStatus(message, type) {
            statusDiv.className = 'status ' + type;
            statusDiv.innerHTML = message;
            if (type !== 'loading') {
                setTimeout(() => { statusDiv.style.display = 'none'; }, 5000);
            }
        }
        
        // Submit form
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const name = document.getElementById('name').value.trim();
            if (!name) {
                showStatus('❌ Vui lòng nhập tên', 'error');
                return;
            }
            
            if (selectedFiles.length === 0) {
                showStatus('❌ Vui lòng chọn ít nhất 1 ảnh', 'error');
                return;
            }
            
            if (selectedFiles.length > 10) {
                showStatus('❌ Chỉ được chọn tối đa 10 ảnh', 'error');
                return;
            }
            
            // Disable button and show progress
            submitBtn.disabled = true;
            progressBar.classList.add('active');
            showStatus('⏳ Đang xử lý...', 'loading');
            
            let successCount = 0;
            let errorMessages = [];
            
            for (let i = 0; i < selectedFiles.length; i++) {
                progressFill.style.width = ((i + 1) / selectedFiles.length * 100) + '%';
                
                const formData = new FormData();
                formData.append('name', name);
                formData.append('image', selectedFiles[i]);
                
                try {
                    const response = await fetch('/api/register', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    
                    if (result.success) {
                        successCount++;
                    } else {
                        errorMessages.push(`Ảnh ${i+1}: ${result.error}`);
                    }
                } catch (error) {
                    errorMessages.push(`Ảnh ${i+1}: Lỗi kết nối`);
                }
            }
            
            // Reset form
            submitBtn.disabled = false;
            progressBar.classList.remove('active');
            progressFill.style.width = '0%';
            
            if (successCount > 0) {
                let msg = `✅ Đã đăng ký ${successCount}/${selectedFiles.length} ảnh cho "${name}"`;
                if (errorMessages.length > 0) {
                    msg += '<br><small>Lỗi: ' + errorMessages.join(', ') + '</small>';
                }
                showStatus(msg, 'success');
                // Reload page after 2s
                setTimeout(() => { location.reload(); }, 2000);
            } else {
                showStatus('❌ Không đăng ký được ảnh nào.<br>' + errorMessages.join('<br>'), 'error');
            }
        });
        
        // Delete employee
        async function deleteEmployee(name) {
            if (!confirm(`Bạn có chắc muốn xóa "${name}"?\\n\\nThao tác này không thể hoàn tác!`)) {
                return;
            }
            
            showStatus('⏳ Đang xóa...', 'loading');
            
            try {
                const response = await fetch('/api/employees/' + encodeURIComponent(name), {
                    method: 'DELETE'
                });
                const result = await response.json();
                
                if (result.success) {
                    showStatus(`✅ Đã xóa "${name}"`, 'success');
                    // Remove card from UI
                    const card = document.getElementById('card-' + name.replace(/ /g, '_'));
                    if (card) card.remove();
                } else {
                    showStatus('❌ ' + result.error, 'error');
                }
            } catch (error) {
                showStatus('❌ Lỗi kết nối: ' + error, 'error');
            }
        }
    </script>
</body>
</html>
'''


# ============================================================================
# API ENDPOINTS
# ============================================================================

@management_bp.route('/manage')
def manage_page():
    """Trang quản lý thành viên (Web UI)"""
    employees = []
    
    if _recognizer is not None:
        # Lấy danh sách từ recognizer database
        for name in _recognizer.get_registered_names():
            count = _recognizer.get_embedding_count(name)
            employees.append({
                'name': name,
                'embedding_count': count
            })
    
    # Sort by name
    employees.sort(key=lambda x: x['name'])
    
    return render_template_string(
        MANAGE_TEMPLATE,
        employees=employees,
        total=len(employees)
    )


@management_bp.route('/api/employees', methods=['GET'])
def api_get_employees():
    """
    GET /api/employees
    Lấy danh sách thành viên với số embeddings.
    
    Response:
        [{"name": "Nguyen Van A", "embedding_count": 5}, ...]
    """
    if _recognizer is None:
        return jsonify({'error': 'Recognizer chưa được khởi tạo'}), 500
    
    employees = []
    for name in _recognizer.get_registered_names():
        employees.append({
            'name': name,
            'embedding_count': _recognizer.get_embedding_count(name)
        })
    
    return jsonify(employees)


@management_bp.route('/api/register', methods=['POST'])
def api_register():
    """
    POST /api/register
    Đăng ký khuôn mặt mới.
    
    Form data:
        - name: Tên người dùng
        - image: File ảnh (JPEG/PNG)
    
    Response:
        {"success": true, "message": "..."}
        or
        {"success": false, "error": "..."}
    """
    if _detector is None or _recognizer is None:
        return jsonify({'success': False, 'error': 'System chưa sẵn sàng'}), 500
    
    # Validate input
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Thiếu tên người dùng'}), 400
    
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'Thiếu file ảnh'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'success': False, 'error': 'Chưa chọn file'}), 400
    
    try:
        # Đọc ảnh từ upload
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'error': 'Không thể đọc file ảnh'}), 400
        
        # Detect faces
        faces = _detector.detect_faces(frame)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'Không tìm thấy khuôn mặt trong ảnh'}), 400
        
        if len(faces) > 1:
            return jsonify({'success': False, 'error': f'Phát hiện {len(faces)} khuôn mặt. Chỉ cho phép 1 khuôn mặt/ảnh'}), 400
        
        # Crop face
        face = faces[0]
        x, y, w, h = face['box']
        
        # Expand box slightly for better recognition
        padding = int(max(w, h) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        face_img = frame[y:y+h, x:x+w]
        
        if face_img.size == 0:
            return jsonify({'success': False, 'error': 'Crop khuôn mặt thất bại'}), 400
        
        # Add to database
        success = _recognizer.add_face(name, face_img)
        
        if success:
            count = _recognizer.get_embedding_count(name)
            return jsonify({
                'success': True,
                'message': f'Đã thêm ảnh cho "{name}". Tổng: {count} embeddings',
                'embedding_count': count
            })
        else:
            return jsonify({'success': False, 'error': 'Không thể trích xuất embedding từ ảnh'}), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@management_bp.route('/api/employees/<name>', methods=['DELETE'])
def api_delete_employee(name):
    """
    DELETE /api/employees/<name>
    Xóa thành viên khỏi database.
    
    Response:
        {"success": true, "message": "..."}
        or
        {"success": false, "error": "..."}
    """
    if _recognizer is None:
        return jsonify({'success': False, 'error': 'Recognizer chưa được khởi tạo'}), 500
    
    name = name.strip()
    
    if name not in _recognizer.get_registered_names():
        return jsonify({'success': False, 'error': f'Không tìm thấy "{name}" trong database'}), 404
    
    try:
        success = _recognizer.remove_face(name)
        
        if success:
            # Cũng xóa khỏi employee database nếu có
            try:
                from ..data.database import remove_employee
            except ImportError:
                from data.database import remove_employee
            
            remove_employee(name)
            
            return jsonify({
                'success': True,
                'message': f'Đã xóa "{name}" khỏi hệ thống'
            })
        else:
            return jsonify({'success': False, 'error': 'Xóa thất bại'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def register_management_routes(app):
    """
    Đăng ký routes vào Flask app.
    Gọi từ web_server.py hoặc main.py.
    """
    app.register_blueprint(management_bp)
    print("[Web Management] Routes registered: /manage, /api/register, /api/employees")
