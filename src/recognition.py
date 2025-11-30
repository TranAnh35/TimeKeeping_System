# recognition.py
import cv2
import numpy as np
import os
import sys
import pickle
import platform as _platform

# Support both direct script execution and module import
try:
    from .tflite_helper import get_interpreter
except ImportError:
    from tflite_helper import get_interpreter

# Detect platform
_IS_PI = _platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")

class FaceRecognizer:
    def __init__(self, model_path="models/recognition/MobileFaceNet.tflite", db_path="face_db.pkl", 
                 enable_histogram_eq=None):
        """
        Args:
            model_path: Đường dẫn model TFLite
            db_path: Đường dẫn database embeddings
            enable_histogram_eq: Bật histogram equalization (None=auto: tắt trên Pi để tiết kiệm CPU)
        """
        self.interpreter = get_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']

        self.batch_size = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        self.db_path = db_path
        
        # Histogram equalization: tắt mặc định trên Pi để tiết kiệm CPU (~5-10ms/frame)
        if enable_histogram_eq is None:
            self.enable_histogram_eq = not _IS_PI
        else:
            self.enable_histogram_eq = enable_histogram_eq

        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)

            self._migrate_db_format()
        else:
            self.db = {}
    
    def _migrate_db_format(self):
        """Chuyển đổi database từ format cũ sang format mới nếu cần"""
        migrated = False
        for label, data in self.db.items():
            if isinstance(data, np.ndarray) and len(data.shape) == 1:
                self.db[label] = [data]
                migrated = True
        if migrated:
            self.save_db()
    
    def get_db_info(self):
        """Trả về thông tin database để hiển thị"""
        total_embeddings = sum(len(embs) if isinstance(embs, list) else 1 for embs in self.db.values())
        return len(self.db), total_embeddings

    def _preprocess_face(self, face_img):
        """
        Preprocess face với các bước cải thiện chất lượng:
        1. Resize về kích thước model
        2. Histogram equalization để cân bằng ánh sáng (optional, tắt trên Pi)
        3. Normalize theo chuẩn MobileFaceNet [-1, 1]
        """
        # Resize
        img = cv2.resize(face_img, (self.input_width, self.input_height))
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Histogram equalization (optional - tốn ~5-10ms trên Pi)
        # Hữu ích khi đeo kính hoặc ánh sáng không đều
        if self.enable_histogram_eq:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # Normalize to [-1, 1] (chuẩn MobileFaceNet)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        
        return img

    def get_embedding(self, face_img):
        """Trích xuất embedding từ ảnh khuôn mặt"""
        img = self._preprocess_face(face_img)
        
        input_data = np.stack([img, img], axis=0)  # Shape: [2, 112, 112, 3]
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        emb = output[0].astype(np.float32)
        
        # L2 normalize
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        
        # Giải phóng bộ nhớ tạm
        del img, input_data, output
        
        return emb

    def recognize(self, face_img, threshold=0.6):
        """
        Nhận diện khuôn mặt với chiến lược voting:
        - Tính distance với TẤT CẢ embeddings của mỗi người
        - Dùng AVERAGE của top-3 matches thay vì chỉ min
        - Giúp ổn định hơn khi có nhiều biến thể (kính, góc, ánh sáng)
        """
        if len(self.db) == 0:
            return None, float('inf')
            
        emb = self.get_embedding(face_img)
        
        # Tính average distance cho mỗi người
        person_scores = {}
        
        for label, embeddings in self.db.items():
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
            
            # Tính tất cả distances
            distances = [np.linalg.norm(db_emb - emb) for db_emb in embeddings]
            
            # Lấy average của top-3 (hoặc ít hơn nếu không đủ)
            distances.sort()
            top_k = min(3, len(distances))
            avg_dist = sum(distances[:top_k]) / top_k
            
            # Bonus: Nếu có ít nhất 1 match rất gần (<0.3), giảm distance
            min_dist = distances[0]
            if min_dist < 0.3:
                avg_dist = avg_dist * 0.8  # Boost 20%
            
            person_scores[label] = {
                'avg_dist': avg_dist,
                'min_dist': min_dist,
                'matches': len([d for d in distances if d < threshold])
            }
        
        # Kiểm tra nếu không có kết quả
        if not person_scores:
            return None, float('inf')
        
        # Tìm người có avg_dist nhỏ nhất
        best_label = min(person_scores, key=lambda x: person_scores[x]['avg_dist'])
        best_info = person_scores[best_label]
        
        # Quyết định dựa trên avg_dist
        if best_info['avg_dist'] < threshold:
            return best_label, best_info['min_dist']
        
        return None, best_info['min_dist']

    def recognize_with_confidence(self, face_img, threshold=0.6):
        """
        Phiên bản nâng cao: trả về confidence score và thông tin chi tiết
        """
        if len(self.db) == 0:
            return None, 0.0, {}
            
        emb = self.get_embedding(face_img)
        
        results = []
        for label, embeddings in self.db.items():
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
            
            distances = sorted([np.linalg.norm(db_emb - emb) for db_emb in embeddings])
            top_k = min(3, len(distances))
            avg_dist = sum(distances[:top_k]) / top_k
            
            # Convert distance to confidence (0-100%)
            # distance 0 -> 100%, distance >= threshold -> 0%
            confidence = max(0, (threshold - avg_dist) / threshold) * 100
            
            results.append({
                'label': label,
                'confidence': confidence,
                'avg_dist': avg_dist,
                'min_dist': distances[0]
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if results[0]['confidence'] > 0:
            return results[0]['label'], results[0]['confidence'], results
        return None, 0.0, results

    def add_face(self, label, face_img):
        """Thêm embedding mới cho người (không ghi đè, thêm vào list)"""
        emb = self.get_embedding(face_img)
        
        if label in self.db:
            if not isinstance(self.db[label], list):
                self.db[label] = [self.db[label]]
            self.db[label].append(emb)
        else:
            self.db[label] = [emb]

    def remove_face(self, label):
        """Xóa người khỏi database"""
        if label in self.db:
            del self.db[label]
            return True
        return False

    def get_registered_names(self):
        """Lấy danh sách tên đã đăng ký"""
        return list(self.db.keys())

    def save_db(self, db_path=None):
        """Lưu database ra file. Sử dụng self.db_path nếu không chỉ định."""
        if db_path is None:
            db_path = self.db_path
        with open(db_path, "wb") as f:
            pickle.dump(self.db, f)
