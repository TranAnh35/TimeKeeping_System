# recognition.py
from tflite_helper import get_interpreter
import cv2
import numpy as np
import os
import pickle

class FaceRecognizer:
    def __init__(self, model_path="models/recognition/MobileFaceNet.tflite", db_path="face_db.pkl"):
        self.interpreter = get_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']

        self.batch_size = input_shape[0]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
            
        print(f"[Recognition] Model input size: {self.input_width}x{self.input_height}")
        print(f"[Recognition] Input shape: {input_shape} (batch={self.batch_size})")
        
        self.db_path = db_path

        if os.path.exists(db_path):
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)

            self._migrate_db_format()
            self._print_db_info()
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
            print("[Recognition] Đã migrate database sang format mới (multi-embedding)")
    
    def _print_db_info(self):
        """In thông tin database"""
        total_embeddings = sum(len(embs) if isinstance(embs, list) else 1 for embs in self.db.values())
        print(f"[Recognition] Loaded {len(self.db)} người với tổng {total_embeddings} embeddings")
        for label, embs in self.db.items():
            count = len(embs) if isinstance(embs, list) else 1
            print(f"   - {label}: {count} ảnh")

    def get_embedding(self, face_img):
        img = cv2.resize(face_img, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Dùng float32 để tiết kiệm RAM (không dùng float64)
        img = img.astype(np.float32) / 255.0
        
        input_data = np.stack([img, img], axis=0)  # Shape: [2, 112, 112, 3]
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        emb = output[0].astype(np.float32)  # Đảm bảo float32
        
        emb = emb / np.linalg.norm(emb)
        
        # Giải phóng bộ nhớ tạm
        del img, input_data, output
        
        return emb

    def recognize(self, face_img, threshold=0.6):
        """Nhận diện khuôn mặt, trả về (label, distance) hoặc (None, distance)"""
        if len(self.db) == 0:
            return None, float('inf')
            
        emb = self.get_embedding(face_img)
        best_label, best_dist = None, float('inf')
        
        for label, embeddings in self.db.items():
            if not isinstance(embeddings, list):
                embeddings = [embeddings]
            
            for db_emb in embeddings:
                dist = np.linalg.norm(db_emb - emb)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
        
        if best_dist < threshold:
            return best_label, best_dist
        return None, best_dist

    def add_face(self, label, face_img):
        """Thêm embedding mới cho người (không ghi đè, thêm vào list)"""
        emb = self.get_embedding(face_img)
        
        if label in self.db:
            if not isinstance(self.db[label], list):
                self.db[label] = [self.db[label]]
            self.db[label].append(emb)
            print(f"[Recognition] Thêm ảnh cho {label}, tổng: {len(self.db[label])} ảnh")
        else:
            self.db[label] = [emb]
            print(f"[Recognition] Đăng ký mới: {label}")

    def remove_face(self, label):
        """Xóa người khỏi database"""
        if label in self.db:
            del self.db[label]
            print(f"[Recognition] Đã xóa: {label}")
            return True
        return False

    def get_registered_names(self):
        """Lấy danh sách tên đã đăng ký"""
        return list(self.db.keys())

    def save_db(self, db_path="face_db.pkl"):
        with open(db_path, "wb") as f:
            pickle.dump(self.db, f)
