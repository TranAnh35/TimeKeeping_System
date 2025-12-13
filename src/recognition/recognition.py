# src/recognition/recognition.py
"""
Face Recognition module - INT8 Optimized
=========================================
Model: MobileFaceNet_int8.tflite (INT8 Quantized)
Input: [batch, 112, 112, 3] int8, quantized
Output: [batch, 128] int8 embedding (dequantized to float32)

Database: face_db.pkl (embedding dim = 128)

Thread-safe: Sử dụng Lock cho TFLite inference và database operations.
"""
import cv2
import numpy as np
import os
import sys
import pickle
import threading
import platform as _platform

# Support both direct script execution and module import
try:
    from ..core.tflite_helper import get_interpreter
    from ..core.settings import settings
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.tflite_helper import get_interpreter
    from core.settings import settings

# ============================================================================
# CONFIGURATION - INT8 Model
# ============================================================================
DEFAULT_MODEL_PATH = "models/recognition/MobileFaceNet_int8.tflite"
DEFAULT_DB_PATH = "face_db.pkl"  # Database for INT8 model

# Model specs (từ quantization)
INPUT_HEIGHT = 112
INPUT_WIDTH = 112
EMBEDDING_DIM = 128  # INT8 model output dimension (KHÁC với Float32!)

# Default quantization parameters (sẽ được cập nhật từ model)
DEFAULT_INPUT_SCALE = 0.007874015718698502
DEFAULT_INPUT_ZERO_POINT = 0
DEFAULT_OUTPUT_SCALE = 0.07505225390195847
DEFAULT_OUTPUT_ZERO_POINT = 0

# Platform detection
_IS_PI = _platform.system() == "Linux" and os.path.exists("/proc/device-tree/model")


class FaceRecognizer:
    """
    Face Recognizer sử dụng MobileFaceNet INT8 Quantized.
    Tối ưu cho Raspberry Pi.
    
    Embedding dim = 128
    """
    
    def __init__(self, model_path=None, db_path=None, enable_histogram_eq=None):
        """
        Args:
            model_path: Đường dẫn model TFLite INT8
            db_path: Đường dẫn database embeddings (default: face_db.pkl)
            enable_histogram_eq: Bật histogram equalization (None=auto: tắt trên Pi)
        """
        # Thread-safety locks
        self._inference_lock = threading.Lock()  # Lock cho TFLite inference
        self._db_lock = threading.Lock()  # Lock cho database operations
        
        # Hot-reload tracking
        self._db_mtime = 0.0  # Last modification time of db file
        
        # Use defaults if not specified
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if db_path is None:
            db_path = DEFAULT_DB_PATH
            
        self.model_path = model_path
        self.db_path = db_path
        
        # Default values (sẽ được cập nhật sau khi load model)
        self.batch_size = 1
        self.input_height = INPUT_HEIGHT
        self.input_width = INPUT_WIDTH
        self._embedding_dim = EMBEDDING_DIM
        
        # Quantization parameters
        self._input_scale = DEFAULT_INPUT_SCALE
        self._input_zero_point = DEFAULT_INPUT_ZERO_POINT
        self._output_scale = DEFAULT_OUTPUT_SCALE
        self._output_zero_point = DEFAULT_OUTPUT_ZERO_POINT
        
        # Load model
        self.interpreter = get_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input dtype and verify INT8
        self._input_dtype = self.input_details[0]['dtype']
        if self._input_dtype not in [np.int8, np.uint8]:
            print(f"[CẢNH BÁO] Model không phải INT8, dtype={self._input_dtype}")
        
        # Get input shape from model
        raw_shape = self.input_details[0].get('shape', [1, INPUT_HEIGHT, INPUT_WIDTH, 3])
        input_shape = tuple(int(dim) for dim in raw_shape)
        
        self.batch_size = max(1, int(input_shape[0])) if len(input_shape) >= 4 else 1
        self.input_height = int(input_shape[1]) if len(input_shape) >= 2 else INPUT_HEIGHT
        self.input_width = int(input_shape[2]) if len(input_shape) >= 3 else INPUT_WIDTH
        
        # Get embedding dimension from output
        output_shape = self.output_details[0].get('shape', [1, EMBEDDING_DIM])
        self._embedding_dim = int(output_shape[-1]) if len(output_shape) >= 2 else EMBEDDING_DIM
        
        # Get quantization parameters from model
        input_quant = self.input_details[0].get('quantization_parameters', {})
        if input_quant.get('scales'):
            self._input_scale = float(input_quant['scales'][0])
        if input_quant.get('zero_points'):
            self._input_zero_point = int(input_quant['zero_points'][0])
        
        output_quant = self.output_details[0].get('quantization_parameters', {})
        if output_quant.get('scales'):
            self._output_scale = float(output_quant['scales'][0])
        if output_quant.get('zero_points'):
            self._output_zero_point = int(output_quant['zero_points'][0])
        
        # Indices for fast access
        self._input_index = self.input_details[0]['index']
        self._output_index = self.output_details[0]['index']
        
        # Reuse input buffer (INT8 hoặc Float32 tùy model)
        buffer_dtype = self._input_dtype if self._input_dtype in [np.int8, np.uint8] else np.float32
        self._input_buffer = np.zeros(
            (self.batch_size, self.input_height, self.input_width, 3),
            dtype=buffer_dtype
        )
        
        # Histogram equalization: tắt trên Pi để tiết kiệm CPU
        if enable_histogram_eq is None:
            self.enable_histogram_eq = not _IS_PI
        else:
            self.enable_histogram_eq = enable_histogram_eq
        
        # Load database
        self._load_db()
        
        print(f"[INT8 Recognizer] Model: {model_path}")
        print(f"[INT8 Recognizer] Database: {db_path}")
        print(f"[INT8 Recognizer] Embedding dim: {self._embedding_dim}")
        print(f"[INT8 Recognizer] Input scale={self._input_scale}, zp={self._input_zero_point}")
    
    def _load_db(self):
        """Load database từ file và cập nhật mtime tracking"""
        self.db = {}
        if os.path.exists(self.db_path):
            try:
                self._db_mtime = os.path.getmtime(self.db_path)
                with open(self.db_path, "rb") as f:
                    self.db = pickle.load(f)
                self._migrate_db_format()
            except Exception as e:
                print(f"[CẢNH BÁO] Không thể load database: {e}")
                self.db = {}
                self._db_mtime = 0.0
    
    def reload_db_if_changed(self):
        """
        Kiểm tra và reload database nếu file đã thay đổi.
        Gọi hàm này định kỳ để hot-reload database.
        
        Returns:
            bool: True nếu database đã được reload
        """
        if not os.path.exists(self.db_path):
            return False
        
        try:
            current_mtime = os.path.getmtime(self.db_path)
            if current_mtime > self._db_mtime:
                with self._db_lock:
                    # Double-check inside lock
                    current_mtime = os.path.getmtime(self.db_path)
                    if current_mtime > self._db_mtime:
                        old_count = len(self.db)
                        self._load_db()
                        new_count = len(self.db)
                        print(f"[Hot-Reload] Database reloaded: {old_count} -> {new_count} người")
                        return True
        except Exception as e:
            print(f"[Hot-Reload] Lỗi check database: {e}")
        
        return False

    def _migrate_db_format(self):
        """Migrate database từ format cũ sang format mới nếu cần"""
        migrated = False
        for label, data in list(self.db.items()):
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            
            # Kiểm tra embedding dimension
            if arr.shape[-1] != self._embedding_dim:
                print(f"[CẢNH BÁO] {label}: embedding dim {arr.shape[-1]} != {self._embedding_dim}, bỏ qua")
                del self.db[label]
                migrated = True
                continue
                
            if not isinstance(data, np.ndarray) or arr.ndim != data.ndim:
                self.db[label] = arr
                migrated = True
        
        if migrated:
            self.save_db()

    def _preprocess(self, face_img):
        """
        Preprocess face image cho INT8 model.
        
        Pipeline:
        1. Resize về 112x112
        2. BGR -> RGB
        3. (Optional) Histogram equalization
        4. Normalize về [-1, 1] rồi quantize
        """
        # Resize
        img = cv2.resize(face_img, (self.input_width, self.input_height))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Histogram equalization (optional)
        if self.enable_histogram_eq:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        # Normalize và quantize
        if self._input_dtype == np.int8:
            # Normalize về [-1, 1] rồi quantize về int8
            img_float = (img.astype(np.float32) - 127.5) / 127.5
            img = np.clip(
                img_float / self._input_scale + self._input_zero_point,
                -128, 127
            ).astype(np.int8)
        elif self._input_dtype == np.uint8:
            img = img.astype(np.uint8)
        else:
            # Fallback to float32
            img = (img.astype(np.float32) - 127.5) / 127.5
        
        return img

    def _dequantize(self, output):
        """Dequantize INT8 output về float32"""
        if output.dtype in [np.int8, np.uint8]:
            return (output.astype(np.float32) - self._output_zero_point) * self._output_scale
        return output.astype(np.float32)

    def get_embedding(self, face_img):
        """
        Trích xuất embedding từ ảnh khuôn mặt.
        Thread-safe với inference lock.
        
        Args:
            face_img: BGR image (numpy array)
            
        Returns:
            numpy array shape (128,), L2 normalized
        """
        img = self._preprocess(face_img)
        
        # Thread-safe inference
        with self._inference_lock:
            # Fill buffer
            self._input_buffer[:] = img
            
            # Inference
            self.interpreter.set_tensor(self._input_index, self._input_buffer)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self._output_index)
            
            # Dequantize
            output = self._dequantize(output)
            
            # Get first embedding
            emb = np.array(output[0], dtype=np.float32, copy=True)
        
        # L2 normalize (outside lock - pure numpy, thread-safe)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        
        return emb

    def recognize(self, face_img, threshold=0.6):
        """
        Nhận diện khuôn mặt.
        
        Args:
            face_img: BGR image
            threshold: Ngưỡng distance (nhỏ hơn = match)
            
        Returns:
            (label, distance) hoặc (None, inf) nếu không match
        """
        # Thread-safe check db size
        with self._db_lock:
            if len(self.db) == 0:
                return None, float('inf')
            # Copy db keys to avoid modification during iteration
            db_snapshot = {k: np.array(v, copy=True) for k, v in self.db.items()}
        
        emb = self.get_embedding(face_img)
        
        best_label = None
        best_avg_dist = float('inf')
        best_min_dist = float('inf')
        
        # Iterate over snapshot (thread-safe)
        for label, embeddings_arr in db_snapshot.items():
            if embeddings_arr.ndim == 1:
                embeddings_arr = embeddings_arr[np.newaxis, :]
            if embeddings_arr.size == 0:
                continue
            
            # Tính L2 distance
            distances = np.linalg.norm(embeddings_arr - emb, axis=1)
            
            # Top-3 voting
            top_k = min(3, len(distances))
            top_distances = np.partition(distances, top_k - 1)[:top_k]
            avg_dist = float(np.mean(top_distances))
            min_dist = float(np.min(distances))
            
            # Boost nếu có match rất tốt
            if min_dist < 0.3:
                avg_dist *= 0.8
            
            if avg_dist < best_avg_dist:
                best_avg_dist = avg_dist
                best_min_dist = min_dist
                best_label = label
        
        if best_avg_dist < threshold:
            return best_label, best_min_dist
        
        return None, best_min_dist

    def recognize_with_confidence(self, face_img, threshold=0.6):
        """
        Nhận diện với confidence score.
        Thread-safe với db_lock.
        
        Returns:
            (label, confidence, all_results)
        """
        # Thread-safe snapshot
        with self._db_lock:
            if len(self.db) == 0:
                return None, 0.0, []
            db_snapshot = {k: np.array(v, copy=True) for k, v in self.db.items()}
        
        emb = self.get_embedding(face_img)
        results = []
        
        for label, embeddings_arr in db_snapshot.items():
            if embeddings_arr.ndim == 1:
                embeddings_arr = embeddings_arr[np.newaxis, :]
            if embeddings_arr.size == 0:
                continue
            
            distances = np.linalg.norm(embeddings_arr - emb, axis=1)
            
            top_k = min(3, len(distances))
            top_distances = np.partition(distances, top_k - 1)[:top_k]
            avg_dist = float(np.mean(top_distances))
            
            # Convert distance to confidence (0-100%)
            confidence = max(0.0, (threshold - avg_dist) / threshold) * 100
            
            results.append({
                'label': label,
                'confidence': confidence,
                'avg_dist': avg_dist,
                'min_dist': float(np.min(distances))
            })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if results and results[0]['confidence'] > 0:
            return results[0]['label'], results[0]['confidence'], results
        return None, 0.0, results

    def add_face(self, label, face_img):
        """Thêm embedding mới cho người. Thread-safe. Returns True if success."""
        emb = self.get_embedding(face_img)
        if emb is None:
            return False
            
        emb = emb[np.newaxis, :]
        
        with self._db_lock:
            if label in self.db:
                existing = np.asarray(self.db[label], dtype=np.float32)
                if existing.ndim == 1:
                    existing = existing[np.newaxis, :]
                self.db[label] = np.concatenate((existing, emb), axis=0)
            else:
                self.db[label] = emb
            
            # Auto-save (inside lock to prevent race)
            self._save_db_unsafe()
        return True

    def remove_face(self, label):
        """Xóa người khỏi database. Thread-safe. Returns True if success."""
        with self._db_lock:
            if label in self.db:
                del self.db[label]
                self._save_db_unsafe()
                return True
        return False

    def get_embedding_count(self, label):
        """Lấy số lượng embeddings của một người. Thread-safe."""
        with self._db_lock:
            if label not in self.db:
                return 0
            data = self.db[label]
            if isinstance(data, np.ndarray):
                return data.shape[0] if data.ndim == 2 else 1
            return 1

    def get_registered_names(self):
        """Lấy danh sách tên đã đăng ký. Thread-safe."""
        with self._db_lock:
            return list(self.db.keys())

    def get_db_info(self):
        """Trả về thông tin database. Thread-safe."""
        with self._db_lock:
            total_embeddings = 0
            for embs in self.db.values():
                if isinstance(embs, np.ndarray):
                    total_embeddings += embs.shape[0]
                else:
                    total_embeddings += 1
            return len(self.db), total_embeddings

    def _save_db_unsafe(self, db_path=None):
        """Internal save - caller must hold _db_lock"""
        if db_path is None:
            db_path = self.db_path
        with open(db_path, "wb") as f:
            pickle.dump(self.db, f)
        # Update mtime để tránh self-reload
        self._db_mtime = os.path.getmtime(db_path)

    def save_db(self, db_path=None):
        """Lưu database ra file. Thread-safe."""
        with self._db_lock:
            self._save_db_unsafe(db_path)

    @property
    def embedding_dim(self):
        """Embedding dimension của model"""
        return self._embedding_dim


# ============================================================================
# MODULE-LEVEL FUNCTIONS
# ============================================================================
_recognizer = None

def get_recognizer(db_path=None):
    """Get singleton recognizer instance"""
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer(db_path=db_path)
    return _recognizer


if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("Testing Face Recognizer")
    print("=" * 50)
    
    recognizer = FaceRecognizer(db_path="test_db.pkl")
    
    # Test với random image
    test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(3):
        recognizer.get_embedding(test_img)
    
    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        emb = recognizer.get_embedding(test_img)
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"\nEmbedding shape: {emb.shape}")
    print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
    print(f"Average time: {avg_time:.2f}ms")
    print(f"FPS: {1000/avg_time:.1f}")
    
    # Cleanup
    if os.path.exists("test_db.pkl"):
        os.remove("test_db.pkl")
