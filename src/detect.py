# src/detect.py
import cv2
import numpy as np
import os
import sys
from math import ceil

# Support both direct script execution and module import
try:
    from .tflite_helper import get_interpreter
except ImportError:
    from tflite_helper import get_interpreter

# --- CẤU HÌNH ---
MODEL_PATH = "models/detection/version-RFB-320_without_postprocessing.tflite"

class UltraLightFaceDetector:
    def __init__(self, model_path=MODEL_PATH, conf_threshold=0.6):
        self.conf_threshold = conf_threshold
        self._priors_cache = None  # Cache priors để tránh tính lại mỗi frame
        try:
            self.interpreter = get_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = (320, 240)
            # Pre-generate priors một lần duy nhất
            self._priors_cache = self._generate_priors(self.input_shape)
        except Exception as e:
            print(f"[LỖI] Detection model: {e}")
            self.interpreter = None

    def detect_faces(self, frame):
        if self.interpreter is None:
            return []

        h_img, w_img, _ = frame.shape
        
        # 1. Preprocess - dùng float32 để tiết kiệm RAM
        img_resized = cv2.resize(frame, self.input_shape) 
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = ((img_rgb.astype(np.float32) - 127.0) / 128.0)
        img_input = np.expand_dims(img_norm, axis=0)

        # 2. Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()

        # 3. Get Output
        out_a = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        out_b = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        
        # Giải phóng bộ nhớ tạm
        del img_resized, img_rgb, img_norm, img_input

        if out_a.shape[-1] == 2: 
            scores, boxes_enc = out_a, out_b
        else:
            scores, boxes_enc = out_b, out_a
        
        scores = scores[:, 1]
        
        # 4. Filter
        mask = scores > self.conf_threshold
        scores = scores[mask]
        boxes_enc = boxes_enc[mask]
        
        if len(scores) == 0:
            return []

        # 5. Sử dụng cached Priors (đã pre-generate)
        priors = self._priors_cache[mask]

        # Decode Boxes
        variances = [0.1, 0.2]
        boxes = np.concatenate([
            priors[:, :2] + boxes_enc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(boxes_enc[:, 2:] * variances[1])
        ], axis=1)

        # (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        # Scale về ảnh gốc
        boxes[:, 0] *= w_img
        boxes[:, 2] *= w_img
        boxes[:, 1] *= h_img
        boxes[:, 3] *= h_img

        # 6. NMS
        rects = boxes.astype(int)
        keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), self.conf_threshold, 0.3)

        final_results = []
        if len(keep) > 0:
            for i in keep.flatten():
                x_min, y_min, x_max, y_max = rects[i]
                w = x_max - x_min
                h = y_max - y_min
                
                # Clip coordinates
                x = max(0, x_min)
                y = max(0, y_min)
                w = min(w, w_img - x)
                h = min(h, h_img - y)
                
                final_results.append({
                    'box': [x, y, w, h],
                    'confidence': float(scores[i]),
                    'keypoints': {}
                })
        
        return final_results

    def _generate_priors(self, input_shape):
        width, height = input_shape
        
        # Cấu hình chuẩn cho RFB-320
        feature_map_sizes = [
            [ceil(height / 8), ceil(width / 8)],   # 30, 40
            [ceil(height / 16), ceil(width / 16)], # 15, 20
            [ceil(height / 32), ceil(width / 32)], # 8, 10
            [ceil(height / 64), ceil(width / 64)]  # 4, 5
        ]
        
        min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 176, 256]]
        priors = []
        
        for k, size in enumerate(feature_map_sizes):
            fh, fw = size
            for y in range(fh):
                for x in range(fw):
                    cx, cy = (x + 0.5) / fw, (y + 0.5) / fh
                    for min_size in min_sizes[k]:
                        s_kx = min_size / width
                        s_ky = min_size / height
                        priors.append([cx, cy, s_kx, s_ky])
                        
        return np.array(priors, dtype=np.float32)

# --- GLOBAL INSTANCE ---
_detector = UltraLightFaceDetector()

def detect_faces(frame_bgr):
    return _detector.detect_faces(frame_bgr)