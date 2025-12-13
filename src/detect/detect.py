# src/detect/detect.py
"""
Face Detection module - INT8 quantized model.
Model: version-RFB-320_int8_without_postprocessing.tflite

INT8 model có đặc điểm:
- Input: int8 [1, 240, 320, 3] với scale=0.0078125, zero_point=-1
- Output 0: boxes [1, 4420, 4] int8, scale=0.05163755, zp=12
- Output 1: scores [1, 4420, 2] int8, scale=0.00390625, zp=-128

Thread-safe: Sử dụng Lock cho TFLite inference.
"""
import cv2
import numpy as np
import os
import sys
import threading
from math import ceil

# Support both direct script execution and module import
try:
    from ..core.tflite_helper import get_interpreter
    from ..core.settings import settings
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.tflite_helper import get_interpreter
    from core.settings import settings

# --- CẤU HÌNH INT8 ---
MODEL_PATH_INT8 = "models/detection/version-RFB-320_int8_without_postprocessing.tflite"

# INT8 Quantization parameters (từ model)
INPUT_SCALE = 0.0078125
INPUT_ZERO_POINT = -1


class UltraLightFaceDetector:
    """
    Face Detector sử dụng INT8 quantized model. Thread-safe.
    Tối ưu cho Raspberry Pi với tflite-runtime.
    
    Model: version-RFB-320_int8_without_postprocessing.tflite
    - Input: [1, 240, 320, 3] int8
    - Output 0 (boxes): [1, 4420, 4] int8
    - Output 1 (scores): [1, 4420, 2] int8
    """
    
    def __init__(self, model_path=MODEL_PATH_INT8, conf_threshold=0.6):
        # Thread-safety lock
        self._inference_lock = threading.Lock()
        
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        self._priors_cache = None
        
        # Hardcoded INT8 parameters
        self._input_scale = INPUT_SCALE
        self._input_zero_point = INPUT_ZERO_POINT
        
        try:
            self.interpreter = get_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Input shape: [1, 240, 320, 3]
            self.input_shape = (320, 240)  # (width, height)
            
            # Lấy quantization parameters từ model
            inp_quant = self.input_details[0].get('quantization_parameters', {})
            if inp_quant.get('scales') is not None and len(inp_quant['scales']) > 0:
                self._input_scale = float(inp_quant['scales'][0])
            if inp_quant.get('zero_points') is not None and len(inp_quant['zero_points']) > 0:
                self._input_zero_point = int(inp_quant['zero_points'][0])
            
            # Cache input/output indices for faster lookup
            self._input_index = self.input_details[0]['index']
            self._output_indices = [d['index'] for d in self.output_details]
            
            # Lấy output quantization parameters
            self._output_params = []
            for out_detail in self.output_details:
                quant_params = out_detail.get('quantization_parameters', {})
                scale = quant_params.get('scales', [1.0])
                zp = quant_params.get('zero_points', [0])
                self._output_params.append({
                    'scale': float(scale[0]) if len(scale) > 0 else 1.0,
                    'zero_point': int(zp[0]) if len(zp) > 0 else 0
                })
            
            # Pre-generate priors
            self._priors_cache = self._generate_priors(self.input_shape)
            
            print(f"[INT8 Detector] Loaded: {model_path}")
            print(f"[INT8 Detector] Input: scale={self._input_scale}, zp={self._input_zero_point}")
            print(f"[INT8 Detector] Priors: {len(self._priors_cache)}")
            
        except Exception as e:
            print(f"[LỖI] INT8 Detection model: {e}")
            self.interpreter = None

    def _preprocess_int8(self, frame):
        """
        Preprocess ảnh cho INT8 model.
        
        Pipeline:
        1. Resize về 320x240
        2. Convert BGR -> RGB
        3. Normalize về [-1, 1] (float)
        4. Quantize: int8_value = float_value / scale + zero_point
        """
        # Resize
        img_resized = cv2.resize(frame, self.input_shape)
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize về [-1, 1] rồi quantize về int8
        # normalized_value = (pixel - 127.5) / 127.5 (range [-1, 1])
        # quantized = normalized_value / scale + zero_point
        img_float = (img_rgb.astype(np.float32) - 127.5) / 127.5
        
        # Quantize to int8
        img_int8 = np.clip(
            np.round(img_float / self._input_scale + self._input_zero_point),
            -128, 127
        ).astype(np.int8)
        
        return np.expand_dims(img_int8, axis=0)

    def _dequantize_output(self, output, param_idx):
        """
        Dequantize INT8 output về float32.
        Formula: float_value = (int8_value - zero_point) * scale
        """
        if output.dtype == np.int8 or output.dtype == np.uint8:
            params = self._output_params[param_idx]
            return (output.astype(np.float32) - params['zero_point']) * params['scale']
        return output.astype(np.float32)

    def detect_faces(self, frame):
        """
        Detect faces trong frame. Thread-safe.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of dict với keys: 'box' [x, y, w, h], 'confidence', 'keypoints'
        """
        if self.interpreter is None:
            return []

        h_img, w_img = frame.shape[:2]
        
        # 1. Preprocess INT8 (outside lock)
        img_input = self._preprocess_int8(frame)

        # 2. Inference (thread-safe)
        with self._inference_lock:
            self.interpreter.set_tensor(self._input_index, img_input)
            self.interpreter.invoke()

            # 3. Get Output (copy to release interpreter)
            out_0 = np.array(self.interpreter.get_tensor(self._output_indices[0])[0], copy=True)
            out_1 = np.array(self.interpreter.get_tensor(self._output_indices[1])[0], copy=True)
        
        # 4. Post-process (outside lock)
        out_0 = self._dequantize_output(out_0, 0)
        out_1 = self._dequantize_output(out_1, 1)

        # Xác định output nào là scores, output nào là boxes
        # boxes có shape [..., 4], scores có shape [..., 2]
        if out_0.shape[-1] == 4:
            boxes_enc = out_0
            scores = out_1
        else:
            boxes_enc = out_1
            scores = out_0
        
        # Lấy score của class "face" (index 1)
        scores = scores[:, 1]
        
        # 4. Filter theo confidence threshold
        mask = scores > self.conf_threshold
        scores_filtered = scores[mask]
        boxes_filtered = boxes_enc[mask]
        
        if len(scores_filtered) == 0:
            return []

        # 5. Decode boxes sử dụng cached priors
        priors = self._priors_cache[mask]
        
        variances = [0.1, 0.2]
        boxes = np.concatenate([
            priors[:, :2] + boxes_filtered[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(boxes_filtered[:, 2:] * variances[1])
        ], axis=1)

        # (cx, cy, w, h) -> (x_min, y_min, x_max, y_max)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        # Scale về kích thước ảnh gốc
        boxes[:, 0] *= w_img
        boxes[:, 2] *= w_img
        boxes[:, 1] *= h_img
        boxes[:, 3] *= h_img

        # 6. NMS
        rects = boxes.astype(int)
        keep = cv2.dnn.NMSBoxes(rects.tolist(), scores_filtered.tolist(), 
                                 self.conf_threshold, 0.3)

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
                    'confidence': float(scores_filtered[i]),
                    'keypoints': {}
                })
        
        return final_results

    def _generate_priors(self, input_shape):
        """
        Generate anchor boxes (priors) cho SSD-style detection.
        Uses pre-allocated array for efficiency.
        """
        width, height = input_shape
        
        feature_map_sizes = [
            (ceil(height / 8), ceil(width / 8)),   # 30, 40
            (ceil(height / 16), ceil(width / 16)), # 15, 20
            (ceil(height / 32), ceil(width / 32)), # 8, 10
            (ceil(height / 64), ceil(width / 64))  # 4, 5
        ]
        
        min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 176, 256]]
        
        # Pre-calculate total priors for allocation
        total = sum(fh * fw * len(ms) for (fh, fw), ms in zip(feature_map_sizes, min_sizes))
        priors = np.empty((total, 4), dtype=np.float32)
        
        idx = 0
        for k, (fh, fw) in enumerate(feature_map_sizes):
            for y in range(fh):
                cy = (y + 0.5) / fh
                for x in range(fw):
                    cx = (x + 0.5) / fw
                    for min_size in min_sizes[k]:
                        priors[idx] = [cx, cy, min_size / width, min_size / height]
                        idx += 1
                        
        return priors


# --- GLOBAL INSTANCE ---
_detector = None

def get_detector():
    """Lazy initialization để tránh load model khi import"""
    global _detector
    if _detector is None:
        _detector = UltraLightFaceDetector()
    return _detector

def detect_faces(frame_bgr):
    """Wrapper function để detect faces."""
    return get_detector().detect_faces(frame_bgr)


if __name__ == "__main__":
    # Test nhanh
    import time
    
    print("Testing Face Detector...")
    detector = UltraLightFaceDetector()
    
    # Tạo test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(3):
        detector.detect_faces(test_img)
    
    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        results = detector.detect_faces(test_img)
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"Average inference time: {avg_time:.2f}ms")
    print(f"FPS: {1000/avg_time:.1f}")
