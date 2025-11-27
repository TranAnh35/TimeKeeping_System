# src/antispoof.py
from tflite_helper import get_interpreter
import cv2
import numpy as np

class AntiSpoof:
    def __init__(self, model_path="models/anti_spoof/2.7_80x80_MiniFASNetV2.tflite"):
        try:
            self.interpreter = get_interpreter(model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.input_shape = self.input_details[0]['shape']
            self.height = self.input_shape[1]
            self.width = self.input_shape[2]
            
        except Exception as e:
            print(f"[ERROR] Không tìm thấy model AntiSpoof tại: {model_path}")
            print("Hãy kiểm tra lại đường dẫn file .tflite")
            self.interpreter = None

    def is_live(self, face_img):
        if self.interpreter is None:
            return False

        img = cv2.resize(face_img, (self.width, self.height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        
        input_data = np.expand_dims(img, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        logits = output.flatten()
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        self.last_logits = logits
        self.last_probs = probs
        
        predicted_class = np.argmax(probs)
        
        real_prob = probs[2]
        
        return real_prob > 0.5