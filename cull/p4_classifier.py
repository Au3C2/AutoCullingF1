import logging
from pathlib import Path
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

ORIENT_MAP = {
    0: 'front',
    1: 'front_angle',
    2: 'side',
    3: 'rear_angle',
    4: 'rear'
}

_p4_classifier = None

class P4Classifier:
    def __init__(self, model_path: str = "models/p4_car_model.onnx"):
        self.model_path = Path(model_path)
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = []
            for p in ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']:
                if p in available: providers.append(p)
            if not providers: providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            
            dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
            self.session.run(None, {'input': dummy})
            log.info(f"P4Classifier loaded from {model_path}")
        except Exception as e:
            log.warning(f"Failed to load P4 model {model_path}: {e}")
            self.session = None

    def predict_roi(self, img_rgb: np.ndarray, bbox: tuple[float, float, float, float]) -> tuple[str, float, int, float]:
        if self.session is None: return "unknown", 0.0, 1, 1.0
            
        x1, y1, x2, y2 = bbox
        h, w = img_rgb.shape[:2]
        
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1: return "unknown", 0.0, 1, 1.0
            
        roi_arr = img_rgb[y1:y2, x1:x2]
        if roi_arr.size == 0: return "unknown", 0.0, 1, 1.0
            
        # Pillow resize
        roi_pil = Image.fromarray(roi_arr).resize((224, 224), Image.BILINEAR)
        roi = np.array(roi_pil).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        roi = (roi - mean) / std
        
        # HWC to CHW
        roi = np.transpose(roi, (2, 0, 1))
        roi = np.expand_dims(roi, axis=0)
        
        orient_logits, integ_logits = self.session.run(None, {'input': roi})
        
        # Integ
        integ_prob = 1.0 / (1.0 + np.exp(-integ_logits[0]))
        integ_pred = 1 if integ_prob > 0.5 else 0
        
        # Orient
        exp_o = np.exp(orient_logits[0] - np.max(orient_logits[0]))
        o_probs = exp_o / np.sum(exp_o)
        o_idx = np.argmax(o_probs)
        o_conf = o_probs[o_idx]
        o_str = ORIENT_MAP.get(o_idx, "unknown")
        
        return o_str, float(o_conf), int(integ_pred), float(integ_prob)

def get_p4_classifier() -> P4Classifier | None:
    global _p4_classifier
    if _p4_classifier is None:
        model_path = Path("models/p4_car_model.onnx")
        if model_path.exists():
            _p4_classifier = P4Classifier(str(model_path))
        else:
            log.warning("P4 model not found at models/p4_car_model.onnx")
    return _p4_classifier
