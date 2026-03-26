"""
fence_classifier.py — Wire fence detection inference (LITE VERSION).
Uses onnxruntime and remove torch/cv2 deps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

class FenceClassifier:
    """Wire fence binary classifier using ONNX."""
    
    def __init__(self, arch: str = "mobilenetv2", checkpoint_dir: str = "models"):
        self.arch = arch
        self.model_path = Path(checkpoint_dir) / f"fence_{arch}.onnx"
        self.session = None
        
        try:
            import onnxruntime as ort
            if self.model_path.exists():
                self.session = ort.InferenceSession(str(self.model_path))
                log.info(f"FenceClassifier loaded from {self.model_path}")
            else:
                log.warning(f"Fence ONNX model not found at {self.model_path}. Fence veto disabled.")
        except Exception as e:
            log.warning(f"Failed to initialize FenceClassifier ONNX: {e}")
            self.session = None
            
        # Standard ImageNet Normalize
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def predict_image(self, img_path: Union[str, Path]) -> tuple[int, float]:
        img_path = Path(img_path)
        try:
            pil_img = Image.open(img_path).convert("RGB")
            return self.predict_roi(np.array(pil_img), (0, 0, pil_img.width, pil_img.height))
        except Exception:
            return 0, 0.0
        
    def predict_roi(self, img_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, float]:
        if self.session is None: return 0, 0.0
            
        x1, y1, x2, y2 = bbox
        h, w = img_rgb.shape[:2]
        
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        
        if x2 <= x1 or y2 <= y1: return 0, 0.0
            
        roi_arr = img_rgb[y1:y2, x1:x2]
        if roi_arr.size == 0: return 0, 0.0
            
        # Pillow Resize
        roi_pil = Image.fromarray(roi_arr).resize((224, 224), Image.BILINEAR)
        roi = np.array(roi_pil).astype(np.float32) / 255.0
        roi = (roi - self.mean) / self.std
        roi = np.transpose(roi, (2, 0, 1))
        roi = np.expand_dims(roi, axis=0)
        
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: roi})
        logit = outputs[0][0][0]
        prob = 1.0 / (1.0 + np.exp(-logit))
        pred = 1 if prob > 0.5 else 0
        
        return pred, float(prob)

    def predict_batch(self, img_paths: list[Union[str, Path]], batch_size: int = 32) -> np.ndarray:
        results = [self.predict_image(p)[0] for p in img_paths]
        return np.array(results, dtype=np.int32)
