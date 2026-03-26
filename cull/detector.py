"""
detector.py — Cascade object detection for F1 photo culling (LITE VERSION).
Refined for high-fidelity alignment with OpenCV results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import ast

import numpy as np
from PIL import Image
import sys

log = logging.getLogger(__name__)

def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.parent.resolve()
    return base_path / relative_path

_COCO_INTEREST: dict[int, tuple[str, float]] = {
    2:  ("coco_car",        0.7),
    5:  ("coco_airplane",   0.3),
    0:  ("coco_person",     0.5),
}

_CONF_THRESHOLD = 0.25
_F1_CLASS_WEIGHT = 1.0

@dataclass
class Detection:
    label: str
    weight: float
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2.0
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2.0
    def area(self) -> float: return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
    def area_ratio(self, img_w: int, img_h: int) -> float: return self.area() / max(1, img_w * img_h)
    def center_proximity(self, img_w: int, img_h: int) -> float:
        dx = abs(self.cx - img_w / 2.0) / (img_w / 2.0)
        dy = abs(self.cy - img_h / 2.0) / (img_h / 2.0)
        dist = (dx**2 + dy**2) ** 0.5 / (2.0 ** 0.5)
        return max(0.0, 1.0 - dist)
    def subject_score(self, img_w: int, img_h: int) -> float:
        return 0.50 * self.weight + 0.30 * self.area_ratio(img_w, img_h) + 0.20 * self.center_proximity(img_w, img_h)

def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0: return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

class LiteYOLO:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            providers = [p for p in providers if p in available] or ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            meta = self.session.get_modelmeta().custom_metadata_map
            self.imgsz = (640, 640)
            if 'imgsz' in meta:
                try: self.imgsz = ast.literal_eval(meta['imgsz'])
                except: pass
            self.names = {}
            if 'names' in meta:
                try: self.names = {str(k): v for k, v in ast.literal_eval(meta['names']).items()}
                except: pass
            log.info(f"YOLO LITE (Pillow-Precision) loaded: {model_path} ({self.imgsz})")
        except Exception as e:
            log.error(f"Failed to load engine: {e}")
            self.session = None

    def letterbox_pil(self, pil_img: Image.Image, new_shape=(640, 640), color=(114, 114, 114)) -> tuple[np.ndarray, float, tuple[float, float]]:
        w, h = pil_img.size
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        # For large downsampling, Image.BOX (area average) is much closer to OpenCV's INTER_AREA
        # than BICUBIC/BILINEAR. This fixes detection confidence drift.
        resample_mod = Image.BOX if h > new_shape[0] * 2 else Image.BILINEAR
        img_resized = pil_img.resize(new_unpad, resample_mod)
        canvas = Image.new("RGB", (new_shape[1], new_shape[0]), color)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2.0, (new_shape[0] - new_unpad[1]) / 2.0
        # Precision rounding to match Ultralytics C++ implementation
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas.paste(img_resized, (left, top))
        return np.array(canvas), r, (float(left), float(top))

    def detect(self, img_pil: Image.Image, conf_thresh: float = _CONF_THRESHOLD, nms_thresh: float = 0.45) -> list[dict]:
        if self.session is None: return []
        img_canvas, ratio, (dw, dh) = self.letterbox_pil(img_pil, new_shape=self.imgsz)
        input_tensor = img_canvas.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        output = outputs[0][0].transpose()
        boxes, scores_list, class_ids = [], [], []
        for row in output:
            scores = row[4:]
            cid = np.argmax(scores)
            conf = scores[cid]
            if conf > conf_thresh:
                xc, yc, w, h = row[:4]
                x1 = (xc - w/2.0 - dw) / ratio
                y1 = (yc - h/2.0 - dh) / ratio
                bw, bh = w/ratio, h/ratio
                boxes.append([x1, y1, x1+bw, y1+bh])
                scores_list.append(float(conf))
                class_ids.append(int(cid))
        if not boxes: return []
        indices = nms_numpy(np.array(boxes), np.array(scores_list), nms_thresh)
        return [{"cls_id": class_ids[i], "cls_name": self.names.get(str(class_ids[i]), str(class_ids[i])),
                 "conf": scores_list[i], "x1": boxes[i][0], "y1": boxes[i][1], "x2": boxes[i][2], "y2": boxes[i][3]} for i in indices]

def load_f1_model(onnx_path: Path): return LiteYOLO(onnx_path) if onnx_path.exists() else None
def load_coco_model():
    p = Path("models/yolov8n.onnx")
    if not p.exists():
        bundled = get_resource_path("models/yolov8n.onnx")
        if bundled.exists(): p = bundled
    return LiteYOLO(p) if p.exists() else None

def detect(img_rgb: np.ndarray, f1: LiteYOLO | None, coco: LiteYOLO | None, conf: float = _CONF_THRESHOLD) -> list[Detection]:
    detections: list[Detection] = []
    pil_img = Image.fromarray(img_rgb)
    if f1:
        for b in f1.detect(pil_img, conf_thresh=conf):
            detections.append(Detection(label="f1_car", weight=_F1_CLASS_WEIGHT, conf=b["conf"], x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
        if detections:
            h, w = img_rgb.shape[:2]
            detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
            return detections
    if coco:
        for b in coco.detect(pil_img, conf_thresh=conf):
            if b["cls_id"] in _COCO_INTEREST:
                l, w = _COCO_INTEREST[b["cls_id"]]
                detections.append(Detection(label=l, weight=w, conf=b["conf"], x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
    h, w = img_rgb.shape[:2]
    detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
    return detections

class CloudF1Detector:
    def __init__(self, key): pass
    def detect(self, img, conf): return []
