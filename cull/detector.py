"""
detector.py — Cascade object detection for F1 photo culling (LITE VERSION).
Refined for high-fidelity alignment with OpenCV results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import ast

import cv2
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

_NVIDIA_BIN_DIRS = ("cudnn", "cublas", "cuda_nvrtc")


def ensure_nvidia_runtime_on_path() -> None:
    """Prepend NVIDIA pip-wheel runtime DLL dirs (cuDNN/cuBLAS/nvrtc) to PATH.

    onnxruntime-gpu loads its CUDA provider through standard DLL search.
    The nvidia-* wheels install their DLLs under ``site-packages/nvidia/*/bin``,
    which is not on PATH by default — without this, CUDAExecutionProvider fails
    to activate and silently degrades to CPU. No-op when the dirs are absent.
    """
    if sys.platform != "win32":
        return
    import os
    import site

    extra = []
    for base in site.getsitepackages():
        for name in _NVIDIA_BIN_DIRS:
            dll_dir = Path(base) / "nvidia" / name / "bin"
            if dll_dir.is_dir():
                extra.append(str(dll_dir))
    if not extra:
        return
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join(extra + ([existing] if existing else []))


def _has_concrete_input_shape(model_path: Path, sess_opts) -> bool:
    """True when the graph's first input has fully concrete dimensions.

    Ultralytics exports keep batch/height/width symbolic ('batch'/'height'/
    'width'); onnxsim constant-folded exports freeze them to ints. The
    packaged single-file binary ships the frozen graph under the plain base
    name (no ``_static`` sibling), so file-name detection is not enough —
    probe the graph itself. Returns False when the session cannot load.
    """
    try:
        import onnxruntime as _ort
        probe = _ort.InferenceSession(str(model_path), sess_options=sess_opts,
                                      providers=["CPUExecutionProvider"])
        try:
            shape = probe.get_inputs()[0].shape
        finally:
            del probe
        return all(isinstance(d, int) and d > 0 for d in shape)
    except Exception:
        return False


class LiteYOLO:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        try:
            import onnxruntime as ort
            from cull.deterministic import is_deterministic
            deterministic = is_deterministic()
            ensure_nvidia_runtime_on_path()
            available = ort.get_available_providers()
            opts = ort.SessionOptions()
            opts.log_severity_level = 3  # Suppress non-fatal fallback warnings (e.g. CUDA -> CPU)
            if deterministic:
                # Cross-platform bit-identical path: CPU only, single-threaded,
                # deterministic kernels when the ORT build exposes the flag.
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                try:
                    opts.use_deterministic_compute = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                providers = ["CPUExecutionProvider"]
            else:
                providers = None  # filled per-branch below
            model_file = model_path
            if deterministic:
                pass  # CPU path already pinned above; no platform-specific tuning.
            elif sys.platform == "darwin" and 'CoreMLExecutionProvider' in available:
                # Prefer the batch=1 static graph (onnxsim constant-folded):
                # RequireStaticInputShapes qualifies 227/231 nodes in 3
                # partitions vs 7/233-of-318 for the dynamic export — full
                # scoring chain 40.2 -> 26.4 ms/frame on Apple M4, interleaved
                # A/B (2026-08-25). The engine always calls session.run with
                # batch=1. Side effect: ~3% of gate files drift raw_score via
                # the P4 ROI knife-edge (ratings unchanged; see
                # performance_baseline.md). Windows keeps the dynamic model.
                static_path = model_path.with_name(model_path.stem + "_static.onnx")
                if static_path.exists():
                    model_file = static_path
                    static = True
                else:
                    # Packaged binaries ship the frozen graph under the base
                    # name; identical pinned options apply once concrete
                    # input dims are detected.
                    model_file = model_path
                    static = _has_concrete_input_shape(model_file, opts)
                if static:
                    # Pin the compute units. With the default ALL the runtime
                    # silently switches between ANE and GPU depending on
                    # system load; the two paths differ by ~0.01 detection
                    # confidence, which flips rating-boundary files across
                    # runs (observed on DSC00849.heif). ANE is also the
                    # fastest unit for this graph (idle-state full chain:
                    # locked-ANE 18.2 vs ALL 17.8 vs locked-GPU 28.0 ms).
                    providers = [
                        ("CoreMLExecutionProvider", {
                            "RequireStaticInputShapes": "1",
                            "MLComputeUnits": "CPUAndNeuralEngine",
                        }),
                        "CPUExecutionProvider",
                    ]
                else:
                    providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                    providers = [p for p in providers if p in available] or ['CPUExecutionProvider']
            elif not deterministic:
                providers = ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                providers = [p for p in providers if p in available] or ['CPUExecutionProvider']
            self.session = ort.InferenceSession(str(model_file), sess_options=opts, providers=providers)
            log.info(f"YOLO LITE active providers: {self.session.get_providers()}")
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

    def letterbox_numpy(self, img: np.ndarray, new_shape=(640, 640), color=114) -> tuple[np.ndarray, float, tuple[float, float]]:
        """cv2-based letterbox (GIL-released C code) with identical geometry
        math to ``letterbox_pil``; kernels LINEAR/AREA vs PIL BILINEAR/BOX.
        Gate-verified with the v2 P4 model (see performance_baseline.md)."""
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = (int(round(w * r)), int(round(h * r)))
        interp = cv2.INTER_AREA if h > new_shape[0] * 2 else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, new_unpad, interpolation=interp)
        dw, dh = (new_shape[1] - new_unpad[0]) / 2.0, (new_shape[0] - new_unpad[1]) / 2.0
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=np.uint8)
        canvas[top:top + new_unpad[1], left:left + new_unpad[0]] = img_resized
        return canvas, r, (float(left), float(top))

    def _run_session(self, img_canvas: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        input_tensor = img_canvas.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs

    def detect(self, img_pil: Image.Image, conf_thresh: float = _CONF_THRESHOLD, nms_thresh: float = 0.45) -> list[dict]:
        if self.session is None: return []
        img_canvas, ratio, (dw, dh) = self.letterbox_pil(img_pil, new_shape=self.imgsz)
        outputs = self._run_session(img_canvas)
        return self._postprocess(outputs, ratio, dw, dh, conf_thresh, nms_thresh)

    def prepare_numpy(self, img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Letterbox + build this model's NCHW float input tensor in one step.

        The divide runs in place on the astype copy (elementwise-identical to
        ``astype(f32) / 255.0`` — same operands, same IEEE division), saving a
        full 9.4 MB intermediate allocation per frame."""
        img_canvas, ratio, (dw, dh) = self.letterbox_numpy(img, new_shape=self.imgsz)
        x = img_canvas.astype(np.float32)
        x /= 255.0
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1)))[np.newaxis, ...]
        return x, ratio, dw, dh

    def detect_numpy(self, img: np.ndarray, conf_thresh: float = _CONF_THRESHOLD, nms_thresh: float = 0.45,
                     prep: tuple[np.ndarray, float, float, float] | None = None) -> list[dict]:
        """Same pipeline but letterboxing runs in cv2 directly on the numpy
        frame (GIL-free) — removes PIL resize from the consumer thread.

        ``prep`` accepts a precomputed ``prepare_numpy`` result so cascaded
        detectors (F1 miss -> COCO fallback) share one letterbox AND one
        input tensor; both depend only on the frame and ``imgsz``, which is
        equal across the cascade."""
        if self.session is None: return []
        tensor, ratio, dw, dh = prep if prep is not None else self.prepare_numpy(img)
        outputs = self.session.run(None, {self.input_name: tensor})
        return self._postprocess(outputs, ratio, dw, dh, conf_thresh, nms_thresh)

    def _postprocess(self, outputs, ratio: float, dw: float, dh: float,
                     conf_thresh: float, nms_thresh: float) -> list[dict]:
        # Vectorized decode of the 1x(C+4)x8400 output: mask-then-gather instead
        # of a per-row python loop (loop measured 12.4 ms/img, vector ~1 ms).
        # np.argmax / arithmetic are elementwise-identical to the old loop, so
        # the resulting boxes/scores/class_ids are bit-identical (gates verify).
        out0 = outputs[0][0]  # (C+4, 8400)
        cls_scores = out0[4:, :]
        class_ids_all = np.argmax(cls_scores, axis=0)
        conf_all = cls_scores[class_ids_all, np.arange(cls_scores.shape[1])]
        keep = conf_all > conf_thresh
        if not np.any(keep):
            return []
        rows = out0[:4, keep]
        class_ids = class_ids_all[keep].astype(int)
        scores_list = conf_all[keep].astype(float)
        xc, yc, w, h = rows
        x1 = (xc - w / 2.0 - dw) / ratio
        y1 = (yc - h / 2.0 - dh) / ratio
        bw, bh = w / ratio, h / ratio
        boxes = np.stack([x1, y1, x1 + bw, y1 + bh], axis=1)
        indices = nms_numpy(boxes, scores_list, nms_thresh)
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
    prep = None
    if f1:
        # Letterbox + tensor build once up front; when F1 misses, the COCO
        # fallback reuses the same input tensor (identical geometry for equal
        # imgsz) instead of resizing and re-normalizing the full frame.
        if f1.session is not None:
            prep = f1.prepare_numpy(img_rgb)
        for b in f1.detect_numpy(img_rgb, conf_thresh=conf, prep=prep):
            detections.append(Detection(label="f1_car", weight=_F1_CLASS_WEIGHT, conf=b["conf"], x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
        if detections:
            h, w = img_rgb.shape[:2]
            detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
            return detections
    if coco:
        coco_prep = prep if (prep is not None and f1 is not None
                             and tuple(f1.imgsz) == tuple(coco.imgsz)) else None
        for b in coco.detect_numpy(img_rgb, conf_thresh=conf, prep=coco_prep):
            if b["cls_id"] in _COCO_INTEREST:
                l, w = _COCO_INTEREST[b["cls_id"]]
                detections.append(Detection(label=l, weight=w, conf=b["conf"], x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
    h, w = img_rgb.shape[:2]
    detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
    return detections

class CloudF1Detector:
    def __init__(self, key): pass
    def detect(self, img, conf): return []
