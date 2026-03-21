"""
detector.py — Cascade object detection for F1 photo culling.

Detection pipeline
------------------
Stage 1 — F1-specialised YOLO model (local ONNX).
  If any detection passes the confidence threshold → use those results,
  skip Stage 2.

Stage 2 — COCO YOLOv8n fallback.
  Detects generic car / person / airplane classes when no F1 car is found.

If both stages return no detections → empty list (image will be rejected).

Class → semantic weight mapping
--------------------------------
f1_car (all 10 F1 teams → unified)                    : 1.0
coco_car                                              : 0.7
coco_person                                           : 0.5
coco_airplane / coco_helicopter                       : 0.3

Subject score formula (used in scorer.py)
-----------------------------------------
  S_subject = 0.50 * W_class
            + 0.30 * area_ratio        (bbox area / image area)
            + 0.20 * center_proximity  (1 - normalised distance from centre)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO class IDs we care about (YOLOv8 / COCO-80 index)
# ---------------------------------------------------------------------------
_COCO_INTEREST: dict[int, tuple[str, float]] = {
    2:  ("coco_car",        0.7),
    5:  ("coco_airplane",   0.3),
    0:  ("coco_person",     0.5),
    # helicopter is not a COCO-80 class; aircraft proxy via airplane
}

# Confidence threshold applied to all detections
_CONF_THRESHOLD = 0.25

# F1 class label → semantic weight
_F1_CLASS_WEIGHT = 1.0   # all F1 car team labels map to this

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    """A single object detection result."""

    label: str          # semantic label, e.g. "f1_car", "coco_car"
    weight: float       # semantic importance weight [0, 1]
    conf: float         # model confidence [0, 1]
    x1: float           # bbox top-left x  (pixel coords)
    y1: float           # bbox top-left y
    x2: float           # bbox bottom-right x
    y2: float           # bbox bottom-right y

    # ---- derived properties ------------------------------------------------

    @property
    def cx(self) -> float:
        """Bbox centre x."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Bbox centre y."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def area_ratio(self, img_w: int, img_h: int) -> float:
        """Fraction of image area covered by this bbox."""
        return self.area() / max(1, img_w * img_h)

    def center_proximity(self, img_w: int, img_h: int) -> float:
        """Normalised distance from image centre (1 = at centre, 0 = corner)."""
        dx = abs(self.cx - img_w / 2.0) / (img_w / 2.0)
        dy = abs(self.cy - img_h / 2.0) / (img_h / 2.0)
        dist = (dx**2 + dy**2) ** 0.5 / (2.0 ** 0.5)  # normalise to [0, 1]
        return max(0.0, 1.0 - dist)

    def subject_score(self, img_w: int, img_h: int) -> float:
        """Composite subject importance score ∈ [0, 1]."""
        return (
            0.50 * self.weight
            + 0.30 * self.area_ratio(img_w, img_h)
            + 0.20 * self.center_proximity(img_w, img_h)
        )


# ---------------------------------------------------------------------------
# Model loader helpers
# ---------------------------------------------------------------------------


def load_f1_model(onnx_path: Path):
    """Load the F1-specialised YOLO model.
    Respects CULL_BACKEND env var.
    """
    import os
    import platform
    backend = os.environ.get("CULL_BACKEND", "auto").lower()
    is_mac = platform.system() == "Darwin"
    
    # Force ONNX if requested
    if backend == "onnx":
        log.info("Forcing ONNX backend via CULL_BACKEND")
    elif is_mac or backend == "coreml":
        # Check if there is a coreml version in models/
        coreml_path = Path("models/f1_yolov8n.mlpackage")
        if coreml_path.exists():
            try:
                from ultralytics import YOLO
                model = YOLO(str(coreml_path), task="detect")
                log.info("F1 CoreML model loaded from %s", coreml_path)
                return model
            except Exception as e:
                log.warning(f"Failed to load F1 CoreML model: {e}")

    if not onnx_path.exists():
        log.warning(
            "F1 ONNX model not found at %s — F1 detection disabled.",
            onnx_path,
        )
        return None

    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(str(onnx_path), task="detect")
        log.info("F1 YOLO model loaded from %s", onnx_path)
        return model
    except Exception as exc:
        log.warning("Failed to load F1 ONNX model: %s", exc)
        return None


def load_coco_model():
    """Load the YOLOv8n COCO model.
    Respects CULL_BACKEND env var.
    """
    try:
        from ultralytics import YOLO  # type: ignore
        import os
        import platform
        backend = os.environ.get("CULL_BACKEND", "auto").lower()
        
        if backend != "onnx" and platform.system() == "Darwin":
            coreml_path = Path("models/yolov8n.mlpackage")
            if coreml_path.exists():
                model = YOLO(str(coreml_path), task="detect")
                log.info("COCO CoreML model loaded from %s", coreml_path)
                return model
        
        # Try to find local weights first to prevent unwanted downloads in root
        for path in [Path("models/yolov8n.onnx"), Path("models/yolov8n.pt")]:
            if path.exists():
                model = YOLO(str(path))
                log.info("COCO fallback model loaded from %s", path)
                return model
        
        # If not found anywhere, use the string and let it manage it (will download to models/ if we can specify it, 
        # but let's just use "yolov8n.pt" if absolutely necessary)
        # Actually, let's keep it to yolov8n.pt as top-level fallback
        model = YOLO("yolov8n.pt")
        return model
    except Exception as exc:
        raise RuntimeError(f"Failed to load COCO YOLOv8n model: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal: run one YOLO model on a numpy image
# ---------------------------------------------------------------------------


def _run_yolo(
    model,
    img_rgb: np.ndarray,
    conf: float = _CONF_THRESHOLD,
) -> list[dict]:
    """Run a YOLO model and return raw box dicts.

    Returns a list of dicts with keys: cls_id, cls_name, conf, x1, y1, x2, y2.
    """
    results = model(img_rgb, conf=conf, verbose=False)
    boxes_out: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes_out.append({
                "cls_id":   cls_id,
                "cls_name": cls_name,
                "conf":     float(box.conf[0].item()),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return boxes_out


# ---------------------------------------------------------------------------
# Public API: detect
# ---------------------------------------------------------------------------


def detect(
    img_rgb: np.ndarray,
    f1_model,
    coco_model,
    conf: float = _CONF_THRESHOLD,
) -> list[Detection]:
    """Run the cascade detector on a single image.

    Parameters
    ----------
    img_rgb:
        H×W×3 uint8 numpy array (RGB).
    f1_model:
        Loaded F1 YOLO model (from :func:`load_f1_model`), or ``None``.
    coco_model:
        Loaded COCO YOLOv8n model (from :func:`load_coco_model`).
    conf:
        Minimum confidence threshold.

    Returns
    -------
    list[Detection]
        All accepted detections, sorted by subject_score descending.
        Empty list if nothing is detected above threshold.
    """
    detections: list[Detection] = []

    # --- Stage 1: F1-specialised model ---
    if f1_model is not None:
        f1_boxes = _run_yolo(f1_model, img_rgb, conf)
        for b in f1_boxes:
            detections.append(Detection(
                label="f1_car",
                weight=_F1_CLASS_WEIGHT,
                conf=b["conf"],
                x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
            ))
        if detections:
            log.debug("Stage-1 (F1 model): %d detections", len(detections))
            h, w = img_rgb.shape[:2]
            detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
            return detections

    # --- Stage 2: COCO fallback ---
    coco_boxes = _run_yolo(coco_model, img_rgb, conf)
    for b in coco_boxes:
        if b["cls_id"] in _COCO_INTEREST:
            label, weight = _COCO_INTEREST[b["cls_id"]]
            detections.append(Detection(
                label=label,
                weight=weight,
                conf=b["conf"],
                x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
            ))

    if detections:
        log.debug("Stage-2 (COCO model): %d detections", len(detections))
    else:
        log.debug("No detections in either stage")

    h, w = img_rgb.shape[:2]
    detections.sort(key=lambda d: d.subject_score(w, h), reverse=True)
    return detections

# ---------------------------------------------------------------------------
# Cloud-API F1 detection fallback (inference_sdk)
# ---------------------------------------------------------------------------


class CloudF1Detector:
    """Thin wrapper around inference_sdk for F1 detection via Roboflow cloud."""

    MODEL_ID = "formula-one-car-detection/1"
    API_URL  = "https://serverless.roboflow.com"

    def __init__(self, api_key: str) -> None:
        try:
            from inference_sdk import InferenceHTTPClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "inference-sdk is required for cloud F1 detection.\n"
                "  Install: pip install inference-sdk"
            ) from exc
        self._client = InferenceHTTPClient(
            api_url=self.API_URL,
            api_key=api_key,
        )
        log.info("Cloud F1 detector initialised (model: %s)", self.MODEL_ID)

    def detect(self, img_rgb: np.ndarray, conf: float = 0.25) -> list[Detection]:
        """Run cloud inference and return Detection list."""
        from PIL import Image  # type: ignore
        pil_img = Image.fromarray(img_rgb)

        try:
            result = self._client.infer(pil_img, model_id=self.MODEL_ID)
        except Exception as exc:
            log.warning("Cloud F1 inference failed: %s", exc)
            return []

        detections: list[Detection] = []
        for pred in result.get("predictions", []):
            if pred.get("confidence", 0) < conf:
                continue
            x   = pred["x"]
            y   = pred["y"]
            w   = pred["width"]
            h   = pred["height"]
            detections.append(Detection(
                label="f1_car",
                weight=1.0,
                conf=float(pred["confidence"]),
                x1=x - w / 2,
                y1=y - h / 2,
                x2=x + w / 2,
                y2=y + h / 2,
            ))

        h_img, w_img = img_rgb.shape[:2]
        detections.sort(
            key=lambda d: d.subject_score(w_img, h_img), reverse=True
        )
        return detections

