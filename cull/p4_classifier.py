import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from cull.detector import ensure_nvidia_runtime_on_path

log = logging.getLogger(__name__)

def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except Exception:
        base_path = Path(__file__).parent.parent.resolve()
    return base_path / relative_path

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
        base = Path(model_path)
        if not base.exists():
            # Packaged single-file binaries resolve model files from the
            # PyInstaller bundle (_MEIPASS); source runs keep the CWD path.
            bundled = get_resource_path(model_path)
            if bundled.exists():
                base = bundled
        self.model_path = base
        try:
            import onnxruntime as ort
            import os as _os
            from cull.deterministic import is_deterministic
            deterministic = is_deterministic()
            ensure_nvidia_runtime_on_path()
            available = ort.get_available_providers()
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3
            if deterministic:
                sess_opts.intra_op_num_threads = 1
                sess_opts.inter_op_num_threads = 1
                try:
                    sess_opts.use_deterministic_compute = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                providers = ["CPUExecutionProvider"]
                model_file = self.model_path
                opts_for_session = sess_opts
            else:
                model_file = self.model_path
                opts_for_session = ort.SessionOptions()
                opts_for_session.log_severity_level = 3
                # On darwin the single-partition ANE graph is the DEFAULT: with the
                # ThreadPool engine (bounded per-burst submission) it MEASURED
                # +12.6% at 600 JPGs over CPU EP (42.6 vs 37.8 img/s, interleaved
                # A/B 2026-08-25). Only small batches (<~100 files) still favor
                # CPU EP where fixed costs dominate. Set CULL_P4_NATIVE=0 to force
                # the CPU EP path.
                use_ane = _os.environ.get("CULL_P4_NATIVE", "1") != "0"
                if sys.platform == "darwin":
                    from cull.detector import _has_concrete_input_shape as _frozen
                    static_ane = self.model_path.with_name(self.model_path.stem + "_static_ane.onnx")
                    if use_ane and not static_ane.exists():
                        # Packaged binaries ship the frozen graph under the base
                        # name; treat it as the ANE graph when its input dims are
                        # concrete (shape probe, not the CoreML partition count).
                        static_ane = self.model_path if _frozen(self.model_path, ort.SessionOptions()) else None
                    if use_ane and static_ane is not None and static_ane.exists():
                        # Single-partition graph (HardSigmoid/HardSwish unfolded to
                        # Clip/Mul/Add — exact identities): ALL 216/216 nodes run in
                        # ONE CoreML partition on the ANE, 0.4-1.2 ms standalone vs
                        # ~5 ms on CPU (Apple M4, 2026-08-25). Logit diff <= 0.016,
                        # gates bit-identical.
                        providers = [
                            ("CoreMLExecutionProvider", {
                                "RequireStaticInputShapes": "1",
                                "MLComputeUnits": "CPUAndNeuralEngine",
                            }),
                            "CPUExecutionProvider",
                        ]
                        model_file = static_ane
                    else:
                        # CoreML fragments this model to 20/77 nodes (MobileNetV3's
                        # SE/hard-swish composition; HardSigmoid and HardSwish are
                        # unsupported by the EP) and measured 16.6 ms vs 5.0 ms CPU.
                        providers = ['CPUExecutionProvider']
                else:
                    providers = []
                    for p in ['CoreMLExecutionProvider', 'CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']:
                        if p in available: providers.append(p)
                    if not providers: providers = ['CPUExecutionProvider']
            try:
                self.session = ort.InferenceSession(str(model_file), providers=providers,
                                                    sess_options=sess_opts if deterministic else opts_for_session)
            except Exception:
                # GPU EP (CUDA/DML) unusable on this box (no device/driver) —
                # fall back to CPU so CI and GPU-less machines still score.
                if "CPUExecutionProvider" in [p if isinstance(p, str) else p[0] for p in providers]:
                    raise
                self.session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"],
                                                    sess_options=sess_opts if deterministic else opts_for_session)
            
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

        # cv2 INTER_LINEAR resize (GIL-released). The v2 P4 model is trained
        # with resize-kernel randomization so this no longer flips integrity
        # verdicts (v1 required PIL BILINEAR — see performance_baseline.md).
        import cv2
        roi = cv2.resize(roi_arr, (224, 224), interpolation=cv2.INTER_LINEAR)
        roi = roi.astype(np.float32) / 255.0
        
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
        if not model_path.exists():
            bundled = get_resource_path("models/p4_car_model.onnx")
            if bundled.exists(): model_path = bundled
        
        if model_path.exists():
            _p4_classifier = P4Classifier(str(model_path))
        else:
            log.warning("P4 model not found at models/p4_car_model.onnx")
    return _p4_classifier
