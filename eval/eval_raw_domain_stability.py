#!/usr/bin/env python3
"""eval/eval_raw_domain_stability.py — P4 verdict stability on RAW previews.

The labeled-val flip-rate metric can be optimistic (synthetic cuts in the
val split). This checks the thing that actually gates decode-path unlocks:
for every RAW/HEIF gate file, does the P4 integrity verdict change when the
ROI resize kernel changes (cv2 AREA/LINEAR/CUBIC + PIL BILINEAR)?

Runs the production extraction+decode chain (exiftool persistent session /
ffmpeg), YOLO detection, then P4 on the SAME ROI under 4 kernels, and counts
verdict flips + mean |p-0.5| margin on the flippers.

Usage:
    python eval/eval_raw_domain_stability.py [--dir test_arw] [--dir test_nef]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cull.loader import load_image_rgb  # noqa: E402
from cull.detector import load_f1_model, detect  # noqa: E402
from cull.p4_classifier import P4Classifier  # noqa: E402

KERNELS = [
    ("cv2_AREA", cv2.INTER_AREA),
    ("cv2_LINEAR", cv2.INTER_LINEAR),
    ("cv2_CUBIC", cv2.INTER_CUBIC),
    ("pil_BILINEAR", Image.BILINEAR),
]


def variants(roi: np.ndarray):
    outs = []
    for name, k in KERNELS:
        if isinstance(k, int):
            outs.append(cv2.resize(roi, (224, 224), interpolation=k))
        else:
            outs.append(np.array(Image.fromarray(roi).resize((224, 224), k)))
    return outs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", action="append", required=True)
    parser.add_argument("--f1-model", default="models/f1_yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    f1 = load_f1_model(ROOT / args.f1_model)
    try:
        p4 = P4Classifier()
    except Exception:
        p4 = None

    flips = 0
    n_files = 0
    margins = []
    flip_files: list[tuple[str, list[int], list[float]]] = []

    for d in args.dir:
        base = Path(d)
        for p in sorted(base.iterdir()):
            if p.suffix.lower() not in (".arw", ".nef", ".heif", ".hif"):
                continue
            img = load_image_rgb(p, scale_width=1280)
            if img is None:
                continue
            dets = detect(img, f1, None, conf=args.conf)
            primary = next((d for d in dets if d.label == "f1_car"), None)
            if primary is None:
                continue
            h, w = img.shape[:2]
            x1, y1 = max(0, int(primary.x1)), max(0, int(primary.y1))
            x2, y2 = min(w, int(primary.x2)), min(h, int(primary.y2))
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            roi = img[y1:y2, x1:x2]
            n_files += 1
            preds, probs = [], []
            for v in variants(roi):
                o_str, o_conf, i_pred, i_prob = p4.predict_roi(v, (0, 0, v.shape[1], v.shape[0]))
                preds.append(i_pred)
                probs.append(abs(i_prob - 0.5))
            margins.append(probs[0])
            if len(set(preds)) > 1:
                flips += 1
                flip_files.append((p.name, preds, sorted(probs)))

    print(f"files with detections: {n_files}")
    print(f"integrity verdict flips across 4 kernels: {flips}/{n_files} = {flips/max(1,n_files):.1%}")
    print(f"mean |p-0.5| margin (0=knife edge): {np.mean(margins):.4f}")
    for name, preds, probs in flip_files[:10]:
        print(f"  FLIP {name}: preds={preds} margins={[round(m,3) for m in probs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())