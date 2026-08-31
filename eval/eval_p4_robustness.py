#!/usr/bin/env python3
"""eval/eval_p4_robustness.py — accuracy + resize-kernel consistency of P4.

Runs a labeled-ROI validation split (same seed-42 80/20 split as
train/train_p4_multitask.py) through an ONNX P4 model under multiple resize
kernels and reports:

  - integrity accuracy / precision / recall / F1 (cv2 INTER_AREA reference)
  - orientation accuracy
  - kernel flip rate: fraction of val images whose binary integrity verdict
    changes across kernels (the metric that gates pixel-path optimizations)
  - mean decision margin |p - 0.5| (distance from the knife edge)

Usage:
    python eval/eval_p4_robustness.py --model models/p4_car_model.onnx
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

ORIENT_MAP = {"front": 0, "front_angle": 1, "side": 2, "rear_angle": 3, "rear": 4}
ORIENT_NAMES = {v: k for k, v in ORIENT_MAP.items()}


def load_labeled(data_dir: Path):
    """Return (paths, orient_labels, integ_labels) mirroring the trainer."""
    paths, o_labels, i_labels = [], [], []
    for cat_dir in sorted(data_dir.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name in ("无效数据", "ignore"):
            continue
        parts = cat_dir.name.rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in ORIENT_MAP:
            continue
        o, i = ORIENT_MAP[parts[0]], 1 if parts[1] == "full" else 0
        for p in cat_dir.glob("*.jpg"):
            paths.append(p)
            o_labels.append(o)
            i_labels.append(i)
    return paths, o_labels, i_labels


def val_crop(img: np.ndarray) -> np.ndarray:
    """Validation protocol: recover the YOLO native box (center 1/1.3 crop)."""
    h, w = img.shape[:2]
    nw, nh = int(w / 1.3), int(h / 1.3)
    x1, y1 = (w - nw) // 2, (h - nh) // 2
    return img[y1:y1 + nh, x1:x1 + nw]


KERNELS_CV2 = {
    "cv2_AREA": cv2.INTER_AREA,
    "cv2_LINEAR": cv2.INTER_LINEAR,
    "cv2_NEAREST": cv2.INTER_NEAREST,
    "cv2_CUBIC": cv2.INTER_CUBIC,
}

KERNELS_PIL = {
    "pil_BILINEAR": Image.BILINEAR,
    "pil_BICUBIC": Image.BICUBIC,
    "pil_BOX": Image.BOX,
    "pil_HAMMING": Image.HAMMING,
    "pil_LANCZOS": Image.LANCZOS,
}


def resize_with(name: str, img: np.ndarray, size=(224, 224)) -> np.ndarray:
    if name in KERNELS_CV2:
        return cv2.resize(img, size, interpolation=KERNELS_CV2[name])
    pil = Image.fromarray(img).resize(size, KERNELS_PIL[name])
    return np.array(pil)


class P4ONNX:
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def predict(self, img224: np.ndarray) -> tuple[int, float, int]:
        x = img224.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))[None]
        o_logits, i_logits = self.sess.run(None, {"input": x})
        integ_prob = float(1.0 / (1.0 + np.exp(-i_logits[0])))
        integ = 1 if integ_prob > 0.5 else 0
        orient = int(np.argmax(o_logits[0]))
        return integ, integ_prob, orient


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", default="p4_data/labeled")
    parser.add_argument("--model", default="models/p4_car_model.onnx")
    args = parser.parse_args()

    paths, o_labels, i_labels = load_labeled(ROOT / args.data_dir)
    total = len(paths)
    if total == 0:
        print("no labeled images found")
        return 1
    indices = np.arange(total)
    np.random.seed(42)
    np.random.shuffle(indices)
    val_idx = indices[int(0.8 * total):]
    print(f"labeled={total}, val={len(val_idx)} (seed-42 split, same as trainer)")

    model = P4ONNX(ROOT / args.model)

    kernel_names = list(KERNELS_CV2) + list(KERNELS_PIL)
    # per-kernel stats + per-image cross-kernel predictions
    tp = fp = fn = tn = 0
    o_correct = 0
    margins: list[float] = []
    flips = 0
    n_eval = 0
    orient_confusion = Counter()

    for idx in val_idx:
        img = cv2.imdecode(np.fromfile(str(paths[idx]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        roi = val_crop(img)
        preds, probs, orient_pred = [], [], []
        for k in kernel_names:
            i_p, prob, o_p = model.predict(resize_with(k, roi))
            preds.append(i_p)
            probs.append(prob)
            orient_pred.append(o_p)
        ref = preds[0]  # cv2 INTER_AREA reference
        gt = i_labels[idx]
        n_eval += 1
        if ref == 1 and gt == 1: tp += 1
        elif ref == 1 and gt == 0: fp += 1
        elif ref == 0 and gt == 1: fn += 1
        else: tn += 1
        if orient_pred[0] == gt_orient(o_labels, idx): o_correct += 1
        else: orient_confusion[(ORIENT_NAMES[orient_pred[0]], ORIENT_NAMES[o_labels[idx]])] += 1
        margins.append(abs(probs[0] - 0.5))
        if len(set(preds)) > 1:
            flips += 1

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"\nintegrity  acc={(tp+tn)/n_eval:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  (tp={tp} fp={fp} fn={fn} tn={tn})")
    print(f"orientation acc={o_correct/n_eval:.4f}")
    print(f"kernel flip rate: {flips}/{n_eval} = {flips/n_eval:.3%}")
    print(f"mean decision margin |p-0.5|: {np.mean(margins):.4f}")
    if orient_confusion:
        top = orient_confusion.most_common(5)
        print("top orient confusions (pred->gt):", top)
    return 0


def gt_orient(labels, idx):
    return labels[idx]


if __name__ == "__main__":
    sys.exit(main())