#!/usr/bin/env python3
"""utils/extract_p4_rois_dir.py — extract unlabeled P4 ROI crops from image dirs.

Scans one or more photo directories (any camera: Sony HIF/ARW, Nikon NEF,
phone JPG...), detects the primary F1 car and saves padded ROI crops for
manual labeling into ``p4_data/labeled/<orient>_<integrity>/`` folders.

Same extraction protocol as the original training set (decode @1280px,
primary f1_car box + 15% padding) so new labels mix cleanly with v1 data.

Usage:
    python utils/extract_p4_rois_dir.py --dir D:/photos/nikon_gp --out-dir p4_data/unlabeled_nikon
    python utils/extract_p4_rois_dir.py --dir a/ --dir b/ --max-samples 300
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cull.loader import load_image_rgb, EXTENSIONS  # noqa: E402
from cull.detector import load_f1_model, detect  # noqa: E402


def crop_roi(img_rgb, detection, pad_ratio=0.0):
    """Crop the bounding box from the image, converting back to BGR for saving."""
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, int(x1) - pad_w)
    y1 = max(0, int(y1) - pad_h)
    x2 = min(w, int(x2) + pad_w)
    y2 = min(h, int(y2) + pad_h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi_rgb = img_rgb[y1:y2, x1:x2]
    return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir", action="append", required=True,
                        help="photo directory to scan (repeatable)")
    parser.add_argument("--out-dir", default="p4_data/unlabeled",
                        help="output dir for unlabeled ROIs")
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--f1-model", default="models/f1_yolov8n.onnx")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading F1 YOLO model...")
    f1_model = load_f1_model(ROOT / args.f1_model)

    paths: list[Path] = []
    for d in args.dir:
        base = Path(d)
        if not base.is_dir():
            print(f"skip (not a dir): {base}")
            continue
        found = [p for p in base.iterdir()
                 if p.is_file() and p.suffix.lower() in EXTENSIONS
                 and not p.name.startswith("._")]
        print(f"{base}: {len(found)} images")
        paths.extend(found)
    random.seed(42)
    random.shuffle(paths)

    count = 0
    for i, path in enumerate(paths):
        if count >= args.max_samples:
            break
        try:
            img_rgb = load_image_rgb(path, scale_width=1280)
            if img_rgb is None:
                continue
            dets = detect(img_rgb, f1_model, None, conf=args.conf)
            primary = next((d for d in dets if d.label == "f1_car"), None)
            if primary is None:
                continue
            roi_bgr = crop_roi(img_rgb, primary, pad_ratio=0.15)
            if roi_bgr is None or roi_bgr.size == 0:
                continue
            h, w = roi_bgr.shape[:2]
            if w < 50 or h < 50:
                continue
            out_name = f"{path.parent.name}_{path.stem}_roi.jpg"
            ok, buf = cv2.imencode(".jpg", roi_bgr)
            if ok:
                (out_dir / out_name).write_bytes(buf.tobytes())
                count += 1
            if count % 100 == 0 and count:
                print(f"Extracted {count}/{args.max_samples} ROIs...")
        except Exception:
            continue
    print(f"\nDone! Extracted {count} ROIs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())