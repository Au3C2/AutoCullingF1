#!/usr/bin/env python3
"""utils/generate_p4_cuts.py — synthesize cut samples from full ROIs.

The integrity training already patches class balance online (dynamic-cut
augmentation), but offline synthesis fills two real gaps that online
augmentation cannot reach:

1. sparse orientations: front_cut / rear_cut have 1-2 samples each
2. the RAW embedded-preview domain: no labeled cuts exist there at all

Cut = clip 30-55% of the ROI from one edge (matches the label guide's ~1/3
criterion and the trainer's dynamic-cut range), then re-encode JPEG with
random quality + slight noise so the synthesized cut inherits the source
domain's statistics. A clipped car ROI is ground-truth "cut" by construction,
so sources do not need manual labels.

Sources (Tier 1, orientation exact): every labeled `*_full` ROI.
Sources (Tier 2, RAW domain): F1-detector ROIs from RAW cameras whose
orientation the current P4 model classifies with confidence > 0.9 — covers
the ARC/NEF preview domain the labeled set lacks.

Usage:
    python utils/generate_p4_cuts.py                    # Tier 1 + Tier 2
    python utils/generate_p4_cuts.py --tier 1           # labeled fulls only
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ORIENT_MAP = {"front": 0, "front_angle": 1, "side": 2, "rear_angle": 3, "rear": 4}
ORIENT_NAMES = {v: k for k, v in ORIENT_MAP.items()}


def load_p4():
    from cull.p4_classifier import P4Classifier
    return P4Classifier()


def make_cut(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Clip 30-55% of the ROI from one edge and re-encode like a camera."""
    h, w = img.shape[:2]
    ratio = rng.uniform(0.30, 0.55)
    side = rng.randint(0, 3)
    if side == 0:
        img = img[:, int(w * ratio):]
    elif side == 1:
        img = img[:, :int(w * (1 - ratio))]
    elif side == 2:
        img = img[int(h * ratio):, :]
    else:
        img = img[:int(h * (1 - ratio)), :]
    if img.size == 0:
        return None
    # domain-transfer jitter: mild noise + JPEG recompression
    noise = np.random.default_rng().normal(0, rng.uniform(0, 1.2), img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 90)])
    if not ok:
        return None
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2])
    parser.add_argument("--labeled-dir", default="p4_data/labeled")
    parser.add_argument("--cuts-per-source", type=int, default=4)
    parser.add_argument("--raw-dirs", nargs="+", default=["test_arw", "test_nef"])
    parser.add_argument("--raw-roi-dir", default="p4_data/unlabeled_raw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orient-conf", type=float, default=0.90,
                        help="min orientation confidence for Tier-2 RAW sources")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labeled = ROOT / args.labeled_dir
    n_generated = 0

    # ---- Tier 1: labeled fulls with exact orientation ---------------------
    for cat in sorted(labeled.iterdir()):
        if not cat.is_dir() or not cat.name.endswith("_full"):
            continue
        orient = cat.name.rsplit("_", 1)[0]
        out = labeled / f"{orient}_cut"
        out.mkdir(exist_ok=True)
        for src in sorted(cat.glob("*.jpg")):
            img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for i in range(args.cuts_per_source):
                cut = make_cut(img, rng)
                if cut is None:
                    continue
                dst = out / f"{src.stem}_syn{i}_{rng.randint(1000, 9999)}.jpg"
                cv2.imencode(".jpg", cut)[1].tofile(str(dst))
                n_generated += 1
    print(f"Tier 1 done: {n_generated} synthesized cuts from labeled fulls")

    if args.tier < 2:
        return 0

    # ---- Tier 2: RAW-domain cuts via orientation-pseudo-labeled ROIs ------
    roi_dir = ROOT / args.raw_roi_dir
    if not roi_dir.is_dir():
        print(f"Tier 2 skipped: {roi_dir} missing (run "
              f"utils/extract_p4_rois_dir.py --dir test_arw --dir test_nef "
              f"--out-dir p4_data/unlabeled_raw first)")
        return 0
    p4 = load_p4()
    n_raw = 0
    for src in sorted(roi_dir.glob("*.jpg")):
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(str(src), dtype=np.uint8),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if img is None:
            continue
        orient, conf, integ, _ = p4.predict_roi(img, (0, 0, img.shape[1], img.shape[0]))
        if orient not in ORIENT_MAP or conf < args.orient_conf:
            continue
        out = labeled / f"{orient}_cut"
        out.mkdir(exist_ok=True)
        for i in range(args.cuts_per_source):
            cut = make_cut(img, rng)
            if cut is None:
                continue
            cut = cv2.cvtColor(cut, cv2.COLOR_RGB2BGR)
            dst = out / f"{src.stem}_raw{i}_{rng.randint(1000, 9999)}.jpg"
            cv2.imencode(".jpg", cut)[1].tofile(str(dst))
            n_raw += 1
            n_generated += 1
    print(f"Tier 2 done: {n_raw} RAW-domain cuts (orientation conf>={args.orient_conf})")
    print(f"TOTAL: {n_generated} new cut samples in {labeled}")
    return 0


if __name__ == "__main__":
    sys.exit(main())