"""Stage-level profile of the decode + scoring chain on real camera files.

Consumer-serial per-frame stages (what bounds E2E at workers>=4):
  decode (in production: decode pool worker) -> prepare (letterbox+normalize)
  -> f1 run+post -> [coco run+post on f1 miss] -> sharpness -> composition
  -> P4 predict_roi.
Median of 3 passes per file; models warmed up before timing.
"""
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from cull.loader import load_image_rgb
from cull.detector import LiteYOLO, detect
from cull.sharpness import score_sharpness
from cull.composition import score_composition
from cull.p4_classifier import P4Classifier

SETS = {
    "JPG": sorted(Path("tests/test_img").glob("*.jpg")),
    "HEIF": sorted(Path("test_import").glob("*.heif"))[:24],
    "ARW": sorted(Path("test_arw").glob("*.ARW"))[:20],
    "NEF": sorted(Path("test_nef").glob("*.nef"))[:20],
}


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return (time.perf_counter() - t0) * 1000, r


def main() -> None:
    f1 = LiteYOLO(Path("models/f1_yolov8n.onnx"))
    coco = LiteYOLO(Path("models/yolov8n.onnx"))
    p4 = P4Classifier()
    p4.session.run(None, {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})

    warm = load_image_rgb(SETS["JPG"][0], scale_width=1280)
    for _ in range(5):
        prep = f1.prepare_numpy(warm)
        f1.detect_numpy(warm, prep=prep)
        coco.detect_numpy(warm, prep=prep)
        score_sharpness(warm, None)
        p4.predict_roi(warm, (0, 0, warm.shape[1] // 2, warm.shape[0] // 2))

    for fmt, files in SETS.items():
        acc = {k: [] for k in ("decode", "prepare", "f1_run", "f1_post",
                               "coco_run", "sharp", "comp", "p4")}
        misses = 0
        widths = []
        n_frames = 0
        for p in files:
            for _ in range(3):
                n_frames += 1
                dt, img = timed(load_image_rgb, p, scale_width=1280)
                acc["decode"].append(dt)
                widths.append(img.shape[1])

                # production-equivalent detection (returns Detection objects)
                dt, dets = timed(detect, img, f1, coco)
                # decomposition of detect() internals
                dt, prep = timed(f1.prepare_numpy, img)
                acc["prepare"].append(dt)
                prep = f1.prepare_numpy(img)
                dt, out = timed(f1.session.run, None, {f1.input_name: prep[0]})
                acc["f1_run"].append(dt)
                dt, df1 = timed(f1._postprocess, out, *prep[1:4], 0.25, 0.45)
                acc["f1_post"].append(dt)
                if not df1:
                    misses += 1
                    dt, outc = timed(coco.session.run, None, {coco.input_name: prep[0]})
                    dt2, _ = timed(coco._postprocess, outc, *prep[1:4], 0.25, 0.45)
                    acc["coco_run"].append(dt + dt2)
                d0 = dets[0] if dets else None
                bbox = (d0.x1, d0.y1, d0.x2, d0.y2) if d0 else (0, 0, img.shape[1] // 2, img.shape[0] // 2)
                dt, _ = timed(score_sharpness, img, d0)
                acc["sharp"].append(dt)
                h, wd = img.shape[:2]
                dt, _ = timed(score_composition, dets, wd, h, None, True)
                acc["comp"].append(dt)
                dt, _ = timed(p4.predict_roi, img, bbox)
                acc["p4"].append(dt)

        med = {k: statistics.median(v) if v else 0.0 for k, v in acc.items()}
        hit = misses / max(1, n_frames)
        consumer = (med["prepare"] + med["f1_run"] + med["f1_post"]
                    + med["coco_run"] * hit + med["sharp"] + med["comp"] + med["p4"])
        print(f"\n[{fmt}] n={len(files)} files x3 reps, decoded ~{np.mean(widths):.0f}px wide, "
              f"coco-cascade rate {hit:.0%}")
        print(f"  decode (pool worker)  {med['decode']:7.1f} ms")
        print(f"  prepare letterbox     {med['prepare']:7.1f} ms   |")
        print(f"  f1 session.run (DML)  {med['f1_run']:7.1f} ms   |")
        print(f"  f1 postprocess        {med['f1_post']:7.1f} ms   |")
        if med["coco_run"]:
            print(f"  coco run+post (miss)  {med['coco_run']:7.1f} ms   | x{hit:.0%} frames")
        print(f"  sharpness             {med['sharp']:7.1f} ms   |")
        print(f"  composition           {med['comp']:7.1f} ms   |")
        print(f"  P4 predict_roi        {med['p4']:7.1f} ms   |")
        print(f"  => consumer serial    {consumer:7.1f} ms/frame  (single-consumer cap ~{1000 / consumer:.0f} fps)")


if __name__ == "__main__":
    main()
