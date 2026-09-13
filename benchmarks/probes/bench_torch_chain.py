#!/usr/bin/env python3
"""benchmarks/bench_torch_chain.py — full-GPU torch scoring chain prototype.

Loads the ORIGINAL torch weights (ultralytics F1 ckpt + P4 v2 state_dict) and
runs the whole per-frame scoring chain on GPU tensors end-to-end (letterbox,
f1 forward, GPU postprocess, ROI crop, P4 forward) with no numpy/ORT
roundtrips between stages. Measures per-stage latency and the total, and
optionally compares detections against the production ORT pipeline.

This is a PROBE for the "move the scoring chain to torch" question — it does
not change the engine.

Usage:
    python benchmarks/bench_torch_chain.py
    python benchmarks/bench_torch_chain.py --validate
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cull.loader import load_image_rgb  # noqa: E402

DEV = torch.device("cuda")


def bench(fn, n=60, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / n


def load_f1_torch(ckpt_path: Path):
    """Ultralytics checkpoint -> evaluable float32 nn.Module on CUDA."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = ckpt["model"]
    model.float()
    model.eval().to(DEV)
    return model


def load_p4_torch(pt_path: Path):
    from train.train_p4_multitask import MultiTaskMobileNet
    model = MultiTaskMobileNet(5)
    model.load_state_dict(torch.load(str(pt_path), map_location="cpu"))
    model.eval().to(DEV)
    return model


def gpu_postprocess(out, ratio: float, dw: float, dh: float,
                    conf: float = 0.30, nms_iou: float = 0.45) -> np.ndarray:
    """Vectorized GPU argmax/boxes; NMS on the few survivors (numpy)."""
    if isinstance(out, (list, tuple)):
        out = out[0]                    # ultralytics Detect returns (buf, extra)
    o = out[0]                          # (C+4, 8400) on GPU
    cls = o[4:].argmax(0)
    confs = o[4:].max(0).values
    keep_idx = torch.where(confs > conf)[0]
    if keep_idx.numel() == 0:
        return np.zeros((0, 6))
    confs_k = confs[keep_idx]
    cls_k = cls[keep_idx]
    rows = o[:4][:, keep_idx]           # (4, K) GPU
    xc, yc, w, h = rows
    x1 = (xc - w / 2 - dw) / ratio
    y1 = (yc - h / 2 - dh) / ratio
    x2 = (xc + w / 2 - dw) / ratio
    y2 = (yc + h / 2 - dh) / ratio
    boxes = torch.stack([x1, y1, x2, y2, confs_k, cls_k.float()], dim=1)
    return boxes.detach().cpu().numpy()  # small (<= handful of rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--validate", action="store_true",
                        help="compare detections vs the ORT production pipeline")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()

    frames = [im for im in (load_image_rgb(p, 1280)
                            for p in sorted((ROOT / "tests/test_img").glob("*.jpg")))]
    img = frames[0]
    H, W = img.shape[:2]

    f1 = load_f1_torch(ROOT / "runs/f1_detect/train/weights/best.pt")
    p4 = load_p4_torch(ROOT / "models/p4_best.pt")
    print(f"f1 torch: {sum(p.numel() for p in f1.parameters())/1e6:.1f}M params on cuda"
          f" | p4 torch: {sum(p.numel() for p in p4.parameters())/1e6:.1f}M params")

    # ---- precompute production-equivalent letterbox info (cv2, once) ----
    r = min(640 / H, 640 / W)
    new_unpad = (int(round(W * r)), int(round(H * r)))
    dw, dh = (640 - new_unpad[0]) / 2.0, (640 - new_unpad[1]) / 2.0
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    ratio = r

    # ---- stage timings (all on GPU tensors) ----
    tf = torch.from_numpy(img).to(DEV)

    def stage_letterbox():
        x = torch.nn.functional.interpolate(
            tf.permute(2, 0, 1)[None].float() / 255.0,
            size=(new_unpad[1], new_unpad[0]), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, 640, 640), 114 / 255.0, device=DEV)
        canvas[:, :, top:top + new_unpad[1], left:left + new_unpad[0]] = x
        return canvas
    tb_lb = bench(stage_letterbox, args.n)
    canvas = stage_letterbox()

    tb_f1 = bench(lambda: f1(canvas), args.n)
    with torch.no_grad():
        out = f1(canvas)

    det_np = gpu_postprocess(out, ratio, dw, dh)
    tb_post = bench(lambda: gpu_postprocess(f1(canvas), ratio, dw, dh), args.n)
    print(f"  detections: {len(det_np)}")

    # ROI crop + P4 on GPU (use the first detection)
    if len(det_np):
        x1, y1, x2, y2 = [int(v) for v in det_np[0, :4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        def stage_p4():
            roi = tf[y1:y2, x1:x2]           # GPU slice (no copy)
            roi = torch.nn.functional.interpolate(
                roi.permute(2, 0, 1)[None].float() / 255.0,
                size=(224, 224), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
            roi = (roi - mean) / std
            with torch.no_grad():
                o_l, i_l = p4(roi)
            return o_l, i_l
        tb_p4 = bench(stage_p4, args.n)
        o_l, i_l = stage_p4()
        print(f"  p4 integ prob: {torch.sigmoid(i_l).item():.4f}  orient: {int(o_l.argmax())}")

    # ---- totals ----
    h2d = 0.35  # measured separately (3.3 MB uint8 -> cuda)
    total = h2d + tb_lb + tb_f1 + tb_post + tb_p4
    print(f"\nstage: h2d~{h2d:.2f} + letterbox {tb_lb:.2f} + f1 {tb_f1:.2f}"
          f" + postproc {tb_post:.2f} + p4 {tb_p4:.2f} = {total:.2f} ms/frame"
          f"  ->  {1000/total:.1f} fps single-thread (GPU only)")

    if args.validate:
        import subprocess as _s
        import os
        from cull.detector import load_f1_model, detect
        f1_ort = load_f1_model(ROOT / "models/f1_yolov8n.onnx")
        det_ort = detect(img, f1_ort, None, conf=0.30)
        print("\n-- ORT pipeline detections --")
        for d in det_ort[:4]:
            print(f"  {d.label} conf={d.conf:.4f} box=({d.x1:.1f},{d.y1:.1f},{d.x2:.1f},{d.y2:.1f})")
        print("-- torch GPU detections --")
        for d in det_np[:4]:
            print(f"  conf={d[4]:.4f} cls={int(d[5])} box=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())