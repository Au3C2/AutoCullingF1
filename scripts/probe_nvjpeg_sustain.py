"""Sustained nvJPEG GPU decode throughput: 1 process vs 2 processes.
Composite = read file -> decode(cuda) -> GPU resize 1280 -> download."""
import sys, time
from pathlib import Path
import numpy as np

def worker(files, n_loops, q):
    import torch
    import torchvision.io as tio
    import torch.nn.functional as F
    # warm
    for p in files[:2]:
        buf = torch.from_numpy(np.fromfile(str(p), np.uint8))
        t = tio.decode_jpeg(buf, device="cuda").float()[None]
        h, w = t.shape[2], t.shape[3]
        F.interpolate(t, size=(round(h*1280/w), 1280), mode="bilinear", align_corners=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(n_loops):
        for p in files:
            buf = torch.from_numpy(np.fromfile(str(p), np.uint8))
            t = tio.decode_jpeg(buf, device="cuda").float()[None]
            h, w = t.shape[2], t.shape[3]
            small = F.interpolate(t, size=(round(h*1280/w), 1280), mode="bilinear", align_corners=False)
            _ = small[0].permute(1,2,0).round().clamp_(0,255).byte().contiguous().cpu().numpy()
            n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    q.put(n / dt)

if __name__ == "__main__":
    import multiprocessing as mp
    files = sorted(Path("tests/test_img").glob("*.jpg"))
    ctx = mp.get_context("spawn")
    for nproc in (1, 2):
        q = ctx.Queue()
        ps = [ctx.Process(target=worker, args=(files, 8, q)) for _ in range(nproc)]
        [p.start() for p in ps]
        rates = [q.get() for _ in ps]
        [p.join() for p in ps]
        print(f"{nproc} proc: total {sum(rates):.1f} fps  (per-proc {[f'{r:.1f}' for r in rates]})")
