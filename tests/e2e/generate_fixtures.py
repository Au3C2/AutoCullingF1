#!/usr/bin/env python3
"""tests/e2e/generate_fixtures.py — deterministic rendering fixtures for the
Playwright E2E suite.

Runs the REAL culling engine (force + dry-run: full scoring, zero disk writes
to photo metadata) over the local camera sample dirs, then captures per-photo:
  - 640 px preview PNG (base64) exactly as the GUI IPC returns it
  - detection boxes + crop as emitted in the preview payload
  - the high-res Tier-2/HEIF cache JPEG, copied under fixtures/highres/

Output (gitignored, generated at test time):
  tests/e2e/fixtures/previews.json
  tests/e2e/fixtures/highres/<stem>.jpg
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def collect_samples(limit_per_dir: int = 8) -> list[Path]:
    """Pick a deterministic, small, format-diverse photo set."""
    chosen: list[Path] = []
    candidates: list[Path] = []
    for sub, exts in (
        ("test_arw", (".ARW",)),
        ("test_nef", (".nef",)),
    ):
        d = ROOT / sub
        if d.is_dir():
            candidates.extend(sorted(p for p in d.iterdir() if p.suffix in exts))
    for name in ("seed.heif", "seed.jpg"):
        p = ROOT / "tests" / "ci" / "sample" / name
        if p.exists():
            candidates.append(p)

    seen_dirs: dict[str, int] = {}
    for p in candidates:
        key = str(p.parent)
        if seen_dirs.get(key, 0) >= limit_per_dir:
            continue
        seen_dirs[key] = seen_dirs.get(key, 0) + 1
        chosen.append(p)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "e2e" / "fixtures")
    parser.add_argument("--limit-per-dir", type=int, default=8)
    args = parser.parse_args()

    out: Path = args.out
    highres_dir = out / "highres"
    highres_dir.mkdir(parents=True, exist_ok=True)

    photos = collect_samples(args.limit_per_dir)
    if not photos:
        print("No camera sample files found; nothing to generate.", file=sys.stderr)
        return 1
    print(f"Generating fixtures for {len(photos)} photos -> {out}")

    from cull.engine import CullingEngine, EngineConfig
    from cull.gui.preview import render_pil
    from cull.highres import HighResProvider, HighResRequest

    # Group everything into one dir-less run: force full scoring (existing XMP
    # sidecars would otherwise short-circuit to manual_metadata) while
    # dry-run guarantees zero metadata writes to the sample files.
    run_dir = photos[0].parent
    config = EngineConfig(
        input_dir=run_dir,
        force=True,
        dry_run=True,
        workers=4,
        log_full_paths=False,
    )
    engine = CullingEngine(config)
    engine.image_paths = photos
    scores, _elapsed = engine.run()

    score_by_path = {s.path: s for s in scores}
    provider = HighResProvider(cache_dir=highres_dir)

    fixture_photos = []
    crop_assigned = False
    for p in photos:
        score = score_by_path.get(p)
        if score is None:
            from cull.scorer import ImageScore
            score = ImageScore(path=p, s_sharp=0.0, s_comp=0.0, raw_score=0.0, rating=0)

        pil = render_pil(score, max_size=640)
        if pil is None:
            print(f"  SKIP (preview render failed): {p.name}", file=sys.stderr)
            continue
        buf = io.BytesIO()
        pil.save(buf, format="PNG")

        boxes = []
        if getattr(score, "detections", None):
            # Normalized [0,1] photo-relative coordinates (same convention as
            # the GUI preview payload — see cull_photos.do_preview)
            img_w = float(getattr(score, "img_w", 0) or 0)
            img_h = float(getattr(score, "img_h", 0) or 0)
            if img_w <= 0 or img_h <= 0:
                img_w, img_h = 1.0, 1.0
            for det in score.detections:
                boxes.append([
                    max(0.0, min(1.0, float(det.x1) / img_w)),
                    max(0.0, min(1.0, float(det.y1) / img_h)),
                    max(0.0, min(1.0, float(det.x2) / img_w)),
                    max(0.0, min(1.0, float(det.y2) / img_h)),
                    str(det.label), float(det.conf),
                ])
        crop = [float(x) for x in score.crop] if getattr(score, "crop", None) else None

        # High-res cache asset (Tier 2 / HEIF preview stream / Tier 3)
        highres_url = None
        hr = provider.resolve(HighResRequest(file_path=p, gen_id=1))
        if hr is not None and hr.resolved_path.exists():
            dest = highres_dir / f"{p.stem}.jpg"
            shutil.copyfile(hr.resolved_path, dest)
            highres_url = f"/highres/{dest.name}"

        # Ensure at least one fixture carries a crop so the auto crop-focus →
        # high-res on-demand path is exercisable even when scoring finds no
        # single-car frame in the sample set.
        nonlocal_crop = crop
        if nonlocal_crop is None and not crop_assigned and boxes:
            nonlocal_crop = [0.30, 0.25, 0.75, 0.70]
            crop_assigned = True

        fixture_photos.append({
            "path": str(p),
            "name": p.name,
            "dir": str(p.parent),
            "data": base64.b64encode(buf.getvalue()).decode("ascii"),
            "width": pil.width,
            "height": pil.height,
            "boxes": boxes,
            "crop": nonlocal_crop,
            "rating": int(score.rating),
            "highresUrl": highres_url,
            "highresWidth": int(hr.width) if hr else 0,
            "highresHeight": int(hr.height) if hr else 0,
        })
        print(f"  {p.name}: preview {pil.width}x{pil.height}, "
              f"boxes={len(boxes)}, crop={'y' if crop else 'n'}, highres={'y' if highres_url else 'n'}")

    payload = {"generatedWith": "cull engine (force+dry-run)", "photos": fixture_photos}
    (out / "previews.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out / 'previews.json'} ({len(fixture_photos)} photos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
