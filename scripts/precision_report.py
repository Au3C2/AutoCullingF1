"""End-to-end precision report: default (DML) backend vs the deterministic
truth (tests/baselines/deterministic.json), per gate format.

Reports per format: rating flips (excluding KNOWN_RATING_DIVERGENCE),
raw_score drift distribution (excluding KNOWN_CUT_BOUNDARY).
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# A `tests` regular package installed in site-packages shadows the repo's
# namespace dir regardless of sys.path order — load by file path instead.
_conftest = _load_module("repo_conftest", ROOT / "tests" / "conftest.py")
_score_gate = _load_module("repo_score_gate", ROOT / "tests" / "score_gate.py")
SOURCE_DIRS = _conftest.SOURCE_DIRS
KNOWN_RATING_DIVERGENCE = _conftest.KNOWN_RATING_DIVERGENCE
KNOWN_CUT_BOUNDARY = _conftest.KNOWN_CUT_BOUNDARY
load_deterministic_baseline = _conftest.load_deterministic_baseline
baseline_section = _conftest.baseline_section
run_cull_on_copies = _score_gate.run_cull_on_copies


def main() -> None:
    baseline = load_deterministic_baseline()
    for key in ("jpg", "heif", "arw", "nef"):
        truth = baseline_section(key, baseline)
        src = [(ROOT / SOURCE_DIRS[key] / name) for name in truth]
        missing = [p.name for p in src if not p.exists()]
        if missing:
            print(f"[{key}] missing files: {missing}")
            continue
        scores = run_cull_on_copies(src, workers=4)
        flips, cut_boundary = [], []
        drifts = []
        for name, row in scores.items():
            rating, raw = row[0], row[1]
            exp_rating, exp_raw = truth[name]
            if (key, name) in KNOWN_RATING_DIVERGENCE:
                flips.append(f"{name} (known knife-edge: {rating} vs {exp_rating})")
                continue
            if (key, name) in KNOWN_CUT_BOUNDARY:
                cut_boundary.append(name)
                continue
            if rating != exp_rating:
                flips.append(f"{name}: {rating} vs truth {exp_rating}")
            drifts.append(abs(raw - exp_raw))
        drifts.sort()
        n = len(drifts)
        print(f"\n[{key}] n={len(scores)} files (cut-boundary skipped: {len(cut_boundary)})")
        print(f"  rating flips: {len(flips)}" + (f"  -> {flips}" if flips else ""))
        if drifts:
            print(f"  raw |drift|: max {drifts[-1]:.4f}  p95 {drifts[int(0.95 * n) - 1]:.4f}  "
                  f"median {drifts[n // 2]:.4f}")


if __name__ == "__main__":
    main()
