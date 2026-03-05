"""
tune_params.py — Offline parameter tuning for the culling pipeline.

Reads a CSV produced by ``cull_photos.py --dump-scores`` (single session)
or ``eval_multi_session.py`` (multi-session with ``session`` column) and
replays the scoring + burst-group TopN logic with different parameter
combinations.  No image decoding or detection is needed — runs in < 1 s.

Usage
-----
    # Single-session CSV:
    python tune_params.py --scores scores.csv

    # Multi-session CSV (must have ``session`` column):
    python tune_params.py --scores scores_multi_v2.csv

    # Custom search grid:
    python tune_params.py --scores scores_multi_v2.csv \
        --sharp-thresh 0.10 0.15 0.20 \
        --w-sharp 1.5 2.0 2.5 3.0 \
        --w-comp 2.5 3.0 3.5 4.0 \
        --min-raw 3.5 3.8 4.0 4.2 \
        --top-n 5 7 9
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Row:
    """One image's intermediate scores, loaded from CSV."""

    filename: str
    session: str  # session name (empty string if single-session CSV)
    s_sharp: float
    s_comp: float
    n_detections: int
    burst_group: int  # raw burst_group from CSV (per-session numbering)
    group_key: str  # unique key: "session:burst_group" to avoid cross-session collisions
    has_arw: bool  # ground truth


def load_csv(path: Path) -> list[Row]:
    """Read the scores CSV into a list of Row objects.

    Handles both single-session CSVs (no ``session`` column) and
    multi-session CSVs (with ``session`` column).  Constructs a unique
    ``group_key`` from ``(session, burst_group)`` to avoid cross-session
    burst_group ID collisions.
    """
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_session = "session" in (reader.fieldnames or [])
        for r in reader:
            session = r.get("session", "") if has_session else ""
            burst_group = int(r["burst_group"])
            group_key = f"{session}:{burst_group}"
            rows.append(Row(
                filename=r["filename"],
                session=session,
                s_sharp=float(r["s_sharp"]),
                s_comp=float(r["s_comp"]),
                n_detections=int(r["n_detections"]),
                burst_group=burst_group,
                group_key=group_key,
                has_arw=bool(int(r["has_arw"])),
            ))
    return rows


# ---------------------------------------------------------------------------
# Rating mapping (mirrors cull/scorer.py)
# ---------------------------------------------------------------------------

_RATING_BREAKS_DEFAULT = [4.25, 4.50, 4.75, 5.00]


def _raw_to_rating(raw: float, breaks: list[float] = _RATING_BREAKS_DEFAULT) -> int:
    for stars, threshold in enumerate(breaks, start=1):
        if raw < threshold:
            return stars
    return 5


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    """Result of one simulation with a specific parameter set."""

    tp: int
    fp: int
    fn: int
    tn: int
    n_keep: int
    n_reject: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def simulate(
    rows: list[Row],
    *,
    sharp_thresh: float = 0.15,
    w_sharp: float = 2.0,
    w_comp: float = 3.5,
    top_n: int = 7,
    min_raw: float = 4.0,
    rating_breaks: list[float] | None = None,
) -> SimResult:
    """Replay scoring + TopN logic offline and compute metrics vs ground truth.

    Parameters
    ----------
    sharp_thresh : Minimum sharpness before veto.
    w_sharp, w_comp : Weights for raw = w_sharp * s_sharp + w_comp * s_comp.
    top_n : Max frames to keep per burst group.
    min_raw : Minimum raw score to avoid rejection (additional veto gate).
    rating_breaks : Custom breakpoints [b1, b2, b3, b4] for raw→rating mapping.
    """
    if rating_breaks is None:
        rating_breaks = _RATING_BREAKS_DEFAULT

    # --- Phase 1: per-image scoring (veto + raw + rating) --------------------
    # We store (index, raw_score, vetoed, group_key) for TopN phase
    scored: list[dict] = []
    for i, r in enumerate(rows):
        raw = w_sharp * r.s_sharp + w_comp * r.s_comp

        vetoed = False
        if r.n_detections == 0:
            vetoed = True
        elif r.s_sharp < sharp_thresh:
            vetoed = True
        elif raw < min_raw:
            vetoed = True

        rating = -1 if vetoed else _raw_to_rating(raw, rating_breaks)
        scored.append({
            "idx": i,
            "raw": raw,
            "rating": rating,
            "vetoed": vetoed,
            "group_key": r.group_key,
        })

    # --- Phase 2: burst-group TopN selection ---------------------------------
    # Group by group_key (session:burst_group) to avoid cross-session collisions
    groups: dict[str, list[dict]] = {}
    for s in scored:
        groups.setdefault(s["group_key"], []).append(s)

    for g_items in groups.values():
        candidates = [s for s in g_items if not s["vetoed"]]
        candidates.sort(key=lambda s: s["raw"], reverse=True)
        keep_set = {s["idx"] for s in candidates[:top_n]}

        for s in g_items:
            if not s["vetoed"] and s["idx"] not in keep_set:
                s["rating"] = -1

    # --- Phase 3: evaluate vs ground truth -----------------------------------
    tp = fp = fn = tn = 0
    n_keep = n_reject = 0
    for s in scored:
        predicted_keep = s["rating"] > 0
        actual_keep = rows[s["idx"]].has_arw

        if predicted_keep:
            n_keep += 1
        else:
            n_reject += 1

        if predicted_keep and actual_keep:
            tp += 1
        elif predicted_keep and not actual_keep:
            fp += 1
        elif not predicted_keep and actual_keep:
            fn += 1
        else:
            tn += 1

    return SimResult(tp=tp, fp=fp, fn=fn, tn=tn, n_keep=n_keep, n_reject=n_reject)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    rows: list[Row],
    *,
    sharp_thresholds: list[float],
    w_sharps: list[float],
    w_comps: list[float],
    top_ns: list[int],
    min_raws: list[float],
) -> list[tuple[dict, SimResult]]:
    """Run simulate() for every combination and return results sorted by F1."""
    results: list[tuple[dict, SimResult]] = []
    total = (len(sharp_thresholds) * len(w_sharps) * len(w_comps)
             * len(top_ns) * len(min_raws))
    count = 0

    for st in sharp_thresholds:
        for ws in w_sharps:
            for wc in w_comps:
                for tn in top_ns:
                    for mr in min_raws:
                        count += 1
                        params = {
                            "sharp_thresh": st,
                            "w_sharp": ws,
                            "w_comp": wc,
                            "top_n": tn,
                            "min_raw": mr,
                        }
                        result = simulate(
                            rows,
                            sharp_thresh=st,
                            w_sharp=ws,
                            w_comp=wc,
                            top_n=tn,
                            min_raw=mr,
                        )
                        results.append((params, result))

    # Sort by F1 descending, then by precision descending (tie-breaker)
    results.sort(key=lambda x: (x[1].f1, x[1].precision), reverse=True)

    print(f"\nGrid search complete: {total} combinations evaluated.\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tune_params",
        description=(
            "Offline parameter tuning for culling pipeline. "
            "Reads CSV from --dump-scores, replays scoring logic instantly."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores", type=Path, required=True,
        help="Path to scores CSV from cull_photos.py --dump-scores.",
    )
    parser.add_argument(
        "--sharp-thresh", type=float, nargs="+",
        default=[0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        help="Sharpness threshold values to try.",
    )
    parser.add_argument(
        "--w-sharp", type=float, nargs="+",
        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        help="Sharpness weight values to try.",
    )
    parser.add_argument(
        "--w-comp", type=float, nargs="+",
        default=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
        help="Composition weight values to try.",
    )
    parser.add_argument(
        "--top-n", type=int, nargs="+",
        default=[3, 5, 7, 9, 12],
        help="TopN values to try.",
    )
    parser.add_argument(
        "--min-raw", type=float, nargs="+",
        default=[3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5],
        help="Minimum raw score values to try (below → reject).",
    )
    parser.add_argument(
        "--show-top", type=int, default=30,
        help="Number of top results to display.",
    )
    parser.add_argument(
        "--sort-by", choices=["f1", "precision", "recall", "accuracy"],
        default="f1",
        help="Primary sort metric.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.scores.exists():
        print(f"ERROR: scores CSV not found: {args.scores}")
        return 1

    print(f"Loading scores from {args.scores} ...")
    rows = load_csv(args.scores)
    print(f"  {len(rows)} images loaded")

    n_keep_gt = sum(1 for r in rows if r.has_arw)
    n_disc_gt = len(rows) - n_keep_gt
    print(f"  Ground truth: {n_keep_gt} keep ({100*n_keep_gt/len(rows):.1f}%), "
          f"{n_disc_gt} discard ({100*n_disc_gt/len(rows):.1f}%)")

    sessions = set(r.session for r in rows)
    n_groups = len(set(r.group_key for r in rows))
    if sessions != {""}:
        print(f"  {len(sessions)} sessions: {sorted(sessions)}")
    print(f"  {n_groups} burst groups (unique across sessions)")

    # Count search space
    n_combos = (len(args.sharp_thresh) * len(args.w_sharp) * len(args.w_comp)
                * len(args.top_n) * len(args.min_raw))
    print(f"\nSearch grid: {n_combos} combinations")
    print(f"  sharp_thresh: {args.sharp_thresh}")
    print(f"  w_sharp:      {args.w_sharp}")
    print(f"  w_comp:       {args.w_comp}")
    print(f"  top_n:        {args.top_n}")
    print(f"  min_raw:      {args.min_raw}")

    results = grid_search(
        rows,
        sharp_thresholds=args.sharp_thresh,
        w_sharps=args.w_sharp,
        w_comps=args.w_comp,
        top_ns=args.top_n,
        min_raws=args.min_raw,
    )

    # Re-sort by user's chosen metric
    sort_key = {
        "f1": lambda x: (x[1].f1, x[1].precision),
        "precision": lambda x: (x[1].precision, x[1].f1),
        "recall": lambda x: (x[1].recall, x[1].f1),
        "accuracy": lambda x: (x[1].accuracy, x[1].f1),
    }[args.sort_by]
    results.sort(key=sort_key, reverse=True)

    # Display top results
    show = min(args.show_top, len(results))
    print(f"{'='*110}")
    print(f"Top {show} by {args.sort_by.upper()}")
    print(f"{'='*110}")
    print(f"{'#':>3}  {'sharp_t':>7}  {'w_sharp':>7}  {'w_comp':>6}  "
          f"{'top_n':>5}  {'min_raw':>7}  "
          f"{'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Acc':>6}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
          f"{'Keep':>5}  {'Rej':>5}")
    print(f"{'-'*110}")

    for rank, (params, r) in enumerate(results[:show], start=1):
        print(f"{rank:3d}  "
              f"{params['sharp_thresh']:7.2f}  "
              f"{params['w_sharp']:7.2f}  "
              f"{params['w_comp']:6.2f}  "
              f"{params['top_n']:5d}  "
              f"{params['min_raw']:7.2f}  "
              f"{r.precision:6.3f}  "
              f"{r.recall:6.3f}  "
              f"{r.f1:6.4f}  "
              f"{r.accuracy:6.3f}  "
              f"{r.tp:4d}  {r.fp:4d}  {r.fn:4d}  {r.tn:4d}  "
              f"{r.n_keep:5d}  {r.n_reject:5d}")

    # Also show baseline for comparison
    print(f"\n{'='*110}")
    print("Baseline (current defaults: sharp_thresh=0.15, w_sharp=2.0, w_comp=3.5, top_n=7, min_raw=4.0):")
    baseline = simulate(rows, sharp_thresh=0.15, w_sharp=2.0, w_comp=3.5, top_n=7, min_raw=4.0)
    print(f"  Precision={baseline.precision:.3f}  Recall={baseline.recall:.3f}  "
          f"F1={baseline.f1:.4f}  Acc={baseline.accuracy:.3f}  "
          f"TP={baseline.tp}  FP={baseline.fp}  FN={baseline.fn}  TN={baseline.tn}  "
          f"Keep={baseline.n_keep}  Reject={baseline.n_reject}")

    # Also show per-session breakdown for multi-session data
    sessions = set(r.session for r in rows)
    if sessions != {""}:
        print(f"\n{'='*110}")
        print("Per-session breakdown (baseline params):")
        print(f"{'='*110}")
        print(f"  {'Session':<18} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
              f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
              f"{'Keep':>5}  {'Rej':>5}  {'Total':>5}")
        print(f"  {'-'*100}")
        for sess in sorted(sessions):
            sess_rows = [r for r in rows if r.session == sess]
            sess_result = simulate(sess_rows,
                                   sharp_thresh=0.15, w_sharp=2.0,
                                   w_comp=3.5, top_n=7, min_raw=4.0)
            print(f"  {sess:<18} "
                  f"{sess_result.precision:6.3f}  {sess_result.recall:6.3f}  "
                  f"{sess_result.f1:6.4f}  "
                  f"{sess_result.tp:4d}  {sess_result.fp:4d}  "
                  f"{sess_result.fn:4d}  {sess_result.tn:4d}  "
                  f"{sess_result.n_keep:5d}  {sess_result.n_reject:5d}  "
                  f"{len(sess_rows):5d}")

    # Show best result's per-session breakdown
    if sessions != {""} and results:
        best_params = results[0][0]
        print(f"\n{'='*110}")
        print(f"Per-session breakdown (BEST params: "
              f"sharp_thresh={best_params['sharp_thresh']:.2f}, "
              f"w_sharp={best_params['w_sharp']:.1f}, "
              f"w_comp={best_params['w_comp']:.1f}, "
              f"top_n={best_params['top_n']}, "
              f"min_raw={best_params['min_raw']:.1f}):")
        print(f"{'='*110}")
        print(f"  {'Session':<18} {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
              f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  "
              f"{'Keep':>5}  {'Rej':>5}  {'Total':>5}")
        print(f"  {'-'*100}")
        for sess in sorted(sessions):
            sess_rows = [r for r in rows if r.session == sess]
            sess_result = simulate(sess_rows, **best_params)
            print(f"  {sess:<18} "
                  f"{sess_result.precision:6.3f}  {sess_result.recall:6.3f}  "
                  f"{sess_result.f1:6.4f}  "
                  f"{sess_result.tp:4d}  {sess_result.fp:4d}  "
                  f"{sess_result.fn:4d}  {sess_result.tn:4d}  "
                  f"{sess_result.n_keep:5d}  {sess_result.n_reject:5d}  "
                  f"{len(sess_rows):5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
